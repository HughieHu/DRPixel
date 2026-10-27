"""The released retriever on a tiny dataset in the drpixel-bench layout, with a fixed query vector."""
import gzip
import io
import json
import os
import sys
import tarfile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("PIL")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "server"))
import retriever  # noqa: E402

DIM = 4
PAGE_ROWS = [["d1", "drpixel", 0], ["d1", "drpixel", 1], ["d2", "airqa", 0], ["d3", "drpixel", 0]]
PAGE_IDX = [0, 2, 0, 5]                        # d1's body pages skip PDF page 1
TEXT_ROWS = [["d1", "drpixel", 0], ["d2", "airqa", 0], ["d3", "drpixel", 0]]
TEXTS = ["first passage", "x" * 3000, "third passage"]


class FixedEncoder:
    def __init__(self, v):
        self.v = np.asarray(v, np.float32)

    def encode(self, q):
        return self.v


def _unit(rows):
    X = np.asarray(rows, np.float32)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float16)


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    (d / "index" / "page").mkdir(parents=True)
    (d / "index" / "text").mkdir(parents=True)
    (d / "corpus" / "pages").mkdir(parents=True)
    np.save(d / "index" / "page" / "embeddings.npy", _unit([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.3, 1, 0, 0], [0, 0, 1, 0]]))
    json.dump(PAGE_ROWS, open(d / "index" / "page" / "rows.json", "w"))
    shard = d / "corpus" / "pages" / "shard-00000.tar"
    pngs = [b"\x89PNG\r\n\x1a\n" + bytes([i]) * 10 for i in range(4)]
    with tarfile.open(shard, "w") as tf:
        for i, b in enumerate(pngs):
            ti = tarfile.TarInfo("%s/%d.png" % (PAGE_ROWS[i][0], PAGE_IDX[i])); ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    with tarfile.open(shard) as tf, gzip.open(d / "corpus" / "pages" / "index.jsonl.gz", "wt") as f:
        for i, m in enumerate(tf.getmembers()):
            f.write(json.dumps({"row": i, "doc_id": PAGE_ROWS[i][0], "block_row": PAGE_ROWS[i][2], "page_idx": PAGE_IDX[i],
                                "shard": "corpus/pages/shard-00000.tar", "offset": m.offset_data, "size": m.size}) + "\n")
    np.save(d / "index" / "text" / "embeddings.npy", _unit([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    json.dump(TEXT_ROWS, open(d / "index" / "text" / "rows.json", "w"))
    blob = b"".join(t.encode() for t in TEXTS)
    open(d / "index" / "text" / "passages.bin", "wb").write(blob)
    np.save(d / "index" / "text" / "passages.offsets.npy", np.cumsum([0] + [len(t.encode()) for t in TEXTS]))
    return str(d), pngs


def _compat(tmp_path, monkeypatch, **c):
    c = {"unresolved_docs": [], "no_pdf_docs": [], **c}
    json.dump(c, open(tmp_path / "paper_compat.json", "w"))
    monkeypatch.setattr(retriever, "HERE", str(tmp_path))
    monkeypatch.setenv("PAPER_COMPAT", "1")


def test_page_ranking_and_images(data, monkeypatch):
    d, pngs = data
    monkeypatch.setenv("PAPER_COMPAT", "0")
    r = retriever.make("page", "cpu", d, FixedEncoder([1, 0, 0, 0]))
    hits = r.search("q", k=3)
    assert [(h["doc"], h["page_idx"]) for h in hits] == [("d1", 0), ("d1", 2), ("d2", 0)]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert [h["image"] for h in hits] == [pngs[0], pngs[1], pngs[2]]
    assert hits[2]["corpus"] == "airqa"


def test_text_ranking_and_cap(data, monkeypatch):
    d, _ = data
    monkeypatch.setenv("PAPER_COMPAT", "0")
    monkeypatch.setenv("TEXT_INDEX_DEVICE", "cpu")
    r = retriever.make("text", "cpu", d, FixedEncoder([0, 1, 0, 0]))
    h = r.search("q", k=2)[0]
    assert (h["doc"], h["chunk_idx"]) == ("d2", 0)
    assert len(h["text"]) == retriever.TEXT_MAX_CHARS == 2200


def test_paper_compat(data, tmp_path, monkeypatch):
    d, pngs = data
    _compat(tmp_path, monkeypatch, unresolved_docs=["d2"], no_pdf_docs=["d3"], page_rows={"1": None})
    r = retriever.make("page", "cpu", d, FixedEncoder([1, 1, 1, 0]))
    hits = {(h["doc"], h.get("page_idx")): h for h in r.search("q", k=4)}
    assert ("d2", None) in hits and "image" not in hits[("d2", None)]          # unresolved: no page, no image
    assert hits[("d3", 5)]["image"] != pngs[3] and hits[("d3", 5)]["image"].startswith(b"\x89PNG")   # no PDF: blank page
    assert ("d1", None) in hits and hits[("d1", 0)]["image"] == pngs[0]       # row 1 overridden to no page
    monkeypatch.setenv("TEXT_INDEX_DEVICE", "cpu")
    t = retriever.make("text", "cpu", d, FixedEncoder([0, 1, 0, 0]))
    assert t.search("q", k=1)[0]["text"] == ""                                # unresolved: empty passage
    _compat(tmp_path, monkeypatch, error_docs=["d3"])
    r = retriever.make("page", "cpu", d, FixedEncoder([0, 0, 1, 0]))
    with pytest.raises(FileNotFoundError):                                    # missing source files: search fails
        r.search("q", k=1)
