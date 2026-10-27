"""server/index_pdfs.py on two tiny PDFs with a stand-in encoder; the released retriever must serve the result."""
import gzip
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
fitz = pytest.importorskip("fitz")
pytest.importorskip("torch")
pytest.importorskip("PIL")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "server"))
import index_pdfs  # noqa: E402
import retriever  # noqa: E402


def _pdf(path, pages):
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        for k, line in enumerate(lines):
            page.insert_text((72, 72 + 20 * k), line)
    doc.save(path)
    doc.close()


class FixedEncoder:
    def __init__(self, v):
        self.v = np.asarray(v, np.float32)

    def encode(self, q):
        return self.v


@pytest.fixture
def pdfs(tmp_path):
    a, b = str(tmp_path / "paper one.pdf"), str(tmp_path / "paper2.pdf")
    _pdf(a, [["Introduction", "body text"], ["Results", "more"], ["References", "[1] a"], ["Appendix"]])
    _pdf(b, [["Only page"]])
    return [a, b]


def fake_encoder():
    """One axis per page, in indexing order, so a query vector picks out a page."""
    count = [0]

    def encode(images):
        E = np.zeros((len(images), 8), np.float32)
        for i in range(len(images)):
            E[i, count[0]] = 1.0
            count[0] += 1
        return E
    return encode


def test_body_pages_and_ids(pdfs):
    with fitz.open(pdfs[0]) as d:
        assert index_pdfs.body_pages(d) == [0, 1, 2]
    assert index_pdfs.doc_ids(["x/paper one.pdf", "y/paper one.pdf", "z.pdf"]) == ["paper_one", "paper_one_2", "z"]


def test_build_is_served_by_the_retriever(pdfs, tmp_path, monkeypatch):
    out = str(tmp_path / "data")
    n = index_pdfs.build(pdfs, out, fake_encoder(), pages="body", batch=2, log=lambda s: None)
    assert n == 4                                       # paper_one pages 0-2 (to References) + paper2 page 0
    rows = json.load(open(os.path.join(out, "index", "page", "rows.json")))
    assert rows == [["paper_one", "local", 0], ["paper_one", "local", 1], ["paper_one", "local", 2], ["paper2", "local", 0]]
    with gzip.open(os.path.join(out, "corpus", "pages", "index.jsonl.gz"), "rt") as f:
        assert [json.loads(l)["page_idx"] for l in f] == [0, 1, 2, 0]

    monkeypatch.setenv("PAPER_COMPAT", "0")
    v = np.zeros(8, np.float32)
    v[1] = 1.0                                          # paper_one, page 1
    hits = retriever.make("page", "cpu", out, FixedEncoder(v)).search("q", k=2)
    assert (hits[0]["doc"], hits[0]["page_idx"]) == ("paper_one", 1)
    with fitz.open(pdfs[0]) as d:
        assert hits[0]["image"] == d[1].get_pixmap(dpi=index_pdfs.DPI).tobytes("png")


def test_all_pages(pdfs, tmp_path):
    assert index_pdfs.build(pdfs, str(tmp_path / "all"), fake_encoder(), pages="all", log=lambda s: None) == 5
