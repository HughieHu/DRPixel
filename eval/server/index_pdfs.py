"""Index your own PDFs for the `Page` channel.

Renders every page with PyMuPDF at 150 dpi and embeds it with Qwen3-VL-Embedding-8B the way the
DRPixelBench pages were embedded: instruction "Represent the user's input.", the image capped at
602,112 pixels, last-token pooling, L2-normalised, stored as fp16. Writes a directory with the layout
of HughieHu/drpixel-bench, which server/search_server.py serves unchanged:

    python server/index_pdfs.py --pdfs my_papers/ --out my_data
    DATA_DIR=my_data SERVER_PARADIGMS=page PAPER_COMPAT=0 python server/search_server.py

Needs a GPU. `--backend vllm` (default; how most of the benchmark's pages were embedded; needs
`pip install vllm`) or `--backend transformers` (the encoder code in this directory, slower). A
paper's doc id is its file name without `.pdf`. As in the benchmark, only the pages up to the
references heading are indexed; `--pages all` indexes every page.
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3-VL-Embedding-8B"
INSTRUCTION = "Represent the user's input."
MAX_PIXELS, MIN_PIXELS, DPI = 602112, 4 * 32 * 32, 150
REF_HEADS = ("references", "reference", "bibliography")
PAGES_PER_SHARD = 2500


def body_pages(doc):
    """Page indices up to and including the page where the references heading appears."""
    for i in range(len(doc)):
        for line in doc[i].get_text("text").splitlines():
            if line.strip().lower().rstrip(":") in REF_HEADS:
                return list(range(i + 1))
    return list(range(len(doc)))


def render(path, pages="body"):
    """[(page_idx, png_bytes, PIL image)] for the pages to index."""
    import fitz
    from PIL import Image
    out = []
    with fitz.open(path) as doc:
        for pi in (body_pages(doc) if pages == "body" else range(len(doc))):
            pix = doc[pi].get_pixmap(dpi=DPI)
            out.append((pi, pix.tobytes("png"), Image.frombytes("RGB", [pix.width, pix.height], pix.samples)))
    return out


def doc_ids(paths):
    """File name without .pdf, made unique and safe for the index."""
    ids, seen = [], set()
    for p in paths:
        base = re.sub(r"[^\w.\-]+", "_", os.path.splitext(os.path.basename(p))[0]) or "doc"
        d, n = base, 1
        while d in seen:
            n += 1
            d = "%s_%d" % (base, n)
        seen.add(d)
        ids.append(d)
    return ids


class Shards:
    """PNG pages into corpus/pages/shard-NNNNN.tar, recording where each image's bytes start."""

    def __init__(self, out):
        self.dir = os.path.join(out, "corpus", "pages")
        os.makedirs(self.dir, exist_ok=True)
        self.k, self.tf, self.n, self.pending, self.entries = -1, None, PAGES_PER_SHARD, [], []

    def add(self, row, doc, block_row, page_idx, png):
        if self.n >= PAGES_PER_SHARD:
            self.close()
            self.k += 1
            self.tf = tarfile.open(self._path(self.k), "w")
            self.n = 0
        ti = tarfile.TarInfo("%s/%d.png" % (doc, page_idx))
        ti.size = len(png)
        self.tf.addfile(ti, io.BytesIO(png))
        self.pending.append({"row": row, "doc_id": doc, "block_row": block_row, "page_idx": page_idx})
        self.n += 1

    def close(self):
        if self.tf is None:
            return
        self.tf.close()
        with tarfile.open(self._path(self.k)) as tf:
            members = tf.getmembers()
        for e, m in zip(self.pending, members):
            self.entries.append(dict(e, shard="corpus/pages/shard-%05d.tar" % self.k, offset=m.offset_data, size=m.size))
        self.tf, self.pending = None, []

    def _path(self, k):
        return os.path.join(self.dir, "shard-%05d.tar" % k)


def vllm_encoder(model):
    from transformers import AutoProcessor
    from vllm import LLM
    prompt = AutoProcessor.from_pretrained(model).apply_chat_template(
        [{"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
         {"role": "user", "content": [{"type": "image"}]}], tokenize=False, add_generation_prompt=True)
    llm = LLM(model=model, runner="pooling", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.55,
              enforce_eager=True, mm_processor_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS})

    def encode(images):
        outs = llm.embed([{"prompt": prompt, "multi_modal_data": {"image": im}} for im in images])
        return np.array([o.outputs.embedding for o in outs], dtype=np.float32)
    return encode


def transformers_encoder(model, device="cuda"):
    import torch
    sys.path.insert(0, HERE)
    from qwen3_vl_embedding import Qwen3VLEmbedder
    emb = Qwen3VLEmbedder(model, device=device, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                          max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS, default_instruction=INSTRUCTION)

    def encode(images):
        with torch.inference_mode():
            return emb.process([{"image": im} for im in images]).float().cpu().numpy()
    return encode


def build(pdfs, out, encode, pages="body", corpus="local", batch=32, log=print):
    """Render, embed and write the dataset; returns the number of indexed pages."""
    os.makedirs(os.path.join(out, "index", "page"), exist_ok=True)
    shards, rows, vecs, buf = Shards(out), [], [], []
    t0 = time.time()

    def flush():
        if buf:
            E = encode(buf)
            vecs.append((E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)).astype(np.float16))
            buf.clear()

    for n, (path, doc) in enumerate(zip(pdfs, doc_ids(pdfs))):
        try:
            pages_ = render(path, pages)
        except Exception as e:
            log("  skipped %s: %s" % (path, str(e)[:120]))
            continue
        for block_row, (pi, png, im) in enumerate(pages_):
            shards.add(len(rows), doc, block_row, pi, png)
            rows.append([doc, corpus, block_row])
            buf.append(im)
            if len(buf) >= batch:
                flush()
        if (n + 1) % 20 == 0:
            log("  %d/%d PDFs, %d pages, %.1f pages/s" % (n + 1, len(pdfs), len(rows), len(rows) / (time.time() - t0)))
    flush()
    shards.close()
    if not rows:
        raise SystemExit("no pages were indexed")
    np.save(os.path.join(out, "index", "page", "embeddings.npy"), np.concatenate(vecs))
    json.dump(rows, open(os.path.join(out, "index", "page", "rows.json"), "w"))
    with gzip.open(os.path.join(out, "corpus", "pages", "index.jsonl.gz"), "wt") as f:
        for e in shards.entries:
            f.write(json.dumps(e) + "\n")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="index_pdfs")
    ap.add_argument("--pdfs", required=True, help="a directory of PDFs (searched recursively) or one PDF")
    ap.add_argument("--out", required=True, help="the data directory to write (DATA_DIR for the server)")
    ap.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    ap.add_argument("--pages", choices=("body", "all"), default="body")
    ap.add_argument("--corpus", default="local", help="corpus label returned with every hit")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    if os.path.isdir(a.pdfs):
        pdfs = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.pdfs) for f in fs if f.lower().endswith(".pdf"))
    else:
        pdfs = [a.pdfs]
    if not pdfs:
        raise SystemExit("no PDFs under %s" % a.pdfs)
    print("%d PDFs -> %s  (backend %s, %s pages)" % (len(pdfs), a.out, a.backend, a.pages), flush=True)
    encode = vllm_encoder(a.model) if a.backend == "vllm" else transformers_encoder(a.model)
    t0 = time.time()
    n = build(pdfs, a.out, encode, a.pages, a.corpus, a.batch, log=lambda s: print(s, flush=True))
    print("indexed %d pages in %.0fs. Serve them with:\n  DATA_DIR=%s SERVER_PARADIGMS=page PAPER_COMPAT=0 "
          "python server/search_server.py" % (n, time.time() - t0, a.out), flush=True)


if __name__ == "__main__":
    main()
