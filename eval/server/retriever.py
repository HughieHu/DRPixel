"""Retrieval over the released DRPixelBench data.

Reads a local copy of the `HughieHu/drpixel-bench` dataset (only `index/` and `corpus/pages/` are
needed) and serves the two released channels:

    make("page").search(query, k)  ->  [{doc, corpus, score, page_idx, image}]   image = PNG bytes
    make("text").search(query, k)  ->  [{doc, corpus, score, chunk_idx, text}]   text <= 2,200 chars

Queries are encoded and scored exactly as in the paper's runs: Qwen3-VL-Embedding-8B for pages and
Qwen3-Embedding-8B for passages, cosine similarity against the released embeddings, top-k.

`paper_compat.json` reproduces how the paper's server delivered some hits: for documents it could
not resolve, a page hit carries no page number or image and a passage hit carries empty text; for
documents without a source PDF, a page hit carries a blank page; and a page search whose results
include a document with missing source files fails as a whole. Set PAPER_COMPAT=0 to serve every
hit from the released data instead.
"""
import gzip
import io
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_MAX_CHARS = 2200        # just above the 90th percentile of passage length (2,199); the only cut


def _compat():
    """(unresolved docs, docs without a PDF, docs that fail a page search, {row: page_idx or None})."""
    if os.environ.get("PAPER_COMPAT", "1") == "0":
        return set(), set(), set(), {}
    c = json.load(open(os.path.join(HERE, "paper_compat.json")))
    return (set(c["unresolved_docs"]), set(c["no_pdf_docs"]), set(c.get("error_docs", [])),
            {int(k): v for k, v in c.get("page_rows", {}).items()})


def _blank_page():
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (1024, 1400), "white").save(b, "PNG")
    return b.getvalue()


# ------------------------------------------------------------------ query encoders
def _last_token_pool(h, mask):
    if mask[:, -1].sum() == mask.shape[0]:          # left padding
        return h[:, -1]
    lens = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), lens]


class TextQueryEncoder:
    """Qwen3-Embedding-8B with its retrieval instruction; last-token pooling, L2-normalised."""
    INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, device, model="Qwen/Qwen3-Embedding-8B", max_length=4096):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side="left")
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to(device).eval()
        self.device, self.max_length = device, max_length

    def encode(self, q):
        text = "Instruct: %s\nQuery: %s" % (self.INSTRUCT, q)
        with torch.inference_mode():
            enc = self.tok([text], padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            e = torch.nn.functional.normalize(_last_token_pool(h, enc["attention_mask"]), p=2, dim=1)
        return e[0].float().cpu().numpy()


class PageQueryEncoder:
    """Qwen3-VL-Embedding-8B, text query into the page-image space (the model's own embedder)."""

    def __init__(self, device, model="Qwen/Qwen3-VL-Embedding-8B"):
        from qwen3_vl_embedding import Qwen3VLEmbedder
        self.emb = Qwen3VLEmbedder(model, device=device, torch_dtype=torch.bfloat16,
                                   attn_implementation="sdpa")
        self.device = self.emb.model.device
        # The embedder replaces the query with "NULL" if preprocessing raises (e.g. an old
        # qwen-vl-utils); every search would then return the same pages. Fail now instead.
        probe = "page-encoder self-check"
        ids = self.emb._preprocess_inputs([self.emb.format_model_input(text=probe)])["input_ids"][0]
        if probe not in self.emb.processor.tokenizer.decode(ids):
            raise RuntimeError("query preprocessing failed; install qwen-vl-utils>=0.0.13")

    def encode(self, q):
        inp = self.emb._preprocess_inputs([self.emb.format_model_input(text=q)])
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.inference_mode():
            o = self.emb.forward(inp)
            e = torch.nn.functional.normalize(
                self.emb._pooling_last(o["last_hidden_state"], o["attention_mask"]), p=2, dim=-1)
        return e[0].float().cpu().numpy()


# ------------------------------------------------------------------ retrievers
def _topk(X, qvec, k):
    q = torch.as_tensor(np.asarray(qvec), dtype=X.dtype, device=X.device).reshape(1, -1)
    q = q / (q.float().norm() + 1e-9).to(q.dtype)
    scores = (X @ q.T).squeeze(1)
    top = torch.topk(scores, min(k, scores.numel()))
    return list(zip(top.indices.tolist(), (float(v) for v in top.values)))


class PageRetriever:
    def __init__(self, data_dir, device="cuda", encoder=None):
        self.enc = encoder or PageQueryEncoder(device)
        d = os.path.join(data_dir, "index", "page")
        X = np.load(os.path.join(d, "embeddings.npy"))
        self.rows = json.load(open(os.path.join(d, "rows.json")))
        if X.shape[0] != len(self.rows):
            raise ValueError("page embeddings (%d rows) and rows.json (%d) disagree" % (X.shape[0], len(self.rows)))
        self.X = torch.from_numpy(X).to(device)
        # row -> where its PNG is (corpus/pages/index.jsonl.gz, one line per row in row order)
        pdir = os.path.join(data_dir, "corpus", "pages")
        n = len(self.rows)
        self.page_idx = np.zeros(n, np.int32)
        self.shard = np.zeros(n, np.int32)
        self.offset = np.zeros(n, np.int64)
        self.size = np.zeros(n, np.int64)
        shards = {}
        with gzip.open(os.path.join(pdir, "index.jsonl.gz"), "rt") as f:
            for line in f:
                e = json.loads(line)
                i = e["row"]
                if e["doc_id"] != self.rows[i][0]:
                    raise ValueError("pages index row %d is %s, embeddings row is %s" % (i, e["doc_id"], self.rows[i][0]))
                self.page_idx[i] = e["page_idx"]
                self.shard[i] = shards.setdefault(os.path.basename(e["shard"]), len(shards))
                self.offset[i], self.size[i] = e["offset"], e["size"]
        self.shard_paths = [os.path.join(pdir, s) for s, _ in sorted(shards.items(), key=lambda x: x[1])]
        self.fds = {}
        self.unresolved, self.no_pdf, self.error_docs, self.page_rows = _compat()
        self.blank = _blank_page() if self.no_pdf else None
        # an overridden row may point at another page of the same paper; find that page's image
        docs = {self.rows[r][0] for r in self.page_rows}
        self.row_of_page = {(self.rows[i][0], int(self.page_idx[i])): i for i in range(n) if self.rows[i][0] in docs}

    def image(self, i):
        if self.size[i] == 0:
            return None
        s = int(self.shard[i])
        if s not in self.fds:
            self.fds[s] = os.open(self.shard_paths[s], os.O_RDONLY)
        return os.pread(self.fds[s], int(self.size[i]), int(self.offset[i]))

    def search(self, query, k=3):
        top = _topk(self.X, self.enc.encode(query), k)
        bad = [self.rows[i][0] for i, _ in top if self.rows[i][0] in self.error_docs]
        if bad:
            raise FileNotFoundError("source files missing for %s" % bad[0])
        out = []
        for i, score in top:
            doc, corpus, _ = self.rows[i]
            r = {"doc": doc, "corpus": corpus, "score": round(score, 4)}
            if doc in self.unresolved:
                out.append(r); continue
            p = self.page_rows.get(i, int(self.page_idx[i]))
            if p is not None:
                r["page_idx"] = p
                j = i if p == self.page_idx[i] else self.row_of_page.get((doc, p))
                r["image"] = self.blank if doc in self.no_pdf else (self.image(j) if j is not None else None)
            out.append(r)
        return out


class TextRetriever:
    def __init__(self, data_dir, device="cuda", encoder=None):
        self.enc = encoder or TextQueryEncoder(device)
        d = os.path.join(data_dir, "index", "text")
        self.rows = json.load(open(os.path.join(d, "rows.json")))
        Xm = np.load(os.path.join(d, "embeddings.npy"), mmap_mode="r")
        if Xm.shape[0] != len(self.rows):
            raise ValueError("text embeddings (%d rows) and rows.json (%d) disagree" % (Xm.shape[0], len(self.rows)))
        # Keep the index on the GPU in fp16 when it fits next to the encoder; otherwise on the CPU in
        # fp32. TEXT_INDEX_DEVICE=cuda|cpu forces the choice.
        want = os.environ.get("TEXT_INDEX_DEVICE")
        if not want:
            try:
                free = torch.cuda.mem_get_info(torch.device(device))[0]
            except Exception:
                free = 0
            want = device if free > Xm.shape[0] * Xm.shape[1] * 2 + (6 << 30) else "cpu"
        X = torch.empty(Xm.shape, dtype=torch.float32 if want == "cpu" else torch.float16, device=want)
        for i in range(0, Xm.shape[0], 1 << 16):
            j = min(i + (1 << 16), Xm.shape[0])
            X[i:j] = torch.from_numpy(np.array(Xm[i:j])).to(X.dtype)
        self.X = X
        self.off = np.load(os.path.join(d, "passages.offsets.npy"))
        self.txt = np.memmap(os.path.join(d, "passages.bin"), dtype=np.uint8, mode="r")
        if not (len(self.off) == len(self.rows) + 1 and int(self.off[-1]) == self.txt.shape[0]):
            raise ValueError("passages.offsets.npy does not match passages.bin and rows.json")
        self.unresolved = _compat()[0]

    def search(self, query, k=5):
        out = []
        for i, score in _topk(self.X, self.enc.encode(query), k):
            doc, corpus, chunk_idx = self.rows[i]
            text = "" if doc in self.unresolved else bytes(self.txt[self.off[i]:self.off[i + 1]]).decode("utf-8")
            out.append({"doc": doc, "corpus": corpus, "score": round(score, 4),
                        "chunk_idx": int(chunk_idx), "text": text[:TEXT_MAX_CHARS]})
        return out


def make(paradigm, device="cuda", data_dir=None, encoder=None):
    data_dir = data_dir or os.environ.get("DATA_DIR", "")
    if not data_dir:
        raise ValueError("set DATA_DIR to a local copy of HughieHu/drpixel-bench")
    if paradigm == "page":
        return PageRetriever(data_dir, device, encoder)
    if paradigm == "text":
        return TextRetriever(data_dir, device, encoder)
    raise ValueError("unknown paradigm %r (released: page, text)" % paradigm)
