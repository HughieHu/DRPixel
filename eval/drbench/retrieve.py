"""Retrieval client — talks to the search endpoint at SEARCH_URL.

The client needs no index, no PDFs and no GPU; those live behind the endpoint
(server/search_server.py).

    from drbench.retrieve import Retriever
    r = Retriever("page")                  # or "text" / "figure"
    hits = r.search("query text", k=5)
    # text   -> {doc, corpus, score, chunk_idx, text}
    # page   -> {doc, corpus, score, page_idx, image}      image = PNG bytes
    # figure -> {doc, corpus, score, page_idx, hash, image}

The server sends images as base64 (`image_b64`); we decode to bytes under `image` so the harness
sees exactly the same hit dict as the in-process retriever used for the paper's numbers.
"""
import base64
import gzip
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("SEARCH_URL", "").rstrip("/")
PARADIGMS = ("text", "page", "figure")


class SearchError(RuntimeError):
    pass


class Retriever:
    def __init__(self, paradigm, url=None, timeout=180, retries=4):
        if paradigm not in PARADIGMS:
            raise ValueError("paradigm must be one of %s, got %r" % (PARADIGMS, paradigm))
        self.paradigm = paradigm
        self.url = (url or DEFAULT_URL).rstrip("/")
        if not self.url:
            raise SearchError(
                "No search endpoint. Set SEARCH_URL (see .env.example), or run your own:\n"
                "  python server/search_server.py   # needs the corpus + index")
        self.timeout = timeout
        self.retries = retries

    def health(self):
        req = urllib.request.Request(self.url + "/health")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())

    def search(self, query, k=5):
        body = json.dumps({"paradigm": self.paradigm, "query": query, "k": k}).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url + "/search", data=body,
                    headers={"Content-Type": "application/json", "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if (resp.headers.get("Content-Encoding") or "") == "gzip":
                        raw = gzip.decompress(raw)
                    d = json.loads(raw)
                return [_decode(h) for h in d.get("hits", [])]
            except urllib.error.HTTPError as e:
                try:
                    last = json.loads(e.read()).get("error") or str(e)
                except Exception:
                    last = str(e)
            except Exception as e:
                last = "%s: %s" % (type(e).__name__, str(e)[:200])
            time.sleep(2 * (attempt + 1))
        # Raise. Returning [] here would make a broken endpoint look like "nothing was found":
        # the run exits 0, every question gets answered, every search is empty, and the numbers
        # look real but are garbage.
        raise SearchError("search failed after %d attempts (%s): %s"
                          % (self.retries, self.paradigm, last))


def _decode(h):
    d = dict(h)
    b64 = d.pop("image_b64", None)
    if b64:
        d["image"] = base64.b64decode(b64)
    return d
