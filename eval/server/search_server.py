"""Search endpoint for the benchmark.

Serves the released retrieval channels over HTTP:

    POST /search  {"paradigm": "text"|"page", "query": str, "k": int}
    -> {"hits": [ {...}, ... ]}

Hit shape per paradigm (identical to the in-process retriever, so scores are comparable to the
paper):
    text    {doc, corpus, score, chunk_idx, text}            # text capped at 2,200 characters
    page    {doc, corpus, score, page_idx, image_b64}        # image_b64 = base64(PNG)

Why `image_b64` and not `image`: the retriever returns raw PNG *bytes* under `"image"`, and bytes
are not JSON. The server sends base64; the client decodes it back to bytes, so the harness sees the
same hit dict either way.

Images dominate the payload (~4k tokens per page at the default render settings), so responses
are gzipped when the client advertises it.

The server reads a local copy of the HughieHu/drpixel-bench dataset (see eval/README.md, §3) and
needs a GPU for the query encoders. The client needs none of this, only `SEARCH_URL`.

    DATA_DIR=/path/to/drpixel-bench python server/search_server.py

The figure channel of `Text_Fig` is not served: its index is not released yet.
"""
import base64
import gzip
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retriever                         # noqa: E402

PORT = int(os.environ.get("RETRIEVER_PORT", "19001"))
PARADIGMS = [p for p in os.environ.get("SERVER_PARADIGMS", "text,page").split(",") if p]
MAX_K = 20
# One GPU may not hold both encoders and both indices; PAGE_DEVICE / TEXT_DEVICE split them.
DEVICE = {"page": os.environ.get("PAGE_DEVICE", "cuda"), "text": os.environ.get("TEXT_DEVICE", "cuda")}

# Build each paradigm separately. One missing index must not take down the others — but it must
# be LOUD, and /health must then report only what is actually served, so a client asking for the
# page config fails at startup instead of getting empty hits for 500 questions.
R, BROKEN = {}, {}
for _p in PARADIGMS:
    try:
        R[_p] = retriever.make(_p, DEVICE.get(_p, "cuda"))
        print("  built %-8s ok" % _p, flush=True)
    except Exception as _e:
        BROKEN[_p] = "%s: %s" % (type(_e).__name__, str(_e)[:200])
        print("  !! %-8s FAILED -> %s" % (_p, BROKEN[_p]), flush=True)
if not R:
    sys.exit("no paradigm could be built; refusing to start: %s" % BROKEN)

LOCK = threading.Lock()                  # retrieval is a GPU op; serialize it
STATS = {"calls": 0, "errors": 0, "t": 0.0}

print("search_server READY paradigms=%s port=%d%s"
      % (list(R), PORT, (" broken=%s" % list(BROKEN)) if BROKEN else ""), flush=True)


def _encode_hits(hits):
    """Raw PNG bytes under 'image' -> base64 under 'image_b64'; everything else passes through."""
    out = []
    for h in hits:
        d = {k: v for k, v in h.items() if k != "image"}
        if h.get("image") is not None:
            d["image_b64"] = base64.b64encode(h["image"]).decode("ascii")
        out.append(d)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        hdrs = {"Content-Type": "application/json"}
        if "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 4096:
            body = gzip.compress(body, 1)
            hdrs["Content-Encoding"] = "gzip"
        self.send_response(code)
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, "paradigms": list(R), "broken": BROKEN,
                             "max_k": MAX_K, **STATS})
        else:
            self._send(404, {"error": "GET %s not found; POST /search" % self.path})

    def do_POST(self):
        if self.path.rstrip("/") != "/search":
            return self._send(404, {"error": "POST /search only"})
        t0 = time.time()
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            par = req.get("paradigm", "text")
            if par not in R:
                return self._send(400, {"error": "unknown paradigm %r; served: %s" % (par, list(R))})
            q = (req.get("query") or "").strip()
            if not q:
                return self._send(400, {"error": "empty query"})
            k = max(1, min(int(req.get("k", 5)), MAX_K))
            with LOCK:
                hits = R[par].search(q, k=k)
            STATS["calls"] += 1
            STATS["t"] += time.time() - t0
            self._send(200, {"hits": _encode_hits(hits)})
        except Exception as e:
            STATS["errors"] += 1
            # Return the reason: a silent empty hit list would look like "nothing found" and the
            # run would finish with every search empty.
            self._send(500, {"hits": [], "error": "%s: %s" % (type(e).__name__, str(e)[:300])})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
