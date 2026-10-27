"""Offline stand-ins for the two endpoints the client talks to: a search endpoint and an
OpenAI-compatible model endpoint. No corpus, GPU, model or API key needed.

They exercise the protocol hazards that matter: the model endpoint strips the stop string (as the
OpenAI API does), page hits carry real PNG bytes through base64, and text passages are long
enough (with an end marker) to show they arrive uncut.
"""
import base64
import json
import os
import re
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BENCH = json.load(open(os.path.join(os.path.dirname(__file__), "..", "eval", "bench", "eval500.json")))
BY_Q = {it["qid"]: it for it in BENCH}


def _png(w=8, h=8):
    """A minimal valid PNG, so the client's base64 decode has real bytes to carry."""
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


PNG = _png()


# --------------------------------------------------------------------------- fake search
class SearchH(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "paradigms": ["text", "page", "figure"], "max_k": 20,
                         "calls": 0, "errors": 0, "t": 0.0})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        par, q, k = req.get("paradigm"), (req.get("query") or ""), int(req.get("k", 5))
        # The mock "retrieves" the gold doc/page when the query mentions the qid token we plant,
        # so PgHit is deterministic and we can assert on it.
        m = re.search(r"QID:(\S+)", q)
        it = BY_Q.get(m.group(1)) if m else None
        hits = []
        for i in range(k):
            if it and i == 0:
                doc, page = it["doc"], it.get("page")
            else:
                doc, page = "0000.%05d" % i, i
            h = {"doc": doc, "corpus": "mock", "score": 1.0 - 0.1 * i}
            if par == "text":
                h["chunk_idx"] = i
                # ~1,800 chars with an end marker: the client must pass passages through uncut
                h["text"] = ("passage %d for %s " % (i, q[:40])) + "w" * 1750 + " END%d" % i
            elif par == "page":
                h["page_idx"] = page if page is not None else i
                h["image_b64"] = base64.b64encode(PNG).decode()
            else:
                h["page_idx"] = page if page is not None else i
                h["hash"] = "h%d" % i
                h["image_b64"] = base64.b64encode(PNG).decode()
            hits.append(h)
        self._send(200, {"hits": hits})

    def log_message(self, *a):
        pass


# --------------------------------------------------------------------------- fake model
STATE = {}
LOCK = threading.Lock()


class ModelH(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        # --- quirk route: behave like an endpoint that rejects `temperature` and renamed
        # `max_tokens`, to exercise the adapter's learning.
        if STATE.get("strict"):
            bad = None
            if "temperature" in req:
                bad = ("Unsupported value: 'temperature' does not support 0 with this model. "
                       "Only the default (1) is supported.")
            elif "max_tokens" in req:
                bad = ("Unsupported parameter: 'max_tokens' is not supported with this model. "
                       "Use 'max_completion_tokens' instead.")
            if bad:
                body = json.dumps({"error": {"message": bad, "type": "invalid_request_error"}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        msgs = req.get("messages") or []

        # Recover the question (and its qid) from the user turn.
        qtext = ""
        for m in msgs:
            c = m.get("content")
            s = c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in (c or []) if isinstance(p, dict))
            if "QUESTION:" in s:
                qtext = s.split("QUESTION:", 1)[1].strip()
        qid = next((k for k, v in BY_Q.items() if v["question"].strip()[:60] == qtext[:60]), None)
        # Count real observations only: a user turn whose first part is exactly "<tool_output>".
        # Substring-matching "tool_output" would also match the SYSTEM PROMPT (which documents the
        # block) and the retrieval-pushback message, which made the mock answer on turn 1.
        def _is_obs(m):
            c = m.get("content")
            if isinstance(c, str) or not c:
                return False
            return any(isinstance(p, dict) and p.get("text") == "<tool_output>" for p in c)

        n_obs = sum(1 for m in msgs if m.get("role") == "user" and _is_obs(m))
        forced = any("used all" in json.dumps(m.get("content"))[:4000] for m in msgs[-2:])

        if forced or n_obs >= 1:
            gold = BY_Q.get(qid, {}).get("gold_answer", "unknown")
            out = "<think>I have the page in front of me.</think><answer>%s</answer>" % gold[:300]
        else:
            out = ('<think>I need the page. Searching.</think>'
                   '<call_tool name="%s">QID:%s %s</call_tool>'
                   % (STATE["tool"], qid, qtext[:60]))

        # Behave like the OpenAI API: strip whatever stop string terminated generation.
        for s in (req.get("stop") or []):
            if out.endswith(s):
                out = out[: -len(s)]
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000 + 500 * n_obs, "completion_tokens": len(out) // 4},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def start(tool, strict=False):
    """Start both mocks on free ports; returns (search_url, model_url, stop)."""
    STATE.update(tool=tool, strict=strict)
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), h) for h in (SearchH, ModelH)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    urls = ["http://127.0.0.1:%d" % s.server_address[1] for s in servers]

    def stop():
        for s in servers:
            s.shutdown()
            s.server_close()
    return urls[0], urls[1] + "/v1", stop
