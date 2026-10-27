"""THE place you change to evaluate your own model.

Any endpoint that speaks the OpenAI chat-completions API works — hosted or local, no code edit:

    export MODEL_NAME=gpt-4o
    export MODEL_BASE_URL=https://api.openai.com/v1
    export MODEL_API_KEY=sk-...

    # a local vLLM / sglang server:
    export MODEL_NAME=Qwen/Qwen3.6-27B
    export MODEL_BASE_URL=http://localhost:8000/v1
    export MODEL_API_KEY=EMPTY

The model must accept **interleaved images** (retrieved pages are sent as image parts). If yours
is text-only, use the `Text` config, which never sends an image.

If your model is not OpenAI-compatible, subclass `Client` and override `complete()`; that is the
only method that touches the wire.

----------------------------------------------------------------------------------------------
The harness uses only these names from this module: AGENT_MODEL, text_part, img_bytes, mcall,
q_reset, q_get (eval.py and doctor.py also read BASE_URL). Keep that surface if you replace it.
"""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request

from . import envfloat, envint, ssl_context

AGENT_MODEL = os.environ.get("MODEL_NAME", "")
BASE_URL = os.environ.get("MODEL_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("MODEL_API_KEY", "EMPTY")
MAX_TOKENS = envint("MODEL_MAX_TOKENS", 8000)    # every reported run used 8000
TEMPERATURE = envfloat("MODEL_TEMPERATURE", 0)
TIMEOUT = envint("MODEL_TIMEOUT", 240)

# Optional sampling knobs, sent ONLY when set. Default is greedy (temperature 0), which is the
# right default for a benchmark. But the paper's open-model runs used the model card's sampling
# settings, so reproducing those numbers needs them passed through:
#   MODEL_TEMPERATURE=0.6 MODEL_TOP_P=0.95 MODEL_TOP_K=20 MODEL_PRESENCE_PENALTY=1.0
# top_k is not in the OpenAI schema; it is sent only if you set it, since some servers reject it.
def _optf(name):
    v = os.environ.get(name, "")
    return float(v) if v.strip() else None


TOP_P = _optf("MODEL_TOP_P")
TOP_K = _optf("MODEL_TOP_K")
PRESENCE_PENALTY = _optf("MODEL_PRESENCE_PENALTY")

# The agent protocol is in-band text: the model writes <think>…</think> then ONE
# <call_tool …>…</call_tool> or <answer>…</answer>, then stops. These stops enforce "then stops".
STOPS = ["</call_tool>", "</answer>"]

_CTX = ssl_context()

# Per-question token counters. Thread-local because the harness runs several questions at once.
_tls = threading.local()


# ---------------------------------------------------------------- message parts (harness contract)
def text_part(s):
    return {"_t": "text", "text": s}


def img_bytes(data, mime="image/jpeg"):
    return {"_t": "img", "data": data, "mime": mime}


# ---------------------------------------------------------------- token accounting
def q_reset():
    _tls.d = {"in": 0, "out": 0, "reasoning": 0}


def q_get():
    d = getattr(_tls, "d", None) or {"in": 0, "out": 0, "reasoning": 0}
    return d["in"], d["out"], d["reasoning"]


def _add(i, o, r):
    d = getattr(_tls, "d", None)
    if d is None:
        d = {"in": 0, "out": 0, "reasoning": 0}
        _tls.d = d
    d["in"] += int(i or 0)
    d["out"] += int(o or 0)
    d["reasoning"] += int(r or 0)       # part of `out`, reported only by servers that expose it


# ---------------------------------------------------------------- the wire
def _content(parts):
    out = []
    for p in parts or []:
        if p.get("_t") == "img":
            url = "data:%s;base64,%s" % (p.get("mime", "image/jpeg"),
                                         base64.b64encode(p["data"]).decode())
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            out.append({"type": "text", "text": p.get("text", "")})
    return out


def _messages(msgs):
    body = []
    for m in msgs:
        c = _content(m["parts"])
        # A text-only turn may be sent as a plain string; some endpoints are stricter about the
        # array form for non-multimodal models.
        if isinstance(c, list) and len(c) == 1 and c[0].get("type") == "text":
            c = c[0]["text"]
        body.append({"role": m["role"], "content": c})
    return body


class Client:
    """Override `complete()` for a non-OpenAI-compatible model; nothing else talks to the wire."""

    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or AGENT_MODEL
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        # "OpenAI-compatible" is a family, not a spec. Reasoning models reject `temperature` and
        # renamed `max_tokens`; some servers reject `stop`. Rather than make the user find out
        # from an HTTP 400 on question 1, learn each quirk once from the error and carry on.
        self.quirks = set()
        self._qlock = threading.Lock()
        if not self.model:
            raise RuntimeError("MODEL_NAME is not set (see .env.example)")
        if not self.base_url:
            raise RuntimeError("MODEL_BASE_URL is not set (see .env.example)")

    def _body(self, msgs, max_tokens):
        body = {"model": self.model, "messages": _messages(msgs)}
        if "no_max_tokens" in self.quirks:
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
        if "no_temperature" not in self.quirks:
            body["temperature"] = TEMPERATURE
        if "no_stop" not in self.quirks:
            body["stop"] = STOPS
        if TOP_P is not None and "no_top_p" not in self.quirks:
            body["top_p"] = TOP_P
        if TOP_K is not None and "no_top_k" not in self.quirks:
            body["top_k"] = int(TOP_K)
        if PRESENCE_PENALTY is not None and "no_presence_penalty" not in self.quirks:
            body["presence_penalty"] = PRESENCE_PENALTY
        return body

    def _learn(self, detail):
        """Return True if this 400 is one we can adapt to and should retry.

        A quirk we have ALREADY recorded still means retry, not give up: it means another thread
        learned it after we built our body, so the request we just sent was stale. Treating
        "already known" as unlearnable loses one question per racing thread.
        """
        d = (detail or "").lower()
        if "temperature" in d:
            q = "no_temperature"
        elif "max_completion_tokens" in d or "max_tokens" in d:
            q = "no_max_tokens"
        elif "stop" in d:
            q = "no_stop"
        elif "top_p" in d:
            q = "no_top_p"
        elif "top_k" in d:
            q = "no_top_k"
        elif "presence_penalty" in d:
            q = "no_presence_penalty"
        else:
            return False                     # not a parameter we know how to drop
        with self._qlock:
            fresh = q not in self.quirks
            self.quirks.add(q)
        if fresh:
            print("[model] endpoint rejected a parameter; adapting (%s)" % q, flush=True)
        return True

    def complete(self, msgs, max_tokens):
        for _ in range(4):                   # at most one retry per learnable quirk
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(self._body(msgs, max_tokens)).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.api_key})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT, context=_CTX) as resp:
                    d = json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
                try:
                    detail = e.read().decode()[:500]
                except Exception:
                    detail = ""
                if self._learn(detail):
                    continue
                raise urllib.error.HTTPError(e.url, e.code, detail, e.headers, None)
            u = d.get("usage") or {}
            _add(u.get("prompt_tokens"), u.get("completion_tokens"),
                 (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            return (d["choices"][0]["message"].get("content") or "")
        raise RuntimeError("endpoint kept rejecting the request body (quirks tried: %s)"
                           % sorted(self.quirks))


_client = None


def client():
    global _client
    if _client is None:
        _client = Client()
    return _client


def _transient(e):
    s = str(e).lower()
    return any(t in s for t in ("429", "rate limit", "rate_limit", "timeout", "timed out",
                                "502", "503", "504", "overloaded", "connection reset",
                                "temporarily unavailable"))


def mcall(msgs, retries=3, transient_retries=12):
    """One model turn -> assistant text. Retries transient errors with backoff; raises on a
    permanent one (bad key, unknown model, malformed request) rather than returning empty text,
    because empty text looks like a model that chose to say nothing."""
    last = None
    delay = 2.0
    for attempt in range(retries + transient_retries):
        try:
            # A retry after an error gets a larger output budget: 1x, 2x, then 3x MAX_TOKENS.
            return client().complete(msgs, MAX_TOKENS * (1 + min(attempt, 2)))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode()[:300]
            except Exception:
                detail = ""
            # An error re-raised after its body was already consumed has nothing left to read;
            # fall back to the message so the reason is never lost.
            detail = detail or str(getattr(e, "reason", "") or e)[:300]
            last = "HTTP %s %s" % (e.code, detail)
            if e.code in (400, 401, 403, 404) and "rate" not in detail.lower():
                raise RuntimeError(
                    "model endpoint rejected the request (%s).\n"
                    "  MODEL_NAME=%r MODEL_BASE_URL=%r\n"
                    "  Check the name is one this endpoint serves and the key is valid."
                    % (last, AGENT_MODEL, BASE_URL))
            if attempt >= retries + transient_retries - 1:
                break
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:200])
            if not _transient(e) and attempt >= retries - 1:
                break
        time.sleep(delay)
        delay = min(delay * 1.7, 30)
    raise RuntimeError("model call failed after %d attempts: %s"
                       % (retries + transient_retries, last))
