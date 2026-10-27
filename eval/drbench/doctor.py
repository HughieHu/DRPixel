"""Check your setup before spending a full run.

    python -m drbench.doctor [Text|Page|Text_Fig]     # default Page

Makes a handful of calls and tells you, specifically, what is wrong. Worth the 20 seconds: the
failure this catches most often is a model that silently ignores images, which does not crash —
it just scores badly on the page config and looks like a weak model.
"""
import sys

from . import configs, judge
from . import model as gem
from .retrieve import Retriever, SearchError

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def _p(tag, name, detail=""):
    print("[%s] %-28s %s" % (tag, name, detail))


def main(argv=None):
    want = (argv or sys.argv[1:])
    cfg = configs.resolve(want[0]) if want else "Page"
    if cfg not in configs.CONFIGS:
        sys.exit("unknown config %r; pick one of %s" % (cfg, list(configs.CONFIGS)))
    fails = []

    print("\n=== config: %s ===" % cfg)

    # ---------------------------------------------------------------- model endpoint
    if not gem.AGENT_MODEL:
        _p(BAD, "MODEL_NAME", "not set")
        fails.append("MODEL_NAME")
    else:
        _p(OK, "MODEL_NAME", gem.AGENT_MODEL)
    if not gem.BASE_URL:
        _p(BAD, "MODEL_BASE_URL", "not set")
        fails.append("MODEL_BASE_URL")
    else:
        _p(OK, "MODEL_BASE_URL", gem.BASE_URL)

    if not fails:
        try:
            txt = gem.mcall([{"role": "user", "parts": [gem.text_part(
                "Reply with exactly: OK")]}], retries=1, transient_retries=1)
            _p(OK, "model responds", repr(txt[:40]))
        except Exception as e:
            _p(BAD, "model responds", str(e)[:150])
            fails.append("model call")

        # Does it honour the protocol's stop strings / can we repair them?
        try:
            txt = gem.mcall([{"role": "user", "parts": [gem.text_part(
                'Output exactly this and nothing else: <answer>42</answer>')]}],
                retries=1, transient_retries=1)
            if "<answer>" in txt and "</answer>" in txt:
                _p(OK, "answer tag round-trips", "stop-string handling fine")
            elif "<answer>" in txt:
                _p(WARN, "answer tag", "closing tag missing but repaired client-side")
            else:
                _p(WARN, "answer tag", "model did not emit the tag: %r" % txt[:60])
        except Exception as e:
            _p(WARN, "answer tag", str(e)[:120])

    # ---------------------------------------------------------------- search endpoint
    pars = list(configs.CONFIGS[cfg].values())
    try:
        r = Retriever(pars[0])
        h = r.health()
        served = h.get("paradigms") or []
        _p(OK, "search endpoint", "%s serving %s" % (r.url, served))
        missing = [p for p in pars if p not in served]
        if missing:
            _p(BAD, "paradigms for this config", "endpoint lacks %s" % missing)
            fails.append("paradigms")
    except SearchError as e:
        _p(BAD, "search endpoint", str(e)[:200])
        fails.append("SEARCH_URL")
    except Exception as e:
        _p(BAD, "search endpoint", "unreachable: %s" % str(e)[:150])
        fails.append("SEARCH_URL")

    # ---------------------------------------------------------------- a real retrieval
    img, mime = None, "image/png"
    if "paradigms" not in fails and "SEARCH_URL" not in fails:
        for par in pars:
            try:
                hits = Retriever(par).search("retrieval augmented generation benchmark", k=2)
                if not hits:
                    _p(WARN, "search %s" % par, "returned 0 hits")
                    continue
                keys = sorted(k for k in hits[0] if k != "image")
                has_img = isinstance(hits[0].get("image"), (bytes, bytearray))
                _p(OK, "search %s" % par,
                   "%d hits, fields=%s%s" % (len(hits), keys,
                                             ", image=%d bytes" % len(hits[0]["image"]) if has_img else ""))
                if has_img and img is None:
                    img = hits[0]["image"]
                    mime = "image/jpeg" if par == "figure" else "image/png"   # as the harness sends them
            except Exception as e:
                _p(BAD, "search %s" % par, str(e)[:150])
                fails.append("search %s" % par)

    # ---------------------------------------------------------------- can the model SEE?
    if cfg in configs.NEEDS_VISION and img and "model call" not in fails:
        try:
            txt = gem.mcall([{"role": "user", "parts": [
                gem.img_bytes(img, mime),
                gem.text_part("Can you see the image above? Answer YES or NO only."),
            ]}], retries=1, transient_retries=1)
            if "yes" in txt.lower():
                _p(OK, "model accepts images", repr(txt[:40]))
            else:
                _p(BAD, "model accepts images", "replied %r" % txt[:60])
                fails.append("vision")
        except Exception as e:
            _p(BAD, "model accepts images", str(e)[:150])
            fails.append("vision")
    elif cfg in configs.NEEDS_VISION:
        _p(WARN, "model accepts images", "not tested (no image to test with)")
    else:
        _p(OK, "vision not needed", "%s sends no images" % cfg)

    # ---------------------------------------------------------------- judge
    if judge.JUDGE_API_KEY:
        try:
            v = judge.judge_answer("What is 2+2?", "4", "The answer is 4.", votes=1)
            _p(OK, "judge", "%s -> correct=%s" % (judge.JUDGE_MODEL, v["correct"]))
        except Exception as e:
            _p(BAD, "judge", str(e)[:150])
            fails.append("judge")
    else:
        _p(WARN, "judge", "no JUDGE_API_KEY/OPENAI_API_KEY -- you can run but not score")

    print()
    if fails:
        print("NOT READY: %s" % ", ".join(fails))
        return 1
    print("READY. Smoke-test 10 questions, then run the full 500:")
    print("  python -m drbench.eval --config %s --limit 10" % cfg)
    print("  python -m drbench.eval --config %s" % cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
