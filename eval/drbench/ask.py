"""Ask one question and print the answer with the pages the agent retrieved.

    python -m drbench.ask "Which GPU does the paper use for its main experiments?" [--config Page]

Uses the same model (MODEL_*) and search endpoint (SEARCH_URL) as drbench.eval, so it works over
the benchmark corpus or over your own PDFs indexed with server/index_pdfs.py.
"""
import argparse
import json

from . import configs
from .harness import run_episode
from .retrieve import Retriever


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drbench.ask")
    ap.add_argument("question")
    ap.add_argument("--config", default="Page", type=configs.resolve, choices=list(configs.CONFIGS))
    ap.add_argument("--verbose", action="store_true", help="print each step as it runs")
    ap.add_argument("--json", action="store_true", help="print the whole episode record")
    a = ap.parse_args(argv)

    tools = {tool: Retriever(par) for tool, par in configs.CONFIGS[a.config].items()}
    r = run_episode(a.question, tools, verbose=a.verbose)
    if a.json:
        print(json.dumps(r, default=str, indent=1))
        return
    print(r["answer"] if r["answer"] is not None else "(no answer: %s)" % r["failed"])
    seen = []
    for s in r["searches"]:
        for h in s["hits"]:
            key = (h.get("doc"), h.get("page_idx"))
            if key not in seen:
                seen.append(key)
    print("\n%d searches; pages retrieved:" % len(r["searches"]))
    for doc, page in seen:
        print("  %s  page %s" % (doc, page))


if __name__ == "__main__":
    main()
