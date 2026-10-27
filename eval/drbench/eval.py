"""One command scores a model on the benchmark.

    python -m drbench.eval --config Page

Runs every question, grades the answers, prints the table. Safe to interrupt: rerun the same
command and it picks up where it stopped (finished questions are read back from the output file).

    --config    Text | Page | Text_Fig     (case-insensitive)
    --limit N   first N questions only — use this for a smoke test before spending a full run
    --no-grade  run but do not call the judge (free; retrieval metrics still work)
    --grade-only  grade an existing results file without re-running the agent
    --out PATH  results file (default results/<model>__<config>.json)
    --concurrency N  questions run in parallel (default 4, or CONCURRENCY)
    --verbose   print each episode as it runs
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import configs, envint, judge
from . import model as gem
from .harness import run_episode
from .retrieve import Retriever

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_prev(path):
    """Read a previous run back.

    Returns `(records_by_qid, done_qids, model_label)`. The first two are deliberately different:
    `done_qids` is only the questions worth NOT re-running, while `records_by_qid` is everything
    on disk. Conflating them loses the errored records from the report — which would also zero
    out `n_failed` and make the failures invisible.
    """
    recs, done, label = {}, set(), None
    if not os.path.exists(path):
        return recs, done, label
    try:
        prev = json.load(open(path))
        label = prev.get("model")
        for r in prev.get("results", []):
            recs[r["qid"]] = r
            if r.get("failed") or not (r.get("answer") or "").strip() \
               or (r.get("answer") or "").startswith("ERROR"):
                continue
            done.add(r["qid"])
    except Exception as e:
        print("warning: could not read %s (%s); starting fresh" % (path, e), file=sys.stderr)
    return recs, done, label


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drbench.eval")
    ap.add_argument("--config", required=True, type=configs.resolve, choices=list(configs.CONFIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int,
                    default=envint("CONCURRENCY", 4))
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = configs.CONFIGS[args.config]
    bench = json.load(open(os.path.join(HERE, "bench", "eval500.json")))
    if args.limit:
        bench = bench[:args.limit]

    out = args.out or os.path.join(
        HERE, "results", "%s__%s.json" % ((gem.AGENT_MODEL or "model").replace("/", "_"), args.config))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    prev, done, prev_label = _load_prev(out)
    todo = [] if args.grade_only else [it for it in bench if it["qid"] not in done]
    # Keep every prior record except the ones we are about to replace by re-running them.
    retry = {it["qid"] for it in todo}
    results = [r for qid, r in prev.items() if qid not in retry]

    if not args.grade_only:
        print("model   %s  ->  %s" % (gem.AGENT_MODEL or "(MODEL_NAME unset!)", gem.BASE_URL or "(MODEL_BASE_URL unset!)"))
        print("config  %s  tools=%s" % (args.config, dict(cfg)))
        print("bench   %d questions (%d already done, %d to run)" % (len(bench), len(done), len(todo)))
        print("out     %s" % out)

        # Build the retrievers up front and check the endpoint ONCE, so a misconfigured
        # SEARCH_URL fails here instead of 500 questions deep.
        tools = {tn: Retriever(par) for tn, par in cfg.items()}
        if todo:
            try:
                h = list(tools.values())[0].health()
                missing = [p for p in cfg.values() if p not in (h.get("paradigms") or [])]
                if missing:
                    sys.exit("search endpoint does not serve %s (it serves %s)"
                             % (missing, h.get("paradigms")))
                print("search  %s  serving %s" % (list(tools.values())[0].url, h.get("paradigms")))
            except SystemExit:
                raise
            except Exception as e:
                sys.exit("cannot reach the search endpoint: %s\n  SEARCH_URL=%r"
                         % (e, os.environ.get("SEARCH_URL", "")))

        t0 = time.time()
        lock = threading.Lock()
        n_done = [0]

        def process(item):
            res = run_episode(item["question"], tools, verbose=args.verbose)
            return {"qid": item["qid"], "question": item.get("question"),
                    "gold_answer": item.get("gold_answer"),
                    "modality": item.get("modality"), "doc": item.get("doc"),
                    "arxiv": item.get("arxiv"), "page": item.get("page"),
                    "answer": res["answer"], "failed": res.get("failed"),
                    "correct": None, "reason": None,
                    "n_calls": res["n_calls"], "calls_by_tool": res.get("calls_by_tool"),
                    "in_tok": res.get("in_tok"), "out_tok": res.get("out_tok"),
                    "reasoning_tok": res.get("reasoning_tok"), "wall_s": res.get("wall_s"),
                    "hits": judge.retrieval_hits(item, res["searches"], cfg),
                    "searches": res["searches"], "steps": res.get("steps"),
                    "meta": res.get("meta")}

        def save():
            tmp = out + ".tmp"
            json.dump({"config": args.config, "model": gem.AGENT_MODEL,
                       "results": results}, open(tmp, "w"), indent=1)
            os.replace(tmp, out)      # atomic: a crash mid-write must not destroy the run

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(process, it) for it in todo]
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    print("  question failed hard: %s" % str(e)[:200], flush=True)
                    continue
                with lock:
                    results.append(rec)
                    n_done[0] += 1
                    if n_done[0] % 5 == 0 or n_done[0] == len(todo):
                        save()
                        print("  %d/%d  %.0fs" % (n_done[0], len(todo), time.time() - t0),
                              flush=True)
        save()

    # ------------------------------------------------------------------ grade
    if args.grade_only and not results:
        sys.exit("nothing to grade: %s has no finished questions" % out)

    if not args.no_grade:
        ungraded = [r for r in results if r.get("correct") is None and not r.get("failed")]
        if ungraded:
            if not judge.JUDGE_API_KEY:
                print("\nskipping grading: no JUDGE_API_KEY / OPENAI_API_KEY set.\n"
                      "Retrieval metrics below are complete; Acc needs the judge.", file=sys.stderr)
            else:
                print("\ngrading %d answers (%s x%d)..."
                      % (len(ungraded), judge.JUDGE_MODEL, judge.JUDGE_VOTES), flush=True)
                lock = threading.Lock()
                n = [0]

                def grade(r):
                    v = judge.judge_answer(r["question"], r["gold_answer"], r["answer"])
                    return r, v

                with ThreadPoolExecutor(max_workers=max(4, args.concurrency)) as ex:
                    for fut in as_completed([ex.submit(grade, r) for r in ungraded]):
                        r, v = fut.result()
                        with lock:
                            r["correct"], r["reason"] = v["correct"], v["reason"]
                            n[0] += 1
                            if n[0] % 25 == 0:
                                print("  graded %d/%d" % (n[0], len(ungraded)), flush=True)

    # ------------------------------------------------------------------ report
    m = judge.metrics(results)
    by_mod = judge.by_modality(results)
    # Grading an existing file must not relabel whose run it was.
    label = (prev_label if args.grade_only else None) or gem.AGENT_MODEL or prev_label or "model"
    json.dump({"config": args.config, "model": label, "metrics": m,
               "by_modality": by_mod, "results": results}, open(out, "w"), indent=1)

    print("\n=== %s / %s ===" % (label, args.config))
    print(judge.format_table(m, by_mod, graded=not args.no_grade))
    print("\nfull records: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
