"""Step 4: apply the next-action gate to the reasoning weights and write the training file.

    python build_train.py --split split.jsonl --na 'nextact.shard*.jsonl' --out train.jsonl
    python build_train.py --split split.jsonl --out train.jsonl      # na_think already in split.jsonl

For every reasoning step that has a next-action credit (`na_think`, from compute_nextact.py):
  keep it  if its credit is at least the median credit of all reasoning steps in this file AND the
           query it led to is itself kept (weight > 0);
  mask it  otherwise (loss_scale 0).
A step that was masked in step 1 but passes this gate is restored with the median weight of the
reasoning steps step 1 kept. Steps without a credit keep their step-1 weight.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import statistics as st
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, help="split_think.py output")
ap.add_argument("--na", nargs="+",
                help="compute_nextact.py shard file(s) or glob(s); without it, the na_think in --split is used")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def key(r):
    return hashlib.sha1(json.dumps([m.get("content") for m in r["messages"]],
                                   ensure_ascii=False).encode()).hexdigest()


def is_think(m):
    c = (m.get("content") or "").strip()
    return m["role"] == "assistant" and c.startswith("<think>") and c.endswith("</think>")


def kept(m):
    return (m.get("loss_scale") or 0) > 0 and m.get("loss") is not False


rows = [json.loads(l) for l in open(args.split)]
na = {}
if args.na:
    files = [f for g in args.na for f in sorted(glob.glob(g))]
    for f in files:
        for l in open(f):
            r = json.loads(l)
            na[key(r)] = {i: m["na_think"] for i, m in enumerate(r["messages"]) if "na_think" in m}
    print("read %d shard(s), %d trajectories carry a next-action credit" % (len(files), len(na)))
else:
    for r in rows:
        na[key(r)] = {i: m.pop("na_think") for i, m in enumerate(r["messages"]) if "na_think" in m}
    print("next-action credits read from %s" % args.split)
hit = [r for r in rows if key(r) in na]
cov = 100.0 * len(hit) / len(rows)
print("%d trajectories, %d matched (%.1f%%)" % (len(rows), len(hit), cov))
if len(hit) < len(rows):
    sys.exit("coverage %.1f%%, %d trajectories missing. The threshold is the median over this file's "
             "trajectories, so it cannot be set from a partial set: complete compute_nextact.py first."
             % (cov, len(rows) - len(hit)))

allna = [v for r in hit for v in na[key(r)].values()]
if not allna:
    sys.exit("no next-action credits at all; cannot set a threshold")
thr = st.median(allna)
w_rescue = st.median([m["loss_scale"] for r in rows for m in r["messages"]
                      if is_think(m) and kept(m)])

c = collections.Counter()
for r in rows:
    d = na[key(r)]
    msgs = r["messages"]
    for i, m in enumerate(msgs):
        if not is_think(m) or i not in d:
            continue
        nq = msgs[i + 1] if i + 1 < len(msgs) else {}
        keep = d[i] >= thr and kept(nq)
        was = kept(m)
        if keep and not was:
            m["loss_scale"] = w_rescue
            m.pop("loss", None)
            c["rescued"] += 1
        elif (not keep) and was:
            m["loss_scale"] = 0.0
            c["dropped"] += 1
        else:
            c["unchanged_" + ("kept" if keep else "masked")] += 1

# An image path that does not resolve does not raise at training time: the trainer feeds a blank
# page instead. Refuse to write a file that would do that.
missing = sorted({p for r in rows for p in (r.get("images") or []) if not os.path.exists(p)})
if missing:
    for p in missing[:5]:
        print("   missing image: %s" % p)
    sys.exit("%d images do not resolve; refusing to write." % len(missing))

with open(args.out, "w") as fo:
    for r in rows:
        for m in r["messages"]:
            m.pop("na_think", None)
        # ms-swift refuses a dataset with columns other than these two
        fo.write(json.dumps({k: r[k] for k in ("messages", "images") if k in r}, ensure_ascii=False) + "\n")

tk = [m for r in rows for m in r["messages"] if is_think(m)]
nz = [m["loss_scale"] for r in rows for m in r["messages"]
      if m["role"] == "assistant" and kept(m)]
print("na values=%d  median threshold=%.3f  rescue weight=%.4f" % (len(allna), thr, w_rescue))
print(dict(c))
print("think mask rate -> %.1f%%" % (sum(1 for m in tk if not kept(m)) / len(tk) * 100))
print("nonzero assistant weights: n=%d mean=%.4f" % (len(nz), st.mean(nz)))
print("wrote", args.out)
