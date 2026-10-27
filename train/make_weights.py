"""Step 1: turn per-step counterfactual credit into per-channel loss weights.

    python make_weights.py --shards 'credit.shard*.jsonl' --out weighted.jsonl
    python make_weights.py --shards credit/trajectories.jsonl --rank_pool credit/rank_pool.json --out weighted.jsonl

Input: trajectories whose search-step assistant messages carry `cf_query` (credit for the query)
and `cf_think` (credit for the reasoning). Each channel is mapped separately: credit at or below the
channel's threshold gets weight 0 (masked); a credit above it gets MAXW x k/n, where k is its rank
among the n above-threshold credits of that channel in the ranking pool. Writes `query_scale` /
`think_scale` on each such message.

By default the ranking pool is the input itself, with threshold 0 on both channels. `--rank_pool`
ranks against saved pools instead: a JSON object of named pools, each with its thresholds and the
sorted above-threshold credits of both channels; every record that carries credit names its pool
in `rank_pool`. `--save_rank_pool` writes the input's own pool in the same format, under the name
"default".
"""
import argparse
import bisect
import glob
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True, nargs="+", help="input JSONL file(s) or glob(s)")
ap.add_argument("--out", required=True)
ap.add_argument("--rank_pool", help="JSON of named ranking pools; each record names its pool in `rank_pool`")
ap.add_argument("--save_rank_pool", help="write the input's ranking pool to this JSON file")
args = ap.parse_args()

CH = ("cf_query", "cf_think")
THR = {"cf_query": 0.0, "cf_think": 0.0}   # credit <= THR[channel] is masked
MAXW = 2.0                                 # the top-ranked credit gets MAXW

shards = sorted(f for g in args.shards for f in glob.glob(g))
recs = []
for sh in shards:
    recs += [json.loads(l) for l in open(sh) if l.strip()]
print(f"loaded {len(recs)} records from {len(shards)} shards")

vals = {ch: [m[ch] for r in recs for m in r["messages"] if ch in m] for ch in CH}
if args.rank_pool:
    pools = json.load(open(args.rank_pool))
    print(f"ranking against {args.rank_pool}: pools {', '.join(sorted(pools))}")
else:
    pools = {"default": {"threshold": THR, "max_weight": MAXW,
                         **{ch: sorted(c for c in vals[ch] if c > THR[ch]) for ch in CH}}}
if args.save_rank_pool:
    json.dump(pools, open(args.save_rank_pool, "w"))
    print(f"wrote ranking pool to {args.save_rank_pool}")
for name, p in sorted(pools.items()):
    print(f"  pool {name}: " + "  ".join(f"{ch} masked at <= {p['threshold'][ch]}, {len(p[ch])} credits" for ch in CH))
for ch in CH:
    if vals[ch]:
        print(f"{ch}: n={len(vals[ch])}  range [{min(vals[ch]):.2f},{max(vals[ch]):.2f}]")


def weight(cf, ch, p):
    """0 for credit <= threshold, else max_weight x k/n: its rank k among the n pool credits of its channel."""
    if cf <= p["threshold"][ch]:
        return 0.0
    return round(p["max_weight"] * bisect.bisect_right(p[ch], cf) / len(p[ch]), 4)


nq = nq_mask = nt_mask = 0
sq = st = 0.0
for r in recs:
    if not any(ch in m for m in r["messages"] for ch in CH):
        continue
    name = r.get("rank_pool", "default") if args.rank_pool else "default"
    if name not in pools:
        sys.exit(f"record {r.get('_qid', '?')} names ranking pool {name!r}, which {args.rank_pool} does not have")
    p = pools[name]
    for m in r["messages"]:
        if m.get("role") != "assistant":
            continue
        if "cf_query" in m or "cf_think" in m:
            cfq = m.pop("cf_query", None)
            cft = m.pop("cf_think", None)
            qs = weight(cfq, "cf_query", p) if cfq is not None else 1.0
            ts = weight(cft, "cf_think", p) if cft is not None else 1.0
            m["query_scale"] = qs
            m["think_scale"] = ts
            nq += 1
            sq += qs
            st += ts
            nq_mask += (qs == 0.0)
            nt_mask += (ts == 0.0)

with open(args.out, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nwrote {args.out}  ({len(recs)} recs)")
print(f"search-step msgs weighted: {nq}")
print(f"  query_scale: mean {sq/max(1,nq):.3f}  masked {nq_mask} ({100*nq_mask/max(1,nq):.1f}%)")
print(f"  think_scale: mean {st/max(1,nq):.3f}  masked {nt_mask} ({100*nt_mask/max(1,nq):.1f}%)")
