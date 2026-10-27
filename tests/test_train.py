"""Steps 1, 2 and 4 of the training-data pipeline on synthetic trajectories. Step 3 needs a model
and is not run; its output is stood in for by synthetic next-action credit."""
import hashlib
import json
import random
import subprocess
import sys

from conftest import TRAIN


def trajectories(n=60, seed=0):
    rnd = random.Random(seed)
    for t in range(n):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q%d" % t}]
        for s in range(rnd.randint(1, 4)):
            msgs.append({"role": "assistant", "loss_scale": 1.0,
                         "content": '<think>t%d_%d</think>\n<call_tool name="search_page">x%d</call_tool>' % (t, s, s),
                         "cf_query": round(rnd.uniform(-1, 2), 3), "cf_think": round(rnd.uniform(-1, 2), 3)})
            msgs.append({"role": "user", "content": "<tool_output>obs</tool_output>"})
        msgs.append({"role": "assistant", "content": "<think>done</think>\n<answer>a</answer>", "loss_scale": 1.5})
        yield {"messages": msgs}


def run(*args, cwd):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline(tmp_path):
    (tmp_path / "credit.shard0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trajectories()))
    run(TRAIN + "/make_weights.py", "--shards", "credit.shard*.jsonl", "--out", "weighted.jsonl", cwd=tmp_path)
    run(TRAIN + "/split_think.py", "weighted.jsonl", "split.jsonl", cwd=tmp_path)
    run(TRAIN + "/split_think.py", "weighted.jsonl", "uniform.jsonl", "--uniform", cwd=tmp_path)

    rnd = random.Random(1)
    with open(tmp_path / "nextact.shard0.jsonl", "w") as f:
        for line in open(tmp_path / "split.jsonl"):
            r = json.loads(line)
            m = r["messages"]
            for i, x in enumerate(m):
                c = x["content"].strip()
                if x["role"] == "assistant" and c.startswith("<think>") and c.endswith("</think>") \
                        and i + 1 < len(m) and m[i + 1]["role"] == "assistant":
                    x["na_think"] = round(rnd.uniform(-2, 3), 4)
            f.write(json.dumps(r) + "\n")
    run(TRAIN + "/build_train.py", "--split", "split.jsonl", "--na", "nextact.shard*.jsonl", "--out", "train.jsonl",
        cwd=tmp_path)

    uniform = [json.loads(l) for l in open(tmp_path / "uniform.jsonl")]
    assert all(m.get("loss_scale", 1.0) == 1.0 for r in uniform for m in r["messages"] if m["role"] == "assistant")
    assert {sha(tmp_path / f) for f in ("weighted.jsonl", "split.jsonl", "train.jsonl")} == EXPECTED


EXPECTED = {
    "41ed328c1da456576d9559e33ab3454a9c19c9d35b63d104ec6e323346f51603",
    "dea43e4b141b474694729f8760072e15acd6258320fb018026bbdbba27bf7f28",
    "df4d42927e7920c2054052743b0fc65ecc7190811b34512b203cf91aea5d65e4",
}   # weighted, split, train; regenerate deliberately if a script changes


def test_rank_pool_and_shipped_nextact(tmp_path):
    """Rebuilding from a release-style credit file: named ranking pools, records without credit,
    and the step-3 gain carried in the file instead of separate shards."""
    import bisect
    recs = list(trajectories(n=50, seed=2))
    pools = {}
    for name, members, qthr in (("A", recs[:30], 0.0), ("B", recs, 1.0)):
        thr = {"cf_query": qthr, "cf_think": 0.0}
        pools[name] = {"threshold": thr, "max_weight": 2.0,
                       **{ch: sorted(m[ch] for r in members for m in r["messages"] if ch in m and m[ch] > thr[ch])
                          for ch in thr}}
    (tmp_path / "rank_pool.json").write_text(json.dumps(pools))
    rnd = random.Random(3)
    for i, r in enumerate(recs):
        for m in r["messages"]:
            if i % 10 == 9:                     # a trajectory without credit passes through unweighted
                m.pop("cf_query", None), m.pop("cf_think", None)
            elif "cf_think" in m:
                m["na_think"] = round(rnd.uniform(-2, 3), 4)
        if i % 10 != 9:
            r["rank_pool"] = "AB"[i % 2]
    (tmp_path / "credit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    run(TRAIN + "/make_weights.py", "--shards", "credit.jsonl", "--rank_pool", "rank_pool.json",
        "--out", "weighted.jsonl", cwd=tmp_path)

    def expect(cf, ch, p):
        if cf <= p["threshold"][ch]:
            return 0.0
        return round(2.0 * bisect.bisect_right(p[ch], cf) / len(p[ch]), 4)

    for src, out in zip(recs, (json.loads(l) for l in open(tmp_path / "weighted.jsonl"))):
        if "rank_pool" not in src:
            assert out == src
            continue
        p = pools[src["rank_pool"]]
        for a, b in zip(src["messages"], out["messages"]):
            if "cf_query" in a:
                assert b["query_scale"] == expect(a["cf_query"], "cf_query", p)
                assert b["think_scale"] == expect(a["cf_think"], "cf_think", p)

    run(TRAIN + "/split_think.py", "weighted.jsonl", "split.jsonl", cwd=tmp_path)
    run(TRAIN + "/build_train.py", "--split", "split.jsonl", "--out", "train_inline.jsonl", cwd=tmp_path)
    run(TRAIN + "/build_train.py", "--split", "split.jsonl", "--na", "split.jsonl", "--out", "train_shards.jsonl",
        cwd=tmp_path)
    assert sha(tmp_path / "train_inline.jsonl") == sha(tmp_path / "train_shards.jsonl")
    assert "na_think" not in (tmp_path / "train_inline.jsonl").read_text()
