"""Step 2: split each weighted assistant message at `</think>` so reasoning and query are trained
with separate weights.

    python split_think.py weighted.jsonl split.jsonl
    python split_think.py weighted.jsonl split_uniform.jsonl --uniform     # control: every weight 1.0

One message becomes two: `<think>...</think>` with `loss_scale = think_scale`, then the rest (the
tool call) with `loss_scale = query_scale`. Messages without `</think>` are copied unchanged. A
`na_think` on the message (the released trajectories carry it) moves to the reasoning half.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--uniform", action="store_true", help="set every assistant weight to 1.0")
args = ap.parse_args()

TE = "</think>"


def strip(m):
    return {k: v for k, v in m.items() if k in ("role", "content", "loss", "loss_scale")}


n = nsplit = nmsg_in = nmsg_out = 0
with open(args.dst, "w") as out:
    for line in open(args.src):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n += 1
        msgs = d["messages"]
        nmsg_in += len(msgs)
        new = []
        for m in msgs:
            if (m.get("role") == "assistant" and ("think_scale" in m or "query_scale" in m)
                    and isinstance(m.get("content"), str)):
                k = m["content"].find(TE)
                if k != -1:
                    k += len(TE)
                    ts = 1.0 if args.uniform else float(m.get("think_scale", 1.0))
                    qs = 1.0 if args.uniform else float(m.get("query_scale", 1.0))
                    th = {"role": "assistant", "content": m["content"][:k], "loss_scale": ts}
                    if "na_think" in m:                 # step-3 credit shipped with the trajectory
                        th["na_think"] = m["na_think"]
                    new.append(th)
                    nsplit += 1
                    rest = m["content"][k:]
                    if rest:
                        new.append({"role": "assistant", "content": rest, "loss_scale": qs})
                    continue
            mm = strip(m)
            if args.uniform and mm.get("role") == "assistant" and "loss_scale" in mm:
                mm["loss_scale"] = 1.0
            new.append(mm)
        d["messages"] = new
        nmsg_out += len(new)
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
print("%s: %d records  %d split  messages %d -> %d  %s"
      % (args.dst, n, nsplit, nmsg_in, nmsg_out, "uniform" if args.uniform else "weighted"))
