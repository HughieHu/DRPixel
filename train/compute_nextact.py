"""Step 3: next-action credit for every reasoning step.

    python compute_nextact.py --data split.jsonl --out nextact.jsonl --model /path/to/base-model \
        [--shard 0 --nshard 4]

For each `<think>` message that is followed by an action (a tool call or an answer), score the
action twice under the model: once after the real reasoning, once after an empty `<think>`.

    na_think = log P(action | reasoning) - log P(action | empty reasoning)

i.e. how much the reasoning step raises the likelihood of the action it led to. Written onto the
think message as `na_think`. Output goes to
`<out>.shard<k>.jsonl`; an interrupted shard resumes where it stopped.
"""
import argparse
import json
import os
import re

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--img_root", default="",
                help="directory to look in (by basename) for images whose recorded path does not exist")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
args = ap.parse_args()
MAX_PIXELS = 602112            # the training render cap (train.sh); scoring sees images exactly as training does

from transformers import AutoProcessor  # noqa: E402
try:
    from transformers import AutoModelForImageTextToText as _AM
except Exception:
    from transformers import AutoModelForVision2Seq as _AM

print("loading", args.model, flush=True)
proc = AutoProcessor.from_pretrained(args.model, max_pixels=MAX_PIXELS, trust_remote_code=True)
_ngpu = torch.cuda.device_count()
_dmap = "cuda" if _ngpu <= 1 else "balanced"
print("GPUs=%d device_map=%s" % (_ngpu, _dmap), flush=True)
model = _AM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=_dmap,
                            trust_remote_code=True)
model.eval()

EMPTY_THINK = "<think>\n\n</think>\n\n"


def rimg(p):
    if os.path.exists(p) or not args.img_root:
        return p
    return os.path.join(args.img_root, os.path.basename(p))


def to_parts(content, img_iter):
    parts = []
    for seg in re.split(r"(<image>)", content):
        if seg == "<image>":
            parts.append({"type": "image", "image": next(img_iter)})
        elif seg:
            parts.append({"type": "text", "text": seg})
    return parts or [{"type": "text", "text": content}]


def _base(prefix_pmsgs):
    g = proc.apply_chat_template(prefix_pmsgs, tokenize=False, add_generation_prompt=True)
    return g[:-len("<think>\n")] if g.endswith("<think>\n") else g


def score_seg(base, pre_text, target_text, images):
    """Summed log-likelihood of `target_text` after `base + pre_text`."""
    from PIL import Image
    ims = [Image.open(q).convert("RGB") for q in images] if images else None
    text_pref = base + pre_text
    text_full = text_pref + target_text
    enc = proc(text=[text_full], images=ims, return_tensors="pt").to(model.device)
    n_pref = proc(text=[text_pref], images=ims, return_tensors="pt").input_ids.shape[1]
    labels = enc.input_ids.clone()
    labels[:, :n_pref] = -100
    n_tgt = int((labels != -100).sum().item())
    if n_tgt == 0:
        return None
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return -float(loss.item()) * n_tgt


def process(rec):
    msgs = rec["messages"]
    imgs = [rimg(p) for p in rec.get("images", [])]
    img_before = [0]
    for mm in msgs:
        img_before.append(img_before[-1] + mm["content"].count("<image>"))

    def pmsgs_upto(k):
        it = iter(imgs[:img_before[k]])
        return [{"role": mm["role"], "content": to_parts(mm["content"], it)} for mm in msgs[:k]]

    na = {}
    for i, mm in enumerate(msgs):
        if mm["role"] != "assistant":
            continue
        c = mm["content"].strip()
        if not (c.startswith("<think>") and c.endswith("</think>")):
            continue
        if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "assistant":
            continue
        act = msgs[i + 1]["content"]
        if "<call_tool" not in act and "<answer>" not in act:
            continue
        base = _base(pmsgs_upto(i))
        pimgs = imgs[:img_before[i]]
        lp_with = score_seg(base, c + "\n\n", act, pimgs)
        lp_wo = score_seg(base, EMPTY_THINK, act, pimgs)
        if lp_with is not None and lp_wo is not None:
            na[i] = lp_with - lp_wo
    out_msgs = []
    for i, mm in enumerate(msgs):
        mm2 = dict(mm)
        if i in na:
            mm2["na_think"] = round(na[i], 4)
        out_msgs.append(mm2)
    rec2 = dict(rec)
    rec2["messages"] = out_msgs
    return rec2


recs = [json.loads(l) for l in open(args.data) if l.strip()]
recs = [r for i, r in enumerate(recs) if i % args.nshard == args.shard]
print("shard %d/%d: %d records" % (args.shard, args.nshard, len(recs)), flush=True)

outf = args.out.replace(".jsonl", ".shard%d.jsonl" % args.shard)
progress = outf + ".n"               # records attempted so far, including failures
skip = 0
if os.path.exists(progress):
    try:
        skip = int(open(progress).read().strip())
    except ValueError:
        skip = 0
elif os.path.exists(outf):
    skip = sum(1 for _ in open(outf))
if skip:
    print("RESUME: skipping %d records" % skip, flush=True)
    recs = recs[skip:]

n_ok = 0
with open(outf, "a") as fo:
    for j, r in enumerate(recs):
        r2 = None
        try:
            r2 = process(r)
            fo.write(json.dumps(r2) + "\n")
            fo.flush()
            n_ok += 1
        except Exception as e:
            print("  rec %d ERR: %s" % (j, str(e)[:140]), flush=True)
            with open(outf + ".failed", "a") as ff:
                ff.write("%d\n" % (skip + j))
            torch.cuda.empty_cache()
        with open(progress, "w") as fp:
            fp.write(str(skip + j + 1))
        if j % 20 == 19:
            torch.cuda.empty_cache()
        if j < 3 or j % 25 == 0:
            t = [mm.get("na_think") for mm in (r2["messages"] if r2 else []) if "na_think" in mm]
            print("  [%d/%d] na_think=%s" % (skip + j, skip + len(recs), t), flush=True)
print("DONE shard %d: %d ok" % (args.shard, n_ok), flush=True)
