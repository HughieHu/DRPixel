"""Step 0: counterfactual credit for every search step.

    python compute_credit.py --data trajectories.jsonl --out credit.jsonl --model /path/to/base-model \
        [--shard 0 --nshard 4]

How strongly a history H supports the answer is measured under the frozen base model as
l(H) = log P(answer | H), the summed log-likelihood of the trajectory's final answer (the text inside
its last `<answer>`; only trajectories with a correct final answer are used). For search step i,
with H_i the history up to and including the results of the i-th search and H_T the full history
before the final answer:

    cf_query = l(H_i) - l(H_{i-1})                         what the step's search results add
    cf_think = l(H_T) - l(H_T with step i's <think> removed)  what the step's reasoning contributes

Both are written onto the search step's assistant message; the answer message gets the constant
answer weight as its `loss_scale`. Output goes to `<out>.shard<k>.jsonl`; an interrupted shard
resumes where it stopped.

Each trajectory takes 2 + (number of searches) + (number of reasoning steps) forward passes, the
reasoning ones at full length. The model needs about 125-142 GB in bf16 at full trajectory length:
one H200, or two 80 GB GPUs (the model is split across all visible GPUs). On one H200 a trajectory
takes about a minute.
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
ap.add_argument("--limit", type=int, default=0, help="score only the first N records of the shard")
args = ap.parse_args()
MAX_PIXELS = 602112            # the training render cap (train.sh); scoring sees images exactly as training does
ANSWER_WEIGHT = 1.5            # lambda: constant weight of the final answer

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

ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


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


def score_logp(prefix_pmsgs, answer_text, images):
    """l(H): summed log-likelihood of `answer_text` as the assistant turn after `prefix_pmsgs`."""
    from PIL import Image
    full = prefix_pmsgs + [{"role": "assistant", "content": [{"type": "text", "text": answer_text}]}]
    text_full = proc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    text_pref = proc.apply_chat_template(prefix_pmsgs, tokenize=False, add_generation_prompt=True)
    ims = [Image.open(q).convert("RGB") for q in images] if images else None
    enc = proc(text=[text_full], images=ims, return_tensors="pt").to(model.device)
    n_pref = proc(text=[text_pref], images=ims, return_tensors="pt").input_ids.shape[1]
    labels = enc.input_ids.clone()
    labels[:, :n_pref] = -100
    n_ans = int((labels != -100).sum().item())
    if n_ans == 0:
        return None
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return -float(loss.item()) * n_ans


def process(rec):
    msgs = rec["messages"]
    imgs = [rimg(p) for p in rec.get("images", [])]
    m = ANS_RE.search(msgs[-1]["content"])
    answer = m.group(1).strip() if m else ""
    if not answer:
        return None
    tool_idx = [i for i, mm in enumerate(msgs) if mm["role"] == "tool"]
    img_before = [0]
    for mm in msgs:
        img_before.append(img_before[-1] + mm["content"].count("<image>"))

    def pmsgs_upto(msglist, k):
        """Messages 0..k as model input, with their images."""
        it = iter(imgs[:img_before[k + 1]])
        return [{"role": mm["role"], "content": to_parts(mm["content"], it)} for mm in msglist[:k + 1]]

    # cf_query: the search step is the assistant message before each tool message; its credit is
    # the change in l when its results are added to the history (system + question at the start).
    cf_query = {}
    lp_prev = score_logp(pmsgs_upto(msgs, 1), answer, imgs[:img_before[2]])
    for ti in tool_idx:
        lp_cur = score_logp(pmsgs_upto(msgs, ti), answer, imgs[:img_before[ti + 1]])
        ok = lp_cur is not None and lp_prev is not None
        cf_query[ti - 1] = (lp_cur - lp_prev) if ok else 0.0
        lp_prev = lp_cur

    # cf_think: remove one search step's <think> from the full history before the final answer.
    full_k = len(msgs) - 2
    lp_full = score_logp(pmsgs_upto(msgs, full_k), answer, imgs[:img_before[full_k + 1]])
    cf_think = {}
    for i in cf_query:
        if "<think>" not in msgs[i]["content"]:
            continue
        abl = [dict(mm) for mm in msgs]
        abl[i]["content"] = THINK_RE.sub("", abl[i]["content"], count=1)
        lp_abl = score_logp(pmsgs_upto(abl, full_k), answer, imgs[:img_before[full_k + 1]])
        ok = lp_full is not None and lp_abl is not None
        cf_think[i] = (lp_full - lp_abl) if ok else 0.0

    out_msgs = []
    for i, mm in enumerate(msgs):
        mm2 = dict(mm)
        if mm["role"] == "assistant":
            if "<answer>" in mm["content"]:
                mm2["loss_scale"] = ANSWER_WEIGHT
            elif i in cf_query:
                mm2["cf_query"] = round(cf_query[i], 4)
                if i in cf_think:
                    mm2["cf_think"] = round(cf_think[i], 4)
        out_msgs.append(mm2)
    rec2 = dict(rec)
    rec2["messages"] = out_msgs
    return rec2


recs = [json.loads(l) for l in open(args.data) if l.strip()]
recs = [r for i, r in enumerate(recs) if i % args.nshard == args.shard]
if args.limit:
    recs = recs[:args.limit]
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
            if r2:
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
        if j < 3 or j % 25 == 0:
            q = [mm.get("cf_query") for mm in (r2["messages"] if r2 else []) if "cf_query" in mm]
            t = [mm.get("cf_think") for mm in (r2["messages"] if r2 else []) if "cf_think" in mm]
            print("  [%d/%d] cf_query=%s cf_think=%s" % (skip + j, skip + len(recs), q, t), flush=True)
print("DONE shard %d: %d ok" % (args.shard, n_ok), flush=True)
