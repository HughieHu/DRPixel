"""The agent loop.

Protocol: each turn the model writes `<think>…</think>` then AT MOST ONE
`<call_tool name="...">query</call_tool>` or `<answer>…</answer>`, then stops. We run the tool and
send back a `<tool_output>` block; retrieved figure/page images go in as real image parts. All
observations stay in the context for the rest of the episode.

The system prompt below is byte-identical to the one used for the paper. Editing it makes your
numbers incomparable to the reported ones — if you want to study prompt variants, say so when you
report the result.

Every model uses the in-band `<call_tool>` protocol here, which keeps models comparable. The paper's
GPT-5.4 numbers were produced with provider-native function calling instead, so a GPT-5.4 run of
this harness can differ from them.
"""
import collections
import re
import time

from . import envint
from . import model as gem

TOOL_DESCS = {
    "search_text":   'search_text: retrieve the top-5 most relevant TEXT passages from the paper corpus.',
    "search_figure": 'search_figure: retrieve the top-5 most relevant FIGURE images from the corpus (you will SEE them).',
    "search_page":   'search_page: retrieve the top-3 most relevant whole-PAGE images from the corpus (you will SEE them).',
}
K_BY_TOOL = {"search_text": 5, "search_figure": 5, "search_page": 3}

OBS_DESCS = {
    "search_text":   '  - search_text -> <passage id=T# doc=DOC_ID>the passage text</passage>',
    "search_figure": '  - search_figure -> <figure id=F# doc=DOC_ID> followed by the FIGURE IMAGE (you can see it)',
    "search_page":   '  - search_page -> <page id=P# doc=DOC_ID page=N> followed by the PAGE IMAGE (you can see it)',
}

SYSTEM = """You are a research assistant who answers questions about scientific papers through iterative reasoning and research.

## Process
- Show your reasoning inside <think></think> tags.
- Call a tool with exactly:  <call_tool name="TOOL_NAME">your query</call_tool>
- When you have enough evidence, give the final answer with exactly:  <answer>your answer</answer>

- You may alternate between thinking and tool calls several times.

## STOP after the tag  (this is the most important rule)
Each turn you emit AT MOST ONE <call_tool> or ONE <answer>, and then you STOP WRITING.
Do NOT write a <tool_output> block yourself -- WE run the tool and send you the real results.
Anything you write after the closing </call_tool> is discarded, and any tool output you invent
is a hallucination: the documents, page numbers and numbers in it are not real.

Your turn looks like exactly this and nothing more:

    <think>What I know so far, and what I still need.</think>
    <call_tool name="TOOL_NAME">my query</call_tool>

Your <think> stays visible to you for the rest of the session, so record in it any concrete
value you read off an image -- that is your working memory.

## Tools (use ONLY these)
{tools}

## Tool Output
After each tool call the results come back to you in a <tool_output> block:
{obs}
Results from EVERY tool call stay visible to you for the rest of the session -- you can re-read an
image you retrieved earlier.

## Before you answer -- check yourself
You may only answer when you can point to the SPECIFIC page/passage id you read the answer off
(e.g. "P2_1"). Ask yourself:
  - Is the exact value/claim the question asks for actually VISIBLE in one of the results I received?
  - If NO -> do NOT answer. Search again with a different query.
  - "The retrieved page is about the right topic" is NOT enough. "I remember this paper" is NOT enough.
Answering from memory when the evidence is not in front of you is the single worst failure mode here,
because the answer will look confident and be about the wrong paper.

## CRITICAL ANSWERING RULES
- Answer ONLY from evidence you actually retrieved this session. Do NOT answer from prior knowledge.
- When the answer involves a specific number, score, magnitude, threshold, or a named
  model/method/dataset, you MUST retrieve it and state the SPECIFIC value(s). A vague or directional
  answer ("generally better", "tends to improve", "competitive", "around X") when a precise value
  exists is treated as WRONG.
- NEVER guess or fabricate numbers, model names, method names, arxiv ids or document ids. If after
  searching you still cannot find the specific value the question asks for, say so explicitly --
  do NOT invent one.
- You start this session having retrieved NOTHING. The ONLY evidence that exists is what comes back
  inside a real <tool_output> block that WE send you. If you have not yet received a <tool_output>,
  you have no evidence and you MUST NOT answer -- search first.
- Recalling the paper from memory is not evidence. If your answer would be the same without looking
  at the retrieved pages, you have not done the task.
- For yes/no, comparison, trend, or regime questions: state the verdict/direction AND the supporting
  numbers behind it.

Illustrative GOOD vs BAD answers (fabricated examples, NOT the real question):
- Q "Does pruning hurt accuracy on the small model?"
  BAD : "Yes, pruning generally lowers accuracy somewhat."            (vague -> WRONG)
  GOOD: "Yes -- accuracy drops from 79.3% to 74.1% at 50% sparsity (Table 3), a 5.2-point fall."
- Q "Which retriever wins on the multilingual benchmark, and by how much?"
  BAD : "A single general retriever is competitive across languages."  (names nothing -> WRONG)
  GOOD: "Retriever-Z leads, +1.8 nDCG@10 over the per-language baseline (Table 4)."

Rules: you may call tools at most {budget} times in total; after that you MUST answer.
Issue focused queries. Read the returned passages/images carefully."""

IMG_HINT = ("\n\nIMPORTANT -- you also have an IMAGE tool ({imgtools}). Some questions can ONLY be answered by "
            "looking at a figure, chart, plot, or table image: the exact number / curve shape / comparison is NOT "
            "in the body text. Whenever a question refers to a figure/table/chart, or asks for a value you cannot "
            "find in retrieved text, you MUST use the image tool and READ the actual image.")

# The paper drops the "You are a research assistant who" persona for GPT models.
if gem.AGENT_MODEL.startswith("gpt-"):
    SYSTEM = SYSTEM.replace(
        "You are a research assistant who answers questions about scientific papers through iterative reasoning and research.",
        "You answer questions about scientific papers through iterative reasoning and research.", 1)

_CALL_RE = re.compile(r'<call_tool\s+name\s*=\s*"([^"]+)"\s*>(.*?)</call_tool>', re.S | re.I)
_ANS_RE = re.compile(r'<answer>(.*?)</answer>', re.S | re.I)
_QWRAP_RE = re.compile(r'^\s*query\s*=\s*', re.I)


def _close_tags(txt):
    """Repair an unterminated tag. Needed because endpoints that honour `stop` strings do not
    return the string that stopped them (the OpenAI API strips it; vLLM/sglang may keep it)."""
    t = txt or ""
    if t.count("<answer>") > t.count("</answer>"):
        t += "</answer>"
    if t.count("<call_tool") > t.count("</call_tool>"):
        t += "</call_tool>"
    if t.count("</think>") > t.count("<think>"):
        t = "<think>" + t
    if t.count("<think>") > t.count("</think>"):
        t += "</think>"
    return t


def _parse(txt):
    txt = _close_tags(txt)
    mc = _CALL_RE.search(txt)
    ma = _ANS_RE.search(txt)
    if mc and (not ma or mc.start() < ma.start()):
        return "call", mc.group(1).strip(), mc.group(2).strip(), txt[:mc.end()]
    if ma:
        return "answer", ma.group(1).strip(), None, txt[:ma.end()]
    if "<answer>" in txt:
        _a = txt.split("<answer>", 1)[1].strip()
        return "answer", _a, None, txt
    return "none", txt.strip(), None, txt


def _clean_query(q):
    """Unwrap the two things models wrap queries in: a JSON object, and `query="..."`."""
    if not q:
        return q
    t = q.strip()
    if t.startswith("{"):
        try:
            import json
            d = json.loads(t)
            if isinstance(d, dict) and d.get("query"):
                return str(d["query"]).strip()
        except Exception:
            pass
    m = _QWRAP_RE.match(t)
    if m:
        t = t[m.end():].strip()
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
            t = t[1:-1]
        elif t.startswith(("\"", "'")):
            t = t[1:]
        return t.strip()
    return q


def _hit_record(tool, h, hid):
    r = {"id": hid, "doc": h["doc"], "corpus": h.get("corpus"), "score": h.get("score")}
    for k in ("page_idx", "hash", "chunk_idx"):
        if k in h:
            r[k] = h[k]
    if tool == "search_text":
        r["text"] = h.get("text") or ""         # the passage exactly as the model saw it
    r["has_image"] = h.get("image") is not None
    return r


def _format_obs(tool, hits, step):
    parts = [gem.text_part("<tool_output>")]
    recs = []
    for i, h in enumerate(hits):
        if tool == "search_text":
            hid = "T%d_%d" % (step, i)
            parts.append(gem.text_part('<passage id=%s doc=%s>%s</passage>'
                                       % (hid, h["doc"], h.get("text") or "")))
        elif tool == "search_figure":
            hid = "F%d_%d" % (step, i)
            parts.append(gem.text_part('<figure id=%s doc=%s>' % (hid, h["doc"])))
            if h.get("image"):
                parts.append(gem.img_bytes(h["image"], "image/jpeg"))
        else:
            hid = "P%d_%d" % (step, i)
            parts.append(gem.text_part('<page id=%s doc=%s page=%s>'
                                       % (hid, h["doc"], h.get("page_idx"))))
            if h.get("image"):
                parts.append(gem.img_bytes(h["image"], "image/png"))
        recs.append(_hit_record(tool, h, hid))
    parts.append(gem.text_part("</tool_output>"))
    return parts, recs


BUDGET = 20            # tool calls per question; it is stated in the system prompt
MAX_IMAGES = 24        # images kept in context; every reported run used 24
# Default to a 128k context: right for the open models the paper trains, and safe for everything
# else. Raise it to your endpoint's real limit — too LOW only costs you dropped images, too HIGH
# gets the request rejected mid-episode.
CTX_LIMIT = envint("CTX_LIMIT", 131072)
CTX_SAFE = 0.75        # drop images once the estimated prompt passes 75% of CTX_LIMIT
IMG_TOKENS = 2200      # flat per-image cost in that estimate


def _img_count(msgs):
    return sum(1 for m in msgs for p in m["parts"] if p["_t"] == "img")


def _est_prompt_tokens(msgs):
    """Deliberately crude (chars/4 + a flat cost per image): it only decides WHEN to drop images.
    Reported token counts come from the endpoint's own `usage`, never from here."""
    chars = imgs = 0
    for m in msgs:
        for p in m["parts"]:
            if p["_t"] == "img":
                imgs += 1
            else:
                chars += len(str(p.get("text") or ""))
    return chars // 4 + imgs * IMG_TOKENS


def _drop_oldest_images(msgs, n_drop):
    for m in msgs:
        if n_drop <= 0:
            break
        out = []
        for p in m["parts"]:
            if p["_t"] == "img" and n_drop > 0:
                n_drop -= 1
                out.append(gem.text_part("[image dropped -- context image cap reached]"))
            else:
                out.append(p)
        m["parts"] = out


def _fit_context(msgs):
    budget = int(CTX_LIMIT * CTX_SAFE)
    dropped = 0
    while _est_prompt_tokens(msgs) > budget and _img_count(msgs) > 2:
        _drop_oldest_images(msgs, 3)
        dropped += 3
    return dropped


def run_episode(question, tools, verbose=False):
    """Run one question. `tools` maps tool name -> an object with `.search(query, k)`.

    Returns the per-question record: answer, every search with its hits, per-step reasoning and
    token counts.
    """
    t0 = time.time()
    budget, max_steps = BUDGET, BUDGET + 6
    tdesc = "\n".join(TOOL_DESCS[t] for t in tools)
    odesc = "\n".join(OBS_DESCS[t] for t in tools)
    imgtools = [t for t in tools if t in ("search_figure", "search_page")]

    sys_txt = SYSTEM.format(tools=tdesc, obs=odesc, budget=budget)
    if imgtools:
        sys_txt += IMG_HINT.format(imgtools=", ".join(imgtools))

    msgs = [{"role": "system", "parts": [gem.text_part(sys_txt)]},
            {"role": "user", "parts": [gem.text_part("QUESTION: " + question)]}]

    searches, steps = [], []
    calls_by_tool = collections.Counter()
    n_calls, answer, bad, no_tag = 0, "", 0, 0
    failed = None
    gem.q_reset()
    prev = (0, 0, 0)

    for step in range(max_steps):
        force = n_calls >= budget
        if force:
            msgs.append({"role": "user", "parts": [gem.text_part(
                "You have used all %d tool calls. Give your final answer NOW in <answer></answer> tags." % budget)]})
        try:
            txt = gem.mcall(msgs)
        except Exception as e:
            # The model endpoint is down or rejecting us. Record it and STOP this question; do not
            # fall through to an empty answer, which would be scored as a wrong answer.
            steps.append({"i": step, "raw": "", "action": "error", "error": str(e)[:300]})
            failed = str(e)[:200]
            answer = None
            break

        cur = gem.q_get()
        d_in, d_out, d_rsn = (cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2])
        prev = cur

        kind, a, b, keep = _parse(txt)
        msgs.append({"role": "assistant", "parts": [gem.text_part(keep)]})

        rec = {"i": step, "raw": txt, "action": kind,
               "in_tok": d_in, "out_tok": d_out, "reasoning_tok": d_rsn,
               "t": round(time.time() - t0, 1)}

        if verbose:
            print("STEP %d | %s | %s" % (step, kind, (a or "")[:120]), flush=True)

        if kind == "answer" and n_calls == 0 and not force:
            # Answering with zero retrievals can only be recall from pretraining. Push back once.
            rec["error"] = "answer_before_any_retrieval"
            steps.append(rec)
            msgs.append({"role": "user", "parts": [gem.text_part(
                "You have not retrieved any evidence yet -- no <tool_output> has been sent to you, so "
                "anything you just wrote came from memory, not from this corpus. Do not answer yet. "
                "Emit exactly ONE <call_tool ...>...</call_tool> and then STOP.")]})
            continue

        if kind == "answer" or force:
            answer = a or ""
            rec.update({"tool": "finish", "answer": answer, "note": "forced" if force else ""})
            steps.append(rec)
            break

        if kind == "none":
            no_tag += 1
            if no_tag >= 2:
                answer = (a or "")[:2000]
                rec["error"] = "runaway_no_tag"
                steps.append(rec)
                break

        if kind == "none" or a not in tools:
            bad += 1
            rec.update({"tool": None,
                        "note": "invalid-tag(%s)" % (a[:40] if kind != "none" else "no-tag")})
            steps.append(rec)
            if bad > 3:
                answer = a or "ERROR: no valid tag"
                break
            msgs.append({"role": "user", "parts": [gem.text_part(
                'That was not a valid action. Emit EXACTLY one of:\n'
                '  <call_tool name="%s">your query</call_tool>\n'
                '  <answer>your answer</answer>' % "|".join(tools))]})
            continue

        tool, query = a, (b or "").strip()
        if not query:
            bad += 1
            rec.update({"tool": tool, "query": "", "note": "empty-query"})
            steps.append(rec)
            msgs.append({"role": "user",
                         "parts": [gem.text_part("Your tool call had an empty query. Try again.")]})
            continue

        query = _clean_query(query)

        try:
            hits = tools[tool].search(query, k=K_BY_TOOL[tool])
            err = None
        except Exception as e:
            hits, err = [], str(e)[:200]

        n_calls += 1
        calls_by_tool[tool] += 1

        if err:
            rec.update({"tool": tool, "query": query, "error": err, "hits": []})
            steps.append(rec)
            searches.append({"tool": tool, "query": query, "error": err, "hits": []})
            msgs.append({"role": "user", "parts": [
                gem.text_part("<tool_output>SEARCH ERROR: %s</tool_output>" % err)]})
            continue

        obs_parts, hit_recs = _format_obs(tool, hits, step)
        obs_parts.append(gem.text_part(
            "[tool calls used: %d/%d] If the exact value the question asks for is not visible in the "
            "results above, search again with a different query -- do NOT answer from memory."
            % (n_calls, budget)))
        rec.update({"tool": tool, "query": query, "error": None, "hits": hit_recs})
        steps.append(rec)
        searches.append({"tool": tool, "query": query, "error": None, "hits": hit_recs})
        msgs.append({"role": "user", "parts": obs_parts})

        over = _img_count(msgs) - MAX_IMAGES
        if over > 0:
            _drop_oldest_images(msgs, over)
            rec["images_dropped"] = over
        tok_dropped = _fit_context(msgs)
        if tok_dropped:
            rec["images_dropped"] = rec.get("images_dropped", 0) + tok_dropped

    qin, qout, qrsn = gem.q_get()
    return {"answer": answer, "failed": failed,
            "searches": searches,
            "steps": steps,
            "n_calls": n_calls,
            "calls_by_tool": dict(calls_by_tool),
            "in_tok": qin, "out_tok": qout, "reasoning_tok": qrsn,
            "wall_s": round(time.time() - t0, 1),
            "meta": {"model": gem.AGENT_MODEL, "budget": budget, "tools": list(tools)}}
