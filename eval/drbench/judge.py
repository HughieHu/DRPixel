"""Grading and metrics.

Two separate things live here:

* **Answer grading** — an LLM judge, because the gold answers are sentences containing specific
  values, not short spans: `gpt-5.4-mini` on the **Responses** API with `reasoning.effort = "low"`,
  three independent votes, majority — the paper's judge, call for call.
* **Retrieval metrics** — pure bookkeeping over the hits the run already recorded, no API calls.
  Free, deterministic, and re-runnable on an existing results file.

The judge rubric below is byte-identical to the paper's, and so is the request (see
`_one_vote`). Changing either makes your Acc incomparable to the reported numbers.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from . import ssl_context

# The judge is fixed so that scores are comparable; only where it is reached is configurable.
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_VOTES = 3
JUDGE_EFFORT = "low"
JUDGE_MAX_OUTPUT = 3000
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
_CTX = ssl_context()

_JUDGE = """You STRICTLY grade an answer to a question about a research paper's figure/table/finding.
The GOLD states a SPECIFIC finding one can only know by reading the figure/table -- often exact
numbers, named models/methods/datasets, or a precise quantitative relationship.

GRADING RULE:
1. If the GOLD contains specific numbers, named entities (models/methods/datasets), or a precise
   magnitude, mark correct=true ONLY IF the PREDICTED answer states that SAME specific finding:
   the same numbers (within rounding) AND/OR the same named entities AND/OR the same precise
   relationship. Otherwise correct=FALSE. In particular:
   - Generic / directional answer that gets the trend right but OMITS the gold's specific numbers => WRONG.
   - Answer citing DIFFERENT numbers or DIFFERENT models/methods than the gold => WRONG (it guessed
     from prior knowledge, did not read the source).
   - Vague hedging ("levels off", "tends to", "generally", "competitive") in place of the gold's
     specific finding => WRONG.
2. If the GOLD has NO specific numbers/entities (a pure yes/no verdict or a purely qualitative
   direction), then a correctly matching verdict/direction is enough.
Paraphrase of the SAME specific finding is fine; extra correct detail is fine; a contradicted or
reversed direction is always WRONG.

answer_type = {atype} (verdict / comparison / trend / regime).

CALIBRATION EXAMPLES:
[CORRECT] GOLD: "No, it stays stable..." (no numbers) | PRED: "No, ... remains bounded, avoids collapse" -> verdict matches, gold has no specifics.
[CORRECT] GOLD: "operational factors; limiters = latent size and number of modules" | PRED: "operational overhead; size and number of processed tensors" -> states the same specific finding.
[WRONG] GOLD: "Jina-ColBERT-v2 within ~0.4 nDCG@10 of mDPR-FT on MIRACL" | PRED: "a single model such as CORA or mAggretriever is competitive" -> different models, none of the gold's specifics.
[WRONG] GOLD: "EMOVA 72B trails on RealWorldQA 71.0 vs 75.4" | PRED: "InternVL3-78B, Qwen-VL2.5-72B trail GPT-4o" -> different models, missing 71.0/75.4.
[WRONG] GOLD: "edge vanishes: 66.6 vs 62.7 (Qwen2.5) but 70.3 vs 70.9 (Qwen3)" | PRED: "the advantage diminishes" -> correct direction but NONE of the gold's specific numbers.
[WRONG] GOLD: "rises 2->4, peaks ~4, dips at 5" | PRED: "levels off / plateaus" -> vague, misses the specific peak/dip.

QUESTION: {q}
GOLD ANSWER: {gold}
PREDICTED ANSWER: {pred}
First state the gold's KEY specific finding, then decide.
Output ONLY: {{"gold_key_finding": "<the specific finding the gold makes>", "correct": true or false, "reason": "<one sentence: does PRED state it?>"}}"""


class JudgeError(RuntimeError):
    pass


_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _one_vote(prompt):
    """One judge call. Returns (correct, reason) or None if the call did not produce a verdict.

    This is the paper's judge call, and the details matter:

    * the **Responses** API, not chat-completions;
    * `reasoning.effort = "low"`;
    * **no `temperature`** and **no `response_format`** — the three votes are three identical
      requests, and their disagreement comes from the reasoning model's own non-determinism, not
      from a sampling temperature. Adding temperature would be a different judge;
    * the JSON object is pulled out of the text with a regex rather than requested as a schema.

    Returning None (rather than False) on failure is also load-bearing — see `judge_answer`.
    """
    body = {"model": JUDGE_MODEL,
            "input": [{"role": "user", "content": prompt}],
            "reasoning": {"effort": JUDGE_EFFORT},
            "max_output_tokens": JUDGE_MAX_OUTPUT}
    req = urllib.request.Request(
        JUDGE_BASE_URL + "/responses", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + JUDGE_API_KEY})
    with urllib.request.urlopen(req, timeout=180, context=_CTX) as resp:
        d = json.load(resp)
    txt = "".join(c.get("text", "") for o in (d.get("output") or [])
                  for c in (o.get("content") or []) if c.get("type") == "output_text")
    m = _OBJ_RE.search(txt or "")
    if not m:
        return None
    try:
        r = json.loads(m.group(0))
    except ValueError:
        return None
    return bool(r.get("correct")), str(r.get("reason", ""))[:200]


def _vote(prompt):
    """Retry transient failures; give up with None rather than guessing a verdict."""
    for attempt in range(4):
        try:
            return _one_vote(prompt)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                try:
                    detail = e.read().decode()[:200]
                except Exception:
                    detail = ""
                raise JudgeError("judge endpoint rejected the key (HTTP %s %s). Set JUDGE_API_KEY "
                                 "or OPENAI_API_KEY." % (e.code, detail))
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def judge_answer(question, gold, pred, votes=None):
    """Majority over the votes that came back. Returns {correct, reason, votes}.

    `correct` is **None** when every vote failed. That is not the same as False: such questions are
    dropped from the denominator rather than scored wrong, because a dead judge endpoint is not
    evidence about the model. `metrics()` honours that by counting only
    records whose `correct` is not None.

    Ties go to wrong: with two surviving votes split 1-1, `sum > len/2` is false.
    """
    pred = (pred or "").strip()
    if not pred or pred.startswith("ERROR:"):
        # An empty or errored answer IS wrong — the model had its chance.
        return {"correct": False, "reason": "no answer / error", "votes": []}
    n = votes or JUDGE_VOTES
    prompt = _JUDGE.format(atype="string", q=question, gold=gold, pred=pred)
    got = [v for v in (_vote(prompt) for _ in range(n)) if v is not None]
    if not got:
        return {"correct": None, "reason": "judge produced no verdict (all %d votes failed)" % n,
                "votes": []}
    flags = [c for c, _ in got]
    correct = sum(flags) > len(flags) / 2.0
    reason = next((rs for c, rs in got if c == correct), got[0][1])
    return {"correct": correct, "reason": reason, "votes": flags}


# ------------------------------------------------------------------ retrieval metrics (no API)
PAGE_TOOLS = ("search_page", "search_figure")       # the tools whose hits carry a page


def retrieval_hits(item, searches, tools):
    """Did the run ever retrieve the gold document / the gold page?

    `item["page"]` is the gold page's 0-based index in the PDF — the same index hits report as
    `page_idx`, so they compare directly. `tools` are the config's tools.

    `page_hit` is None — not False — when a page hit cannot be measured, which is the convention
    behind every reported PgHit:
      * the config returns no pages (`Text`: a passage carries `chunk_idx`, never a page);
      * under `Text_Fig`, the agent never called `search_figure`, the only tool that returns a page.
    Under `Page` it is always measured; a question that ended before its first search is a miss.
    """
    gold_doc, gold_page = item["doc"], item.get("page")
    all_hits = [h for s in searches for h in (s.get("hits") or [])]
    doc_hit = any(h.get("doc") == gold_doc for h in all_hits)
    page_tools = [t for t in tools if t in PAGE_TOOLS]
    measurable = gold_page is not None and page_tools and (
        "search_page" in page_tools or any(s.get("tool") in page_tools for s in searches))
    page_hit = any(h.get("doc") == gold_doc and h.get("page_idx") == gold_page
                   for h in all_hits) if measurable else None
    return {"doc_hit": doc_hit, "page_hit": page_hit}


def _pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def _payload(results):
    """How many returned hits actually carried evidence?

    A hit with neither text nor an image is a search that cost a turn and returned nothing the
    model could read. That happens for real — a doc whose OCR chunk is empty, or a figure hit
    with no crop behind it — and it is invisible in `Tool/q`, which counts the call either way.
    An endpoint serving thin payloads therefore looks exactly like a weak model, so the rate is
    reported rather than left to be discovered.
    """
    empty = total = 0
    for r in results:
        for s in (r.get("searches") or []):
            for h in (s.get("hits") or []):
                if (h.get("text") or "").strip():
                    total += 1
                    continue
                total += 1
                if not h["has_image"]:
                    empty += 1
    return empty, total


def metrics(results):
    """The paper's reported quantities.

    PgHit is the share of questions whose gold page was retrieved, over the questions where a page
    hit is measurable (see `retrieval_hits`). Under `Page` that is every question; under
    `Text_Fig` it is the questions on which the agent searched figures at least once, so
    `PgHit_all` (over all questions) is reported alongside it.
    Acc|Pg is accuracy restricted to questions whose gold page was retrieved — the "given it found
    the page, could it read it?" quantity. Do NOT compare Acc|Pg against a document-level
    conditional accuracy; they have different denominators and are not the same measurement.
    """
    n = len(results)
    graded = [r for r in results if r.get("correct") is not None]
    with_page = [r for r in results if (r.get("hits") or {}).get("page_hit") is not None]
    pg = [r for r in with_page if r["hits"]["page_hit"]]
    pg_graded = [r for r in pg if r.get("correct") is not None]
    n_empty, n_hits = _payload(results)
    return {
        "n_hits": n_hits,
        "empty_hits": n_empty,
        "empty_hit_pct": _pct(n_empty, n_hits),
        "n": n,
        "n_graded": len(graded),
        "n_pageable": len(with_page),
        "n_failed": sum(1 for r in results if r.get("failed")),
        "Acc": _pct(sum(bool(r.get("correct")) for r in graded), len(graded)),
        "Tool/q": round(sum(r.get("n_calls") or 0 for r in results) / n, 2) if n else None,
        "DocHit": _pct(sum(bool((r.get("hits") or {}).get("doc_hit")) for r in results), n),
        "PgHit": _pct(len(pg), len(with_page)),
        "PgHit_all": _pct(len(pg), n),
        "Acc|Pg": _pct(sum(bool(r.get("correct")) for r in pg_graded), len(pg_graded)),
        "in_tok/q": round(sum(r.get("in_tok") or 0 for r in results) / n) if n else None,
        "out_tok/q": round(sum(r.get("out_tok") or 0 for r in results) / n) if n else None,
    }


def by_modality(results):
    out = {}
    for m in sorted({r.get("modality") for r in results if r.get("modality")}):
        rs = [r for r in results if r.get("modality") == m]
        mm = metrics(rs)
        out[m] = {k: mm[k] for k in ("n", "Acc", "Tool/q", "PgHit", "Acc|Pg")}
    return out


def format_table(m, by_mod=None, graded=True):
    L = []
    if graded:
        L.append("  Acc          %s%%   (%s x%d majority, %d/%d graded)"
                 % (m["Acc"], JUDGE_MODEL, JUDGE_VOTES, m["n_graded"], m["n"]))
    else:
        L.append("  Acc          not graded (--no-grade)")
    L.append("  Tool/q       %s     (tool calls per question; the answer turn is not a tool call)"
             % m["Tool/q"])
    if m["PgHit"] is None:
        L.append("  PgHit        n/a     this config never returns a page, so a page hit is not "
                 "measurable")
        L.append("  DocHit       %s%%" % m["DocHit"])
    else:
        if m["n_pageable"] == m["n"]:
            L.append("  PgHit        %s%%" % m["PgHit"])
        else:
            L.append("  PgHit        %s%%   over the %d questions where it is measurable  |  %s%% over all %d"
                     % (m["PgHit"], m["n_pageable"], m["PgHit_all"], m["n"]))
        L.append("  DocHit       %s%%" % m["DocHit"])
        L.append("  Acc|Pg       %s%%   accuracy when the gold page WAS retrieved" % m["Acc|Pg"])
    L.append("  tokens/q     %s in / %s out" % (m["in_tok/q"], m["out_tok/q"]))
    if m["n_failed"]:
        L.append("  !! %d questions errored out and are excluded from Acc" % m["n_failed"])
    # Questions the model answered but the judge never returned a verdict on. Separate from
    # model errors, because the cause is our side, not the model's.
    n_nojudge = m["n"] - m["n_graded"] - m["n_failed"]
    if graded and n_nojudge > 0:
        L.append("  !! %d questions got no judge verdict and are excluded from Acc -- rerun with "
                 "--grade-only to retry them" % n_nojudge)
    if m["empty_hit_pct"]:
        L.append("  !! %s%% of returned hits (%d/%d) carried no text and no image -- those "
                 "searches returned nothing readable" % (m["empty_hit_pct"], m["empty_hits"],
                                                         m["n_hits"]))
    if by_mod:
        L.append("")
        L.append("  %-10s %5s %6s %7s %7s" % ("modality", "n", "Acc", "Tool/q", "Acc|Pg"))
        for k, v in by_mod.items():
            L.append("  %-10s %5d %6s %7s %7s" % (k, v["n"], v["Acc"], v["Tool/q"], v["Acc|Pg"]))
    return "\n".join(L)
