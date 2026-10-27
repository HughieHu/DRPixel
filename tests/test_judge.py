import hashlib
import json
import os

from drbench import judge

BENCH = os.path.join(os.path.dirname(__file__), "..", "eval", "bench", "eval500.json")

GOLD = {"doc": "d1", "page": 3}


def hit(doc, page=None):
    h = {"doc": doc}
    if page is not None:
        h["page_idx"] = page
    return h


def test_judge_prompt_unchanged():
    p = judge._JUDGE.format(atype="string", q="Q?", gold="G.", pred="P.")
    assert hashlib.sha256(p.encode()).hexdigest() == \
        "a67beacd1eb451cc5a74f8f8c8f3d74a6d6b3263918eead0440a859762ab222f"
    assert (judge.JUDGE_MODEL, judge.JUDGE_VOTES, judge.JUDGE_EFFORT) == ("gpt-5.4-mini", 3, "low")


PAGE, TEXT = {"search_page": "page"}, {"search_text": "text"}
TEXT_FIG = {"search_text": "text", "search_figure": "figure"}


def search(tool, *hits):
    return {"tool": tool, "hits": list(hits)}


def test_page_hit_convention():
    rh = judge.retrieval_hits
    assert rh(GOLD, [search("search_page", hit("d1", 3))], PAGE)["page_hit"] is True
    assert rh(GOLD, [search("search_page", hit("d1", 4))], PAGE)["page_hit"] is False
    assert rh(GOLD, [], PAGE)["page_hit"] is False                                   # Page, no search: a miss
    assert rh(GOLD, [search("search_page", hit("d1"))], PAGE)["page_hit"] is False    # a hit without a page is a miss
    assert rh(GOLD, [], TEXT)["page_hit"] is None                                    # Text: never measurable
    assert rh(GOLD, [search("search_text", hit("d1"))], TEXT_FIG)["page_hit"] is None  # Text_Fig, no figure search
    assert rh(GOLD, [search("search_figure", hit("d1", 3))], TEXT_FIG)["page_hit"] is True
    assert rh(GOLD, [search("search_text", hit("d1"))], TEXT)["doc_hit"] is True


def test_every_question_has_a_gold_page():
    bench = json.load(open(BENCH))
    assert len(bench) == 500 and len({b["qid"] for b in bench}) == 500
    assert all(isinstance(b["page"], int) and b["page"] >= 0 for b in bench)


def test_majority_and_failed_votes(monkeypatch):
    votes = iter([(True, "a"), None, (False, "b"), (True, "c")])
    monkeypatch.setattr(judge, "_vote", lambda prompt: next(votes))
    assert judge.judge_answer("q", "g", "p")["correct"] is False          # one vote failed: 1-1 tie goes to wrong
    assert judge.judge_answer("q", "g", "p", votes=1)["correct"] is True
    monkeypatch.setattr(judge, "_vote", lambda prompt: None)
    assert judge.judge_answer("q", "g", "p")["correct"] is None           # judge never answered: not a miss
    assert judge.judge_answer("q", "g", "")["correct"] is False           # no answer: wrong
