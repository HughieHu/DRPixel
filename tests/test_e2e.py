"""Every config end to end against the offline mocks: the real client, harness and metrics code."""
import json
import os
import subprocess
import sys

import pytest

import mock_servers
from conftest import EVAL

CASES = [("Text", "search_text"), ("Page", "search_page"), ("Text_Fig", "search_figure")]


@pytest.mark.parametrize("config,tool", CASES)
@pytest.mark.parametrize("strict", [False, True])
def test_config_end_to_end(tmp_path, config, tool, strict):
    search_url, model_url, stop = mock_servers.start(tool, strict=strict)
    try:
        out = tmp_path / "out.json"
        env = dict(os.environ, SEARCH_URL=search_url, MODEL_NAME="mock-model", MODEL_BASE_URL=model_url,
                   MODEL_API_KEY="EMPTY")
        r = subprocess.run([sys.executable, "-m", "drbench.eval", "--config", config.lower(), "--limit", "5",
                            "--no-grade", "--out", str(out)], cwd=EVAL, env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        stop()
    d = json.load(open(out))
    assert d["config"] == config and len(d["results"]) == 5
    for rec in d["results"]:
        gold = mock_servers.BY_Q[rec["qid"]]
        assert not rec["failed"] and rec["answer"] == gold["gold_answer"][:300]   # stripped </answer> repaired
        assert rec["n_calls"] == 1 and rec["hits"]["doc_hit"] is True
        hits = rec["searches"][0]["hits"]
        if config == "Text":
            assert rec["hits"]["page_hit"] is None
            assert all(h["text"].endswith("END%d" % i) for i, h in enumerate(hits))   # passages arrive uncut
        else:
            assert rec["hits"]["page_hit"] is True
            assert all(h["has_image"] for h in hits)
