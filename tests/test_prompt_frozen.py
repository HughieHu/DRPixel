"""The system prompt, tool descriptions, k, tool budget and image cap are what produced the paper's
numbers. Changing any of them makes results incomparable, so a change must be deliberate: copy the
new fingerprint from the failing assertion into FROZEN and say why in the commit."""
import os
import subprocess
import sys

from conftest import EVAL

CODE = ("import hashlib, json; from drbench import harness as h; "
        "print(hashlib.sha256(json.dumps({'s': h.SYSTEM, 'i': h.IMG_HINT, 't': h.TOOL_DESCS, 'o': h.OBS_DESCS, "
        "'k': h.K_BY_TOOL, 'b': h.BUDGET, 'm': h.MAX_IMAGES}, sort_keys=True).encode()).hexdigest())")
FROZEN = {
    "gpt-5.4": "aeeaa121ae7fd38bc440f1e0d084cfb3b5aeb6828b58d05e35646c52406f5271",     # GPT persona
    "qwen3-27b": "fcf76bf43b2622b062c7a7521b13b4342b9987cf7d7dff642654ab97c5f6ed5e",   # every other model
}


def test_prompt_unchanged():
    for model, want in FROZEN.items():
        env = dict(os.environ, MODEL_NAME=model)
        got = subprocess.check_output([sys.executable, "-c", CODE], cwd=EVAL, env=env).decode().strip()
        assert got == want, "system prompt / tool setup changed for %s" % model
