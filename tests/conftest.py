import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "eval")
TRAIN = os.path.join(ROOT, "train")
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JUDGE_API_KEY", "test")       # judge.py reads it at import; never used offline
