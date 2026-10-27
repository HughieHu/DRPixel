"""Register the `stepw` loss scale into an installed ms-swift.

Run once against the environment you will train in:

    python register_stepw.py                      # auto-detects the installed swift
    python register_stepw.py /path/to/site-packages/swift

The plugin is a LossScale with `is_binary = False`, so that swift uses each message's numeric
`loss_scale` as is instead of collapsing it to 0/1. See train/README.md for how to check that the
weights reached the trainer.
"""
import os
import sys

CLASS_SRC = '''# Added by DRPixel register_stepw.py -- see that script's docstring.
from .base import LossScale


class StepWeightLossScale(LossScale):
    """Honour the per-message numeric `loss_scale` field verbatim.

    is_binary = False is the whole point: the default collapses the scale to 0/1 and throws the
    per-step weights away.
    """

    is_binary = False
'''


def find_swift():
    try:
        import swift
        return os.path.dirname(swift.__file__)
    except ImportError:
        sys.exit("cannot import swift -- activate the training environment, or pass its path:\n"
                 "  python register_stepw.py /path/to/site-packages/swift")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else find_swift()
    ld = os.path.join(root, "loss_scale")
    mapping = os.path.join(ld, "mapping.py")
    if not os.path.isfile(mapping):
        sys.exit("not a swift install (no loss_scale/mapping.py): %s" % root)

    mod = os.path.join(ld, "stepw.py")
    with open(mod, "w") as f:
        f.write(CLASS_SRC)
    print("wrote", mod)

    s = open(mapping).read()
    if "loss_scale_map = {" not in s:
        sys.exit("unsupported swift version: %s has no `loss_scale_map = {`" % mapping)
    changed = False
    if "from .stepw import StepWeightLossScale" not in s:
        s = s.replace("loss_scale_map = {",
                      "from .stepw import StepWeightLossScale\n\nloss_scale_map = {", 1)
        changed = True
    if "'stepw'" not in s:
        s = s.replace("loss_scale_map = {",
                      "loss_scale_map = {\n    'stepw': StepWeightLossScale,", 1)
        changed = True
    if changed:
        with open(mapping, "w") as f:
            f.write(s)
        print("patched", mapping)
    else:
        print("mapping.py already registers stepw")

    # Confirm it resolves, and that it is NOT binary. A silent failure here is the whole hazard.
    try:
        from swift.loss_scale.mapping import get_loss_scale
        ls = get_loss_scale("stepw")
        binary = getattr(ls, "is_binary", getattr(ls, "is_binary_loss_scale", None))
        print("get_loss_scale('stepw') -> %s  is_binary=%s" % (type(ls).__name__, binary))
        if binary:
            sys.exit("FATAL: stepw resolved but reports is_binary=True -- per-step weights would "
                     "be discarded. Do not train until this says False.")
        print("OK: per-message loss_scale values will be used verbatim.")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit("could not verify the registration (%s: %s). Check manually: "
                 "get_loss_scale('stepw').is_binary must be False." % (type(e).__name__, str(e)[:120]))


if __name__ == "__main__":
    main()
