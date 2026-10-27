"""drbench — evaluate any model endpoint on DRPixelBench."""
import os
import ssl


def _load_dotenv():
    """Load .env before anything else in the package imports.

    This lives in __init__ on purpose: `model.py` reads MODEL_NAME/MODEL_BASE_URL at import
    time, so loading .env from inside eval.py would be too late — the values would already
    have been read as empty. Real environment variables always win over the file.
    """
    for d in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        p = os.path.join(d, ".env")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                        v = v[1:-1]
                    # A blank value means "leave it at the default", which is how people comment
                    # a setting out. Setting it to "" instead would make int()/float() throw at
                    # import time.
                    if v:
                        os.environ.setdefault(k, v)
        except OSError:
            pass
        return p
    return None


_load_dotenv()


def envint(name, default):
    """int from the environment, tolerating a blank value. A blank is treated as unset rather
    than as an error, so an exported-but-empty variable cannot crash us at import time."""
    v = os.environ.get(name, "")
    try:
        return int(str(v).strip()) if str(v).strip() else int(default)
    except ValueError:
        return int(default)


def ssl_context():
    """TLS context using certifi's CA bundle when available (stale system bundles are common on macOS)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def envfloat(name, default):
    v = os.environ.get(name, "")
    try:
        return float(str(v).strip()) if str(v).strip() else float(default)
    except ValueError:
        return float(default)
