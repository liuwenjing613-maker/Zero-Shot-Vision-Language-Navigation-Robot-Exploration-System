"""Pinned VLN vision backend. A missing or cold model never selects another model."""
import json
from urllib.request import Request, urlopen

MODEL = "qwen3.8:27b"
HOST = "http://127.0.0.1:18281"


def request(path, data=None, timeout=5):
    r = Request(
        HOST + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(r, timeout=timeout) as response:
        return json.load(response)


def status():
    try:
        models = request("/api/ps")["models"]
        m = next(
            (m for m in models if m.get("name") == MODEL or m.get("model") == MODEL), {}
        )
        return dict(
            model=MODEL,
            resident=bool(m),
            vram_bytes=m.get("size_vram", 0),
            expires_at=m.get("expires_at"),
            digest=m.get("digest"),
        )
    except (OSError, ValueError, KeyError) as exc:
        return dict(model=MODEL, resident=False, error=str(exc))


def warm():
    names = [r["id"] for r in request("/v1/models")["data"]]
    if MODEL not in names:
        raise RuntimeError("Project model qwen3.8:27b is unavailable on port 18281")
    result = request(
        "/api/generate", dict(model=MODEL, keep_alive=-1, stream=False), timeout=300
    )
    if result.get("model") != MODEL or not result.get("done"):
        raise RuntimeError("Model warmup did not finish")
    return status()


def client():
    if not status()["resident"]:
        warm()
    from fast_vision import FastVision

    return FastVision()
