"""Serve the lab only on loopback; the desktop reaches it through SSH -L."""
import argparse
import json
import mimetypes
import re
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

APP = Path(__file__).resolve().parent
lock = threading.Lock()
state = dict(id="verified", status="ready", error=None)
modes = {"controlled", "mp3d"}


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, code, data, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def reply(self, code, obj):
        self.send_bytes(
            code,
            json.dumps(obj, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
        )

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            with lock:
                return self.reply(200, dict(state))
        if path in ("/", "/app.js", "/style.css"):
            file = APP / "web" / ("index.html" if path == "/" else path[1:])
        else:
            match = re.fullmatch(
                r"/runs/(verified|[0-9a-f]{12})/(controlled|mp3d)/(steps\.json|[0-9]+\.json|rgb-[0-9]+\.jpg|depth-[0-9]+\.png|archive\.jpg)",
                path,
            )
            if not match:
                return self.reply(404, {"error": "Unknown route"})
            file = self.server.runs.joinpath(*match.groups())
        if not file.is_file():
            return self.reply(404, {"error": "Snapshot unavailable"})
        self.send_bytes(
            200,
            file.read_bytes(),
            mimetypes.guess_type(file.name)[0] or "application/octet-stream",
        )

    def do_POST(self):
        if self.path != "/api/run":
            return self.reply(404, {})
        if (
            self.headers.get("Origin") != "http://" + self.headers.get("Host", "")
            or self.headers.get("X-M1-Lab") != "run"
        ):
            return self.reply(403, {})
        with lock:
            if state["status"] == "running":
                return self.reply(409, dict(state))
            run_id = uuid.uuid4().hex[:12]
            folder = self.server.runs / run_id
            folder.mkdir()
            state.update(id=run_id, status="running", error=None)

            def execute():
                try:
                    with (folder / "run.log").open("w") as out:
                        subprocess.run(
                            [
                                sys.executable,
                                str(APP / "run.py"),
                                str(folder),
                                "--gpu",
                                str(self.server.gpu),
                            ],
                            stdout=out,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    result = dict(status="ready", error=None)
                except Exception as exc:
                    log = (
                        (folder / "run.log")
                        .read_text(errors="replace")
                        .strip()
                        .splitlines()
                    )
                    result = dict(status="failed", error=log[-1] if log else str(exc))
                with lock:
                    state.update(result)

            threading.Thread(target=execute, daemon=True).start()
            self.reply(202, dict(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--port", type=int, default=18764)
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    for mode in modes:
        steps = json.loads((args.runs / "verified" / mode / "steps.json").read_text())
        if not steps or not all(step["passed"] for step in steps):
            raise RuntimeError("Missing verified baseline")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.runs, server.gpu = args.runs.resolve(), args.gpu
    server.serve_forever()
