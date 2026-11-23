"""Serve the lab only on loopback; the desktop reaches it through SSH -L."""
import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

APP = Path(__file__).resolve().parent
lock = threading.Lock()
states = {
    key: dict(id="verified", status="ready", error=None) for key in ("depth", "scan")
}
modes = {"controlled", "mp3d", "room360", "laser360"}


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
        if path == "/api/live/model":
            from model import status

            return self.reply(200, status())
        if path.startswith("/api/live/"):
            try:
                match = re.fullmatch(
                    r"/api/live/([0-9a-f]{32})(?:/(export.zip|[0-9]{1,6}/[a-z.0-9]+))?",
                    path,
                )
                if not match:
                    return self.reply(404, {})
                sid, asset = match.groups()
                if asset is None:
                    return self.reply(200, self.server.live.state(sid))
                if asset == "export.zip":
                    with self.server.live.archive(sid) as data:
                        size = data.seek(0, 2)
                        data.seek(0)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/zip")
                        self.send_header(
                            "Content-Disposition",
                            f'attachment; filename="m1-{sid}.zip"',
                        )
                        self.send_header("Content-Length", str(size))
                        self.end_headers()
                        shutil.copyfileobj(data, self.wfile)
                    return
                step, name = asset.split("/")
                p = self.server.live.asset(sid, int(step), name)
                return self.send_bytes(
                    200,
                    p.read_bytes(),
                    mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                )
            except FileNotFoundError:
                return self.reply(404, {"error": "Frame unavailable"})
        if path in (
            "/live",
            "/live/",
            "/live/app.js",
            "/live/style.css",
            "/live/door.jpg",
            "/live/junction.jpg",
            "/live/reference.json",
        ):
            name = {"/live": "index.html", "/live/": "index.html"}.get(
                path, path.rsplit("/", 1)[-1]
            )
            file = APP.parent / "live" / "web" / name
            return self.send_bytes(
                200,
                file.read_bytes(),
                mimetypes.guess_type(name)[0] or "application/octet-stream",
            )
        if path in ("/api/status", "/api/scan-status"):
            state = states["scan" if path == "/api/scan-status" else "depth"]
            with lock:
                return self.reply(200, dict(state))
        if path in ("/", "/depth", "/laser.js", "/laser.css", "/app.js", "/style.css"):
            file = (
                APP
                / "web"
                / ({"/": "laser.html", "/depth": "index.html"}.get(path, path[1:]))
            )
        elif path in ("/junction", "/junction/"):
            file = APP.parents[1] / "results/junction/index.html"
        elif path in ("/region-entry", "/region-entry/"):
            file = APP.parents[1] / "results/region-entry/index.html"
        elif path in ("/mp3d/topdown.jpg", "/mp3d/topdown.json"):
            file = (
                APP.parents[1] / "results/m1-mp3d-2azQ1b91cZZ" / path.rsplit("/", 1)[1]
            )
        else:
            match = re.fullmatch(
                r"/runs/(verified|[0-9a-f]{12})/(controlled|mp3d|room360|laser360)/(steps\.json|[0-9]+\.json|rgb-[0-9]+\.jpg|depth-[0-9]+\.png|archive\.jpg)",
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
        if self.path not in ("/api/run", "/api/scan", "/api/live/control"):
            return self.reply(404, {})
        if (
            self.headers.get("Origin") != "http://" + self.headers.get("Host", "")
            or self.headers.get("X-M1-Lab") != "run"
        ):
            return self.reply(403, {})
        if self.path == "/api/live/control":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    return self.reply(400, {"error": "Invalid command size"})
                cmd = json.loads(self.rfile.read(size))
                if not isinstance(cmd, dict) or cmd.get("action") not in (
                    "start",
                    "forward",
                    "back",
                    "left",
                    "right",
                    "observe",
                    "frontier",
                    "retry",
                ):
                    return self.reply(400, {"error": "Unknown action"})
                state = self.server.live.move(
                    cmd["session"],
                    cmd["action"],
                    cmd.get("expected"),
                    cmd.get("preset", "door"),
                    cmd.get("frontier"),
                )
                return self.reply(200, state)
            except (ValueError, KeyError) as exc:
                return self.reply(409, {"error": str(exc)})
            except FileNotFoundError:
                return self.reply(404, {"error": "Session unavailable"})
            except Exception as exc:
                import traceback

                traceback.print_exc()
                return self.reply(500, {"error": str(exc)})
        with lock:
            group = "scan" if self.path == "/api/scan" else "depth"
            state = states[group]
            if any(job["status"] == "running" for job in states.values()):
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
                                "--mode",
                                "scans" if group == "scan" else "both",
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
    sys.path.insert(0, str(APP.parent / "live"))
    from live_api import LiveHub

    server.live = LiveHub(server.runs / "live", args.gpu)
    server.serve_forever()
