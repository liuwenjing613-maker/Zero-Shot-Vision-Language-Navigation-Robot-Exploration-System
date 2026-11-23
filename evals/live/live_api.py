"""Durable action log, one Habitat owner and an ordered semantic-memory consumer."""
import os, hashlib, queue, re, threading, time, uuid, zipfile, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from world import World, frames, read, write
from model import MODEL, client, warm, status
from vln_memory.online_memory import OnlineMemory


class LiveHub:
    def __init__(self, root, gpu):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="habitat")
        self.world = World(gpu)
        self.jobs = queue.Queue()
        self.scheduled = set()
        self.lock = threading.Lock()
        self.reducer = None
        self.loaded = None
        threading.Thread(target=self.consume, daemon=True, name="m1-vlm").start()
        threading.Thread(target=self.prewarm, daemon=True, name="warm-27b").start()
        folders = set()
        for folder in list(root.iterdir()):
            if (
                re.fullmatch("[0-9a-f]{32}", folder.name)
                and (folder / "config.json").is_file()
            ):
                folders.add(self.upgraded(folder))
        for folder in sorted(folders):
            if self.pending(folder):
                self.schedule(folder)

    @staticmethod
    def prewarm():
        try:
            warm()
        except Exception as exc:
            print("27B warmup:", exc, flush=True)

    def upgraded(self, folder):
        cfg = read(folder / "config.json")
        if cfg["model"] == MODEL:
            return folder
        if cfg["model"] != "qwen3-vl:8b":
            raise ValueError("Unexpected legacy model")
        sid = uuid.uuid5(uuid.NAMESPACE_URL, "vln-m1/" + folder.name + "/" + MODEL).hex
        dest = self.root / sid
        if not dest.exists():
            tmp = self.root / (".migrate-" + uuid.uuid4().hex)
            tmp.mkdir()
            write(
                tmp / "config.json",
                dict(cfg, session=sid, model=MODEL, source_session=folder.name),
            )
            for f in frames(folder):
                z = tmp / f.name
                z.mkdir()
                for name in (
                    "sensors.npz",
                    "rgb.png",
                    "view.jpg",
                    "map.png",
                    "grid.npz",
                ):
                    if (f / name).exists():
                        os.link(f / name, z / name)
                st = read(f / "step.json")
                write(
                    z / "step.json",
                    dict(st, session=sid, observation_id=f"{sid}-{st['step']:05}"),
                )
            os.rename(tmp, dest)
        return dest

    def folder(self, sid):
        if not isinstance(sid, str) or not re.fullmatch("[0-9a-f]{32}", sid):
            raise ValueError("Invalid session")
        folder = self.root / sid
        return self.upgraded(folder) if (folder / "config.json").exists() else folder

    def pending(self, folder):
        return next(
            (p for p in frames(folder) if not (p / "memory.json").exists()), None
        )

    def schedule(self, folder):
        with self.lock:
            if folder not in self.scheduled:
                self.scheduled.add(folder)
                self.jobs.put(folder)

    def move(self, sid, action, expected=None, preset="door", choice=None):
        folder = self.folder(sid)

        def command():
            if action == "start":
                if not folder.exists():
                    folder.mkdir()
                if frames(folder):
                    return
                if preset not in ("door", "junction"):
                    raise ValueError("Invalid start")
                self.world.open(folder, preset)
            elif not (folder / "config.json").is_file():
                raise FileNotFoundError("Unknown run")
            if shutil.disk_usage(self.root).free < 200 * 1024 * 1024:
                raise ValueError("服务器存储空间不足")
            try:
                self.world.capture(
                    folder, action, -1 if action == "start" else expected, choice
                )
            except Exception:
                self.world.folder = None
                raise

        if action == "retry":
            if not folder.exists():
                raise FileNotFoundError("Unknown run")
            f = self.pending(folder)
            if f and (f / "error.json").exists():
                (f / "error.json").unlink()
        else:
            self.owner.submit(command).result()
        self.schedule(folder)
        return self.state(sid)

    def process(self, folder, f):
        cfg = read(folder / "config.json")
        frame = read(f / "step.json")
        if (
            hashlib.sha256((f / "rgb.png").read_bytes()).hexdigest()
            != frame["rgb_sha256"]
        ):
            raise ValueError("RGB checksum mismatch")
        cache = folder / ("rgb-" + frame["rgb_sha256"] + ".json")
        started = time.time()
        write(f / "progress.json", dict(started=started))
        if not (f / "vlm.json").exists():
            timing = None
            reused_from = None
            if cache.exists():
                result = read(cache)
                reused_from = result["observation_id"]
                result["observation_id"] = frame["observation_id"]
                source = "identical_rgb"
            else:
                if cfg["model"] != MODEL:
                    raise ValueError("Wrong model for live session")
                model = client()
                result = model.perceive_scene(
                    str(f / "rgb.png"),
                    cfg["goal"],
                    task_id=folder.name,
                    observation_id=frame["observation_id"],
                    view_id="front",
                )
                timing = model.timing
                write(cache, result)
                source = "live_model"
            write(
                f / "vlm.json",
                dict(
                    result=result,
                    source=source,
                    reused_from=reused_from,
                    timing=timing,
                    rgb_sha256=frame["rgb_sha256"],
                ),
            )
        vlm = read(f / "vlm.json")
        if self.loaded != folder or self.reducer.last != frame["step"] - 1:
            self.loaded = None
            self.reducer = OnlineMemory(cfg)
            for old in frames(folder)[: frame["step"]]:
                with np.load(old / "sensors.npz") as data:
                    self.reducer.apply(
                        read(old / "step.json"), data, read(old / "vlm.json")["result"]
                    )
            self.loaded = folder
        with np.load(f / "sensors.npz") as data:
            memory = self.reducer.apply(frame, data, vlm["result"])
        memory["rgb_sha256"] = frame["rgb_sha256"]
        # Published last: a memory.json is the commit record for this step.
        write(f / "memory.json", memory)

    def consume(self):
        while True:
            folder = self.jobs.get()
            try:
                f = self.pending(folder)
                if f and not (f / "error.json").exists():
                    for attempt in range(2):
                        try:
                            self.process(folder, f)
                            break
                        except Exception as exc:
                            self.loaded = None
                            if attempt:
                                write(
                                    f / "error.json",
                                    dict(message=str(exc), step=f.name, attempts=2),
                                )
            finally:
                with self.lock:
                    self.scheduled.discard(folder)
                f = self.pending(folder)
                if f and not (f / "error.json").exists():
                    self.schedule(folder)

    def state(self, sid):
        folder = self.folder(sid)
        cfg = folder / "config.json"
        if not cfg.is_file():
            raise FileNotFoundError("Unknown run")
        fs = frames(folder)
        history = []
        last = -1
        trajectory = []
        memory = None
        for i, f in enumerate(fs):
            s = read(f / "step.json")
            trajectory.append(s["xy"])
            status = "queued"
            events = []
            source = None
            error = None
            started = None
            if (f / "memory.json").exists():
                status = "ready"
                last = i
                memory = read(f / "memory.json")
                events = memory["events"]
                source = read(f / "vlm.json")["source"]
            elif (f / "error.json").exists():
                status = "error"
                error = read(f / "error.json")["message"]
            elif (f / "progress.json").exists():
                status = "working"
                started = read(f / "progress.json")["started"]
            history.append(
                dict(
                    step=i,
                    action=s["action"],
                    status=status,
                    events=events,
                    source=source,
                    error=error,
                    started=started,
                )
            )
        return dict(
            config=read(cfg),
            frame=read(fs[-1] / "step.json") if fs else None,
            memory=memory,
            committed=last,
            history=history,
            trajectory=trajectory,
            perception=read(fs[last] / "vlm.json") if last >= 0 else None,
        )

    def asset(self, sid, step, name):
        if name not in (
            "view.jpg",
            "map.png",
            "step.json",
            "memory.json",
            "vlm.json",
            "sensors.npz",
            "grid.npz",
            "rgb.png",
        ):
            raise FileNotFoundError()
        p = self.folder(sid) / f"f{step:05}" / name
        if not p.is_file():
            raise FileNotFoundError()
        return p

    def archive(self, sid):
        folder = self.folder(sid)
        if not (folder / "config.json").is_file():
            raise FileNotFoundError()
        out = tempfile.SpooledTemporaryFile(max_size=4 * 1024 * 1024)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.write(folder / "config.json", "config.json")
            for f in frames(folder):
                for name in (
                    "step.json",
                    "rgb.png",
                    "map.png",
                    "sensors.npz",
                    "grid.npz",
                    "vlm.json",
                    "memory.json",
                    "error.json",
                ):
                    if (f / name).is_file():
                        z.write(f / name, f.name + "/" + name)
        out.seek(0)
        return out
