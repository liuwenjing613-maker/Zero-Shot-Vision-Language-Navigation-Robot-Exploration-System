"""Export a frame-aligned, read-only audit of a recorded MP3D run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from vln_memory.online_memory import OnlineMemory


def export(session, out):
    config = json.loads((session / "config.json").read_text())
    mem = OnlineMemory(config)
    out.mkdir(exist_ok=True, parents=True)
    frames = []
    for path in sorted(session.glob("f[0-9]*")):
        step = json.loads((path / "step.json").read_text())
        vlm = json.loads((path / "vlm.json").read_text())
        rgb = path / "rgb.png"
        if hashlib.sha256(rgb.read_bytes()).hexdigest() != step["rgb_sha256"] or step["rgb_sha256"] != vlm["rgb_sha256"]:
            raise ValueError(f"RGB mismatch in {path}")
        with np.load(path / "sensors.npz") as sensors:
            record = mem.apply(step, sensors, vlm["result"])
        i = step["step"]
        for src, name in [(rgb, "rgb"), (path / "map.png", "grid")]:
            shutil.copyfile(src, out / f"{name}{i}.png")
        frames.append(dict(step=i, action=step["action"], xy=step["xy"], heading=step["heading"],
                           window=step["window"], memory=record["object_memory"], source=vlm["source"]))
    web = Path(__file__).resolve().parent
    live = web.parent / "live" / "web"
    ref = json.loads((live / "reference.json").read_text())[config["preset"]]
    shutil.copyfile(live / (config["preset"]+".jpg"), out / "reference.jpg")
    for name in ("index.html", "viewer.js", "style.css"):
        shutil.copyfile(web / name, out / name)
    data = dict(scene=config["scene"], model=config["model"], reference=ref, frames=frames)
    (out / "data.json").write_text(json.dumps(data, allow_nan=False))
    print(f"{len(frames)} frames, {len(mem.objects.observations)} observations, {len(mem.objects.records)} objects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.session, args.output)
