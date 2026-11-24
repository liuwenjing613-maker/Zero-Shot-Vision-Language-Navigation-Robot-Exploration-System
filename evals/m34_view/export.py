"""Export the real M1-M4 replay with the already published M2 RGB/grid frames."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from vln_memory.online_memory import OnlineMemory


def export(folder, out):
    cfg = json.loads((folder / "config.json").read_text())
    memory = OnlineMemory(dict(cfg, episode_path=out / "events.jsonl"))
    rows = []
    for frame in sorted(folder.glob("f[0-9]*")):
        step = json.loads((frame / "step.json").read_text())
        vlm = json.loads((frame / "vlm.json").read_text())
        assert (
            hashlib.sha256((frame / "rgb.png").read_bytes()).hexdigest()
            == vlm["rgb_sha256"]
            == step["rgb_sha256"]
        )
        with np.load(frame / "sensors.npz") as data:
            result = memory.apply(step, data, vlm["result"])
        for key, region in memory.exploration.m3.regions.items():
            target = result["exploration"]["regions"][key]
            target["mask"] = [
                [int(r), int(c), int(memory.grid.observed_count[r, c] > 0)]
                for r, c in sorted(region["eligible_cells"])
            ]
        rows.append(
            dict(
                step=step["step"],
                window=step["window"],
                xy=step["xy"],
                **result["exploration"],
            )
        )
    (out / "frames.json").write_text(
        json.dumps(
            dict(
                scene=cfg["scene"],
                origin=cfg["origin"],
                resolution=cfg["resolution"],
                frames=rows,
            ),
            allow_nan=False,
        )
    )
    print(
        len(rows),
        len(rows[-1]["frontiers"]),
        len(memory.exploration.m4.events),
        flush=True,
    )


if __name__ == "__main__":
    export(*map(Path, sys.argv[1:]))
