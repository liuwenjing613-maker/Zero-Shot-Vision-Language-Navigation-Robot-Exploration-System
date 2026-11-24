"""Write per-event snapshots directly from production M1 state."""
import base64
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def publish(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf8",
    )
    tmp.replace(path)


def packed(array, dtype):
    return base64.b64encode(array.astype(dtype).tobytes()).decode("ascii")


class Recorder:
    def __init__(self, folder, mode, grid, graph):
        self.folder, self.mode, self.grid, self.graph = Path(folder), mode, grid, graph
        self.folder.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.previous = grid.grid()
        self.previous_visits = 0

    def step(self, title, detail, checks, xy, **extra):
        g, t = self.grid, self.graph
        grid = g.grid()
        counts = dict(
            free=int((grid == 0).sum()),
            occupied=int((grid == 100).sum()),
            unknown=int((grid == -1).sum()),
            visits=int(g.visit_count.sum()),
        )
        i = len(self.steps)
        data = dict(
            title=title,
            detail=detail,
            checks=[dict(label=k, passed=bool(v)) for k, v in checks],
            version=g.map_version,
            shape=grid.shape,
            resolution=g.resolution,
            origin=[g.origin_x, g.origin_y],
            xy=list(map(float, xy)),
            counts=counts,
            changed=int((self.previous != grid).sum()),
            entered=counts["visits"] - self.previous_visits,
            cells=packed(grid, "i1"),
            odds=packed(np.round(g.log_odds * 100), "<i2"),
            visits=[
                [int(r), int(c), int(g.visit_count[r, c])]
                for r, c in zip(*np.nonzero(g.visit_count))
            ],
            places=[asdict(p) for p in t.places.values()],
            edges=[asdict(e) for e in t.edges.values()],
            frontiers=t.frontier_places,
            **extra,
        )
        publish(self.folder / f"{i}.json", data)
        self.steps.append(
            dict(title=title, passed=all(c["passed"] for c in data["checks"]))
        )
        publish(self.folder / "steps.json", self.steps)
        self.previous, self.previous_visits = grid.copy(), counts["visits"]
        failed = [c["label"] for c in data["checks"] if not c["passed"]]
        if failed:
            raise AssertionError(f"{self.mode} / {title} / {failed}")
