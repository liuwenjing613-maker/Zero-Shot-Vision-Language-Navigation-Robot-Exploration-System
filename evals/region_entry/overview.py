"""Align independent textured MP3D references to the 9 m scan windows."""
import json
from pathlib import Path
import sys
import numpy as np
from fixture import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m1_lab"))
from render_topdown import render


def build(folder, out, indices):
    fixtures = json.loads((folder / "fixtures.json").read_text())
    for i in indices:
        pos = fixtures[i]["start"]
        xy = np.array(pos)[[0, 2]]
        frame = dict(
            origin=(xy - 4.5).tolist(), shape=[90, 90], resolution=0.1, xy=xy.tolist()
        )
        ref = out / f"reference-{i}"
        ref.mkdir(parents=True, exist_ok=True)
        path = ref / "window.json"
        path.write_text(json.dumps(frame))
        render(ROOT, path, ref, 3, floor_y=pos[1])


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]), list(map(int, sys.argv[3:])))
