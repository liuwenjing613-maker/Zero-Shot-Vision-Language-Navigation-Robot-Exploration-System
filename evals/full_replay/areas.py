from pathlib import Path
import json, sys
import numpy as np
import habitat_sim as hs
from options import options, REPO

args = options()
R = args.runs_root
sys.path[:0] = [str(REPO / "src"), str(REPO / "code")]
from vln_memory.online_memory import new_grid, fuse
from vln_memory.exploration_adapter import bounded_cells
from vln_memory.frontier_memory import FrontierMemory
from exploration.explorer import FrontierExplorer
from exploration.viewpoint import ViewpointConfig

read = lambda p: json.loads(p.read_text())
pf = hs.PathFinder()
assert pf.load_nav_mesh(str(args.dataset / "2azQ1b91cZZ/2azQ1b91cZZ.navmesh"))
lo, hi = [np.asarray(x)[[0, 2]] for x in pf.get_bounds()]
origin = np.floor(lo - 7)
res = 0.08
cfg = dict(
    shape=np.ceil((hi + 7 - origin)[::-1] / res).astype(int),
    origin=origin,
    resolution=res,
)
explorer = FrontierExplorer(
    ViewpointConfig(sample_radius=1, sensor_range=2, num_rays=16, max_candidates=60)
)


def runs(cells, offset):
    arr = sorted((int(r) - offset[0], int(c) - offset[1]) for r, c in cells)
    out = []
    for r, c in arr:
        if out and r == out[-1][0] and c == out[-1][1] + out[-1][2]:
            out[-1][2] += 1
        else:
            out.append([r, c, 1])
    return out


data = {}
cache = {}
outstats = []
for ep in read(Path("dataset.json")):
    grid = new_grid(cfg)
    name = ep["id"]
    offset = np.rint((ep["extent"][:2] - origin)[::-1] / res).astype(int).tolist()
    fms = {}
    initial = {}
    frs = []
    for fr in ep["frames"]:
        r = fr["row"]
        folder = R / name / r["observation"]
        with np.load(folder / "sensors.npz") as raw:
            xy = [r["pose"][0], r["pose"][2]]
            fuse(grid, dict(step=r["frame"], xy=xy), raw["ranges"])
        a = grid.grid()
        cell = grid.cell(xy)
        key = (a.tobytes(), cell)
        if key not in cache:
            cache[key] = explorer.detect(
                a, res, tuple(origin), cell, r["map_version"], floor_y=r["pose"][1]
            )
        snap = cache[key]
        m = read(folder / "memory.json")["exploration"]
        rec = m["frontiers"]
        mapping = {}
        details = {}
        for c in snap["candidates"]:
            matches = [
                p
                for p in fr["frontiers"]
                if np.allclose(c["centroid_xy"], p["center"], atol=1e-7, rtol=0)
                and np.allclose(c["safe_viewpoint_xy"], p["xy"], atol=1e-7, rtol=0)
            ]
            assert len(matches) == 1
            id = matches[0]["id"]
            mapping[c["local_candidate_id"]] = id
            cells = c["cluster_cells"]
            rid = "R" + id[1:]
            if rid not in initial:
                region = bounded_cells(grid, cells)
                initial[rid] = runs(region, offset)
                fms[rid] = FrontierMemory()
                fms[rid].region(
                    rid, [id], region, {"V1": grid.cell(c["safe_viewpoint_xy"])}
                )
            details[id] = dict(
                cells=runs(cells, offset),
                count=len(cells),
                gain=c["gain_cells"],
                path=c["path_length"],
                clearance=c["clearance"],
                rid=rid,
            )
        assert len(details) == len(fr["frontiers"])
        ranking = explorer.rank(snap, rec, mapping)
        for j, c in enumerate(ranking):
            details[c.frontier_id].update(rank=j + 1, score=c.local_utility)
        for rid, region in m["regions"].items():
            fm = fms[rid]
            fm.records = rec
            fm.evaluate(rid, grid, True)
            out = fm.regions[rid]
            assert [out["observed"], out["eligible"]] == [
                region["observed"],
                region["eligible"],
            ], (name, r["frame"], rid)
        frs.append(details)
    data[name] = dict(frames=frs, regions=initial)
    outstats.append([name, sum(len(x) for x in frs), len(initial)])
    print(outstats[-1], flush=True)
Path("areas.json").write_text(json.dumps(data, separators=(",", ":")))
