import base64, io, json, sys, math
from pathlib import Path
import numpy as np
import habitat_sim as hs
from PIL import Image
from options import options, cases, SOURCE, REPO

args = options()
HERE = args.output
HERE.mkdir(parents=True, exist_ok=True)
RUN = args.runs_root
sys.path.insert(0, str(REPO / "src"))
from vln_memory.online_memory import new_grid, fuse


def read(path):
    return json.loads(path.read_text())


def image_data(image, fmt="WEBP", **kw):
    buf = io.BytesIO()
    image.save(buf, fmt, **kw)
    return (
        "data:image/"
        + fmt.lower()
        + ";base64,"
        + base64.b64encode(buf.getvalue()).decode()
    )


pf = hs.PathFinder()
assert pf.load_nav_mesh(str(args.dataset / "2azQ1b91cZZ/2azQ1b91cZZ.navmesh"))
lo, hi = [np.array(x)[[0, 2]] for x in pf.get_bounds()]
origin = np.floor(lo - 7)
res = 0.08
cfg = dict(
    shape=np.ceil((hi + 7 - origin)[::-1] / res).astype(int),
    origin=origin,
    resolution=res,
)
payload = []
for case in cases(args):
    ident = case["id"]
    label = case.get("label", ident)
    goal = case.get("goal") or read(RUN / ident / "goal.json")["instruction"]
    setting = case.get("setting", "Habitat 实测")
    auditfile = case.get("audit")
    path = RUN / ident
    trace = read(path / "trace.json")
    checks = read(RUN / auditfile) if auditfile else []
    ledger = [json.loads(x) for x in (path / "events.jsonl").read_text().splitlines()]
    grid = new_grid(cfg)
    grids = []
    frames = []
    for row in trace["actions"]:
        obs = row["observation"]
        d = path / obs
        m = read(d / "memory.json")
        p = read(d / "perception.json")
        with np.load(d / "sensors.npz") as raw:
            scan = raw["ranges"]
            dep = raw["depth"]
            tf = raw["camera_pose"]
            fuse(
                grid, dict(xy=[row["pose"][0], row["pose"][2]], step=row["frame"]), scan
            )
            assert grid.map_version == m["map_version"] == row["map_version"]
            arr = grid.grid().copy()
            grids.append(arr)
            pal = np.array(
                [
                    [21, 23, 67],
                    [48, 55, 124],
                    [39, 113, 151],
                    [48, 171, 151],
                    [146, 207, 111],
                    [251, 232, 136],
                ],
                float,
            )
            clean = np.nan_to_num(dep, nan=0, posinf=0)
            idx = np.clip(clean / 6, 0, 1) * (len(pal) - 1)
            low = idx.astype(int)
            frac = (idx - low)[..., None]
            color = (
                pal[low] * (1 - frac) + pal[np.minimum(low + 1, len(pal) - 1)] * frac
            )
            color[(~np.isfinite(dep)) | (dep <= 0)] = [15, 23, 31]
            depth = image_data(Image.fromarray(color.astype("uint8")), quality=85)
            hd = [float(-tf[0, 2]), float(-tf[2, 2])]
        local = {v["local_id"]: v for v in m["object_memory"]["observations"].values()}
        boxes = []
        for typ, items in [
            ("goal", p["target_entities"]),
            ("context", p["context_entities"]),
            ("cue", p["visual_cues"]),
        ]:
            for item in items:
                entry = local.get(item["local_id"], {})
                boxes.append(
                    dict(
                        **item,
                        kind=typ,
                        globalId=entry.get("object_id"),
                        association=entry.get("association_status"),
                        world=entry.get("map_position_xz"),
                    )
                )
        frontiers = [
            dict(
                id=k,
                xy=v["safe_viewpoint_xy"],
                center=v["centroid_xy"],
                status=v["status"],
                visits=v["visit_count"],
            )
            for k, v in m["exploration"]["frontiers"].items()
            if v["last_seen"] == grid.map_version
        ]
        assert len(frontiers) == row["frontiers"], (
            ident,
            obs,
            len(frontiers),
            row["frontiers"],
        )
        objects = []
        for k, v in m["object_memory"]["objects"].items():
            scores = v["category_scores"]
            objects.append(
                dict(
                    id=k,
                    xy=v["map_position_xz"],
                    cat=max(scores, key=scores.get) if scores else "?",
                    score=max(scores.values(), default=0),
                    sigma=v.get("position_sigma_m"),
                    last=v["last_seen"],
                    status=v["track_status"],
                )
            )
        frames.append(
            dict(
                row=row,
                rgb=image_data(Image.open(d / "rgb.png").convert("RGB"), quality=92),
                depth=depth,
                heading=hd,
                boxes=boxes,
                objects=objects,
                frontiers=frontiers,
                place=m["place"],
                area=m["area"],
                places=m["places"],
                edges=m["edges"],
                regions=m["exploration"]["regions"],
                status=p["target_status"],
                latency=p["latency_ms"],
                model=p["model_id"],
                known=int((arr != -1).sum()),
                occupied=int((arr == 100).sum()),
                free=int((arr == 0).sum()),
                audit=[a for a in checks if a["obs"] == obs],
                events=[
                    e
                    for e in ledger
                    if e.get("observation_id") == obs
                    or (
                        not e.get("observation_id")
                        and e.get("timestamp") == row["frame"]
                    )
                ],
            )
        )
    cells = np.argwhere(np.logical_or.reduce([g != -1 for g in grids]))
    r0, c0 = np.maximum(cells.min(0) - 10, 0)
    r1, c1 = np.minimum(cells.max(0) + 11, grids[0].shape)
    for f, g in zip(frames, grids):
        sub = g[r0:r1, c0:c1]
        img = np.zeros((*sub.shape, 3), np.uint8)
        img[:] = [25, 42, 52]
        img[sub == 0] = [187, 211, 211]
        img[sub == 100] = [64, 114, 117]
        f["map"] = image_data(Image.fromarray(img), "PNG", optimize=True)
    epi = dict(
        id=ident,
        label=label,
        goal=goal,
        setting=setting,
        instruction=read(path / "goal.json"),
        terminal=trace["terminal"],
        scene=trace["scene"],
        frames=frames,
        align_with_travel=trace.get("align_with_travel", False),
        yaw_offset=trace.get("yaw_offset", case.get("yaw_offset", 0)),
        planner_at_arrival=trace.get(
            "planner_at_arrival", case.get("planner_at_arrival", False)
        ),
        extent=[
            float(origin[0] + c0 * res),
            float(origin[1] + r0 * res),
            int(c1 - c0),
            int(r1 - r0),
        ],
        res=res,
    )
    payload.append(epi)
    print(ident, len(frames), flush=True)
# Recreate source data and publish only after both geometry and execution checks pass.
import subprocess
from PIL import ImageChops

(HERE / "dataset.json").write_text(
    json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
)
bounds = {}
for ep in payload:
    bounds[ep["id"]] = []
    for f in ep["frames"]:
        img = Image.open(io.BytesIO(base64.b64decode(f["map"].split(",")[1]))).convert(
            "RGB"
        )
        bounds[ep["id"]].append(
            ImageChops.difference(
                img, Image.new("RGB", img.size, (25, 42, 52))
            ).getbbox()
        )
(HERE / "bounds.json").write_text(json.dumps(bounds))
if not payload:
    raise ValueError("No episode traces found")
for script in ("areas.py", "motion.py", "package.py"):
    subprocess.run(
        [sys.executable, str(SOURCE / script), *sys.argv[1:]], cwd=HERE, check=True
    )
