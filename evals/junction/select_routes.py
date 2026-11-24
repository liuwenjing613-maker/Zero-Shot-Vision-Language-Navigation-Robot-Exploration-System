"""Independent MP3D navmesh fixture oracle: straight reachable sectors at two radii."""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "region_entry"))
from fixture import regions, simulator


def gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def openings(bits):
    if bits.all() or not bits.any():
        return []
    starts = np.flatnonzero(bits & ~np.roll(bits, 1))
    groups = []
    for i in starts:
        n = 1
        while bits[(i + n) % len(bits)]:
            n += 1
        if n * 5 >= 15:
            groups.append(((i + (n - 1) / 2) * 5 % 360, n * 5))
    return sorted(groups)


def select(sim):
    pf = sim.pathfinder
    res = 0.08
    floor = 0.12711
    polys = [p for ((_, y), p) in regions().values() if abs(y - floor) < 0.35]
    lo = np.floor(np.concatenate(polys).min(0) - 1)
    hi = np.ceil(np.concatenate(polys).max(0) + 1)
    rows, cols = np.ceil((hi - lo)[::-1] / res).astype(int)
    valid = np.zeros((rows, cols), bool)
    for r, c in np.ndindex(valid.shape):
        x, z = lo + (np.array([c, r]) + 0.5) * res
        pos = pf.snap_point(np.array([x, floor, z]))
        valid[r, c] = (
            np.isfinite(pos).all()
            and abs(pos[1] - floor) < 0.22
            and np.hypot(pos[0] - x, pos[2] - z) < 0.035
        )
    dt = ndi.distance_transform_edt(valid) * res
    angles = np.radians(np.arange(0, 360, 5))
    direction = np.stack((np.sin(angles), np.cos(angles)), axis=1)
    offsets = np.rint(
        direction[:, None, :] * np.arange(0.20, 1.90, 0.08)[None, :, None] / res
    ).astype(int)
    pool = {k: [] for k in ("T", "Y", "cross", "straight", "bend")}
    for r in range(26, rows - 26, 3):
        for c in range(26, cols - 26, 3):
            radius = dt[r, c]
            if not 0.25 < radius < 1.10:
                continue
            cells = offsets + [r, c]
            rays = valid[cells[:, :, 0], cells[:, :, 1]]
            inner = openings(rays[:, :17].all(1))
            outer = openings(rays.all(1))
            if len(inner) != len(outer) or not 2 <= len(outer) <= 4:
                continue
            headings = np.array([v for v, w in outer])
            widths = np.array([w for v, w in outer])
            if max(widths) > 86 or min(widths) < 18:
                continue
            if any(min(gap(a, b) for b, w in inner) > 15 for a in headings):
                continue
            diffs = np.sort(np.diff(np.r_[headings, headings[0] + 360]))
            targets = {
                2: [("straight", [180, 180]), ("bend", [90, 270])],
                3: [("T", [90, 90, 180]), ("Y", [120] * 3)],
                4: [("cross", [90] * 4)],
            }
            for kind, ideal in targets[len(outer)]:
                error = max(abs(diffs - ideal))
                if error > 20:
                    continue
                center = lo + (np.array([c, r]) + 0.5) * res
                # Prefer well-centered, geometrically balanced choices. No detector output is used.
                score = (
                    error / 15
                    + np.std(widths) / 28
                    + abs(radius - 0.53) * 2
                    - min(widths) / 90
                )
                pool[kind].append((score, center, headings.tolist(), widths.tolist()))
    result = {}
    for kind, options in pool.items():
        options.sort(key=lambda x: x[0])
        for score, center, headings, widths in options:
            if any(np.linalg.norm(center - v["center"]) < 2 for v in result.values()):
                continue
            for j in np.argsort(widths)[::-1]:
                a = np.radians(headings[j])
                u = np.array([np.cos(a), np.sin(a)])
                start = center + u * 1.5
                p = np.asarray(pf.snap_point([start[0], floor, start[1]]))
                if not np.isfinite(p).all() or np.linalg.norm(p[[0, 2]] - start) > 0.04:
                    continue
                ok = True
                for _ in range(6):
                    aim = p - np.array([u[0], 0, u[1]]) * 0.25
                    p = np.asarray(pf.try_step(p, aim))
                    if np.linalg.norm(p[[0, 2]] - aim[[0, 2]]) > 0.025:
                        ok = False
                        break
                if ok:
                    result[kind] = dict(
                        center=center.tolist(),
                        headings=headings,
                        sector_widths=widths,
                        arrival_axis=u.tolist(),
                        score=score,
                        floor=floor,
                    )
                    break
            if kind in result:
                break
    # Select an ordinary two-exit portal before running the detector. Doors
    # opening directly onto another branch cannot serve as negative controls.
    previous = Path(__file__).resolve().parents[2] / "results/region-entry"
    for index in (1, 6, 2, 15, 9):
        f = json.loads((previous / f"{index}.json").read_text())["fixture"]
        center = np.array(f["center"])
        c, r = np.floor((center - lo) / res).astype(int)
        cells = offsets + [r, c]
        rays = valid[cells[:, :, 0], cells[:, :, 1]]
        exits = openings(rays.all(1))
        if (
            len(exits) != 2
            or len(openings(rays[:, :17].all(1))) != 2
            or gap(exits[0][0], exits[1][0]) < 145
        ):
            continue
        result["door"] = dict(
            center=center.tolist(),
            headings=[v for v, w in exits],
            sector_widths=[w for v, w in exits],
            arrival_axis=(-np.array(f["normal"])).tolist(),
            floor=floor,
            regions=[f["from"], f["to"]],
        )
        break
    if "door" not in result:
        raise AssertionError("No verified two-exit door fixture")
    return result


if __name__ == "__main__":
    with simulator(3) as sim:
        routes = select(sim)
    Path(sys.argv[1]).write_text(json.dumps(routes, indent=2))
    print([(k, v["center"], v["headings"]) for k, v in routes.items()])
