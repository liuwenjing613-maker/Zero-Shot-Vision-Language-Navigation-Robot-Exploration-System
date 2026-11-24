"""Detect and visit decision points in observed metric occupancy grids."""
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from .portal_memory import free_footprint


def skeletonize(mask):
    """Zhang-Suen parallel thinning, with a fixed background border."""
    image = np.pad(mask.astype(bool), 1)
    center = image[1:-1, 1:-1]
    while True:
        changed = False
        for phase in (0, 1):
            n = [
                image[:-2, 1:-1],
                image[:-2, 2:],
                image[1:-1, 2:],
                image[2:, 2:],
                image[2:, 1:-1],
                image[2:, :-2],
                image[1:-1, :-2],
                image[:-2, :-2],
            ]
            count = np.sum(n, axis=0)
            starts = np.sum([~a & b for a, b in zip(n, n[1:] + n[:1])], axis=0)
            a, b, c, d = n[::2]
            guard = (
                ~((a & b & c) | (b & c & d))
                if phase == 0
                else ~((a & b & d) | (a & c & d))
            )
            drop = center & (count >= 2) & (count <= 6) & (starts == 1) & guard
            if drop.any():
                center[drop] = False
                changed = True
        if not changed:
            return center.copy()


def skeleton_graph(mask, res):
    points = set(map(tuple, np.argwhere(mask)))
    graph = {}
    for r, c in points:
        nbr = []
        for dr, dc in (
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ):
            p = r + dr, c + dc
            if p not in points:
                continue
            if dr and dc and ((r + dr, c) in points or (r, c + dc) in points):
                continue
            nbr.append(p)
        graph[r, c] = set(nbr)
    # Remove terminal spurs before estimating junction degree.
    changed = True
    while changed:
        changed = False
        for leaf in [p for p, v in graph.items() if len(v) == 1]:
            if leaf not in graph or len(graph[leaf]) != 1:
                continue
            path = [leaf]
            dist = 0.0
            previous = None
            p = leaf
            while len(graph[p]) == (1 if previous is None else 2):
                nxt = next(q for q in graph[p] if q != previous)
                dist += math.dist(nxt, p) * res
                previous, p = p, nxt
                if dist >= 1.0:
                    break
                path.append(p)
            if dist < 1.0 and len(graph[p]) >= 3:
                for q in path[:-1]:
                    for v in list(graph[q]):
                        graph[v].discard(q)
                    del graph[q]
                changed = True
    return graph


@dataclass(frozen=True)
class Branch:
    direction_deg: float
    length_m: float
    min_width_m: float
    path: tuple


@dataclass(frozen=True)
class Junction:
    xy: tuple
    branches: tuple


def angle_gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def _trace_branches(graph, cluster, res):
    paths = []
    for source in sorted(cluster):
        for nxt in sorted(graph[source] - cluster):
            path = [source, nxt]
            seen = cluster | {nxt}
            length = res * math.dist(source, nxt)
            while len(graph[nxt]) == 2:
                following = graph[nxt] - seen
                if not following:
                    break
                q = next(iter(following))
                length += res * math.dist(q, nxt)
                nxt = q
                path.append(q)
                seen.add(q)
            if length + 1e-8 >= 1.0 and nxt not in cluster:
                paths.append((path, length))
    return paths


def _observation_cell(paths, root, res, clearance, robot_radius, margin):
    """Fit branch centerlines and select a locally safe observation cell."""
    normals = []
    intercepts = []
    for path, _ in paths:
        coords = (np.array(path) - root) * res
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        band = coords[(arc >= 0.55) & (arc <= 1.45)]
        if len(band) < 3:
            continue
        axis = np.linalg.svd(band - band.mean(0), full_matrices=False)[2][-1]
        normals.append(axis)
        intercepts.append(axis @ band.mean(0))
    fit = np.zeros(2)
    if len(normals) >= 3:
        offset, _, rank, _ = np.linalg.lstsq(normals, intercepts, rcond=None)
        if rank == 2 and np.linalg.norm(offset) <= 0.65:
            fit = offset
    r, c = root
    reach = math.ceil(0.6 / res)
    r0, c0 = max(0, r - reach), max(0, c - reach)
    patch = clearance[r0 : r + reach + 1, c0 : c + reach + 1]
    ys, xs = np.indices(patch.shape)
    ys = (ys + r0 - r) * res
    xs = (xs + c0 - c) * res
    error = np.hypot(ys - fit[0], xs - fit[1])
    ok = (error <= 0.28) & (np.hypot(ys, xs) <= 0.55) & (patch >= robot_radius + margin)
    if ok.any():
        ij = np.unravel_index(
            np.argmax(np.where(ok, patch - 0.70 * error, -np.inf)), patch.shape
        )
        r, c = ij[0] + r0, ij[1] + c0
    return r, c


def _independent_branches(paths, r, c, res, clearance, safe, xy):
    inner = max(0.72, min(1.10, clearance[r, c] + 0.10))
    outer = inner + 0.85
    reach = math.ceil(outer / res)
    r0, c0 = max(0, r - reach), max(0, c - reach)
    local = safe[r0 : r + reach + 1, c0 : c + reach + 1]
    ys, xs = np.ogrid[: local.shape[0], : local.shape[1]]
    radial = np.hypot(ys + r0 - r, xs + c0 - c) * res
    components, _ = ndi.label(local & (radial >= inner) & (radial <= outer))
    # Distinct open angular sectors provide a second independence check in
    # broad junctions where two passages share the same inner annulus.
    angles = np.radians(np.arange(0, 360, 4))
    radii = np.arange(0.16, 1.60 + res / 2, res)
    rr = r + np.rint(np.sin(angles)[:, None] * radii / res).astype(int)
    cc = c + np.rint(np.cos(angles)[:, None] * radii / res).astype(int)
    valid = (rr >= 0) & (rr < safe.shape[0]) & (cc >= 0) & (cc < safe.shape[1])
    rays = np.zeros(rr.shape, bool)
    rays[valid] = safe[rr[valid], cc[valid]]
    bits = rays.all(1)
    sectors = []
    for index in np.flatnonzero(bits & ~np.roll(bits, 1)):
        ids = []
        i = index
        while bits[i]:
            ids.append(i)
            i = (i + 1) % len(bits)
        if 12 <= 4 * len(ids) <= 100:
            sectors.append(ids)
    exits = {}
    for path, length in paths:
        probe = next((p for p in path if math.dist((r, c), p) * res >= 1.22), path[-1])
        heading = math.degrees(math.atan2(probe[0] - r, probe[1] - c)) % 360
        groups = [
            (min(angle_gap(heading, 4 * i) for i in ids), n)
            for n, ids in enumerate(sectors)
        ]
        delta, sector = min(groups, default=(360, None))
        if len(sectors) >= 3 and delta <= 20:
            label = ("sector", sector)
        else:
            probe = next(
                (p for p in path if math.dist((r, c), p) * res >= inner + res), None
            )
            if probe is None or math.dist((r, c), probe) * res > outer:
                continue
            label = components[probe[0] - r0, probe[1] - c0]
            if not label:
                continue
            label = ("annulus", int(label))
        branch = Branch(
            heading,
            length,
            2 * min(clearance[p] for p in path),
            tuple(map(xy, path)),
        )
        if label not in exits or length > exits[label].length_m:
            exits[label] = branch
    distinct = []
    tangents = []
    for branch in sorted(exits.values(), key=lambda b: (-b.length_m, b.direction_deg)):
        # A broad fork can produce two divergent pixels that soon run parallel.
        points = np.asarray(branch.path)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        band = points[(arc >= 0.65) & (arc <= 2.0)]
        tangent = branch.direction_deg
        if len(band) >= 2:
            dx, dz = band[-1] - band[0]
            if math.hypot(dx, dz) > 0.20:
                tangent = math.degrees(math.atan2(dz, dx)) % 360
        if all(
            angle_gap(branch.direction_deg, b.direction_deg) >= 40 for b in distinct
        ) and all(angle_gap(tangent, a) >= 40 for a in tangents):
            distinct.append(branch)
            tangents.append(tangent)
    return distinct


def junctions(grid, robot_radius=0.1, margin=0.06):
    if not np.isfinite([robot_radius, margin]).all() or robot_radius <= 0 or margin < 0:
        raise ValueError("Invalid robot clearance")
    res = grid.resolution
    free = grid.grid() == 0
    distance = ndi.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * res
    clearance = np.maximum(0, distance - res / math.sqrt(2))
    safe = free & (clearance >= robot_radius + margin)
    skeleton = skeletonize(safe)
    graph = skeleton_graph(skeleton, res)
    keys = {p for p, v in graph.items() if len(v) >= 3}
    result = []

    def xy(cell):
        return (
            grid.origin_x + (cell[1] + 0.5) * res,
            grid.origin_y + (cell[0] + 0.5) * res,
        )

    while keys:
        first = min(keys)
        keys.remove(first)
        cluster = {first}
        stack = [first]
        while stack:
            for v in graph[stack.pop()] & keys:
                cluster.add(v)
                keys.remove(v)
                stack.append(v)
        mid = np.mean(list(cluster), axis=0)
        root = min(cluster, key=lambda p: (math.dist(p, mid), p))
        paths = _trace_branches(graph, cluster, res)
        if len(paths) < 3:
            continue
        r, c = _observation_cell(paths, root, res, clearance, robot_radius, margin)
        distinct = _independent_branches(paths, r, c, res, clearance, safe, xy)
        if len(distinct) >= 3:
            result.append(
                Junction(
                    xy((r, c)), tuple(sorted(distinct, key=lambda b: b.direction_deg))
                )
            )
    return result, skeleton


class JunctionDetector:
    """One instance per scene/floor, sharing the topology used by other triggers."""

    def __init__(self, graph, robot_radius=0.1, margin=0.06):
        self.graph = graph
        self.robot_radius = robot_radius
        self.margin = margin
        self.candidates = []
        self.skeleton = None
        self._version = None
        self._grid = None
        self._pending = None
        self._latched = None
        self._visits = []
        self._previous = None
        self._time = -math.inf
        self._observations = set()

    def update(self, grid, xy, timestamp, observation_id, *, area_id=None):
        xy = tuple(map(float, xy))
        if (
            len(xy) != 2
            or not np.isfinite([*xy, timestamp]).all()
            or timestamp <= self._time
        ):
            raise ValueError("Expected ordered finite base observations")
        new_scan = grid is not self._grid or grid.map_version != self._version
        if new_scan:
            self.candidates, self.skeleton = junctions(
                grid, self.robot_radius, self.margin
            )
            self._grid, self._version = grid, grid.map_version
        stable = self._previous is not None and math.dist(self._previous, xy) <= 0.12
        fresh = (
            isinstance(observation_id, str)
            and bool(observation_id)
            and observation_id not in self._observations
            and new_scan
        )
        self._time, self._previous = timestamp, xy
        if fresh:
            self._observations.add(observation_id)
        if self._latched:
            old, old_area = self._latched
            if area_id == old_area and math.dist(old, xy) < 0.85:
                return None
            self._latched = None

        def accessible(target):
            samples = np.linspace(
                xy,
                target,
                max(2, math.ceil(math.dist(xy, target) / (grid.resolution / 2)) + 1),
            )
            return all(
                free_footprint(grid, p, self.robot_radius + self.margin)
                for p in samples
            )

        nearby = []
        for j in self.candidates:
            if math.dist(j.xy, xy) >= 0.5:
                continue
            # Verify the robot and junction share the same safe footprint corridor.
            if accessible(j.xy):
                nearby.append(j)
        if not nearby or not observation_id:
            self._pending = None
            return None
        j = min(nearby, key=lambda j: math.dist(j.xy, xy))
        old = self._pending
        if not fresh:
            return None
        self._pending = (j, area_id, xy)
        if (
            not old
            or old[1] != area_id
            or not stable
            or math.dist(old[2], xy) > 0.12
            or math.dist(old[0].xy, j.xy) > 0.30
        ):
            return None
        agrees = np.array(
            [
                [angle_gap(b.direction_deg, c.direction_deg) < 25 for c in j.branches]
                for b in old[0].branches
            ]
        )
        rows, cols = linear_sum_assignment(~agrees)
        if agrees[rows, cols].sum() < 3:
            return None
        allowed = tuple(
            p.place_id
            for p in self.graph.places.values()
            if p.area_id == area_id
            and math.dist((p.x, p.y), xy) < 1.6
            and accessible((p.x, p.y))
        )
        matches = [
            (math.dist(center, j.xy), place)
            for center, area, place in self._visits
            if area == area_id and place in allowed and math.dist(center, j.xy) < 0.55
        ]
        reuse = min(matches)[1] if matches else None
        place = self.graph.visit(
            "junction:" + observation_id,
            xy,
            timestamp,
            "junction",
            [observation_id],
            area_id=area_id,
            reuse_place=reuse,
            eligible_place_ids=allowed,
            branches=tuple(
                (b.direction_deg, b.length_m, b.min_width_m) for b in j.branches
            ),
        )
        if not matches:
            self._visits.append((j.xy, area_id, place))
        self._latched = (j.xy, area_id)
        self._pending = None
        return place
