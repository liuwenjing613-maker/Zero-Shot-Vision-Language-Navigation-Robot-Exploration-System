"""Scene-local portal proposals and observed region transitions on one floor."""
from dataclasses import dataclass, field
import math
import numpy as np
from .portal_throat import passage_throat


def footprint_state(grid, xy, radius):
    row, col = grid.cell(xy)
    reach = int(math.ceil(radius / grid.resolution))
    unknown = False
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if math.hypot(dr, dc) * grid.resolution > radius + grid.resolution * 0.5:
                continue
            cell = (row + dr, col + dc)
            if grid.contains(cell) and grid.log_odds[cell] >= 0.8:
                return "BLOCKED"
            if not grid.contains(cell) or grid.log_odds[cell] > -0.4:
                unknown = True
    return "PROPOSED" if unknown else "VALIDATED"


def free_footprint(grid, xy, radius):
    return footprint_state(grid, xy, radius) == "VALIDATED"


@dataclass
class Portal:
    portal_id: str
    a: np.ndarray
    b: np.ndarray
    observations: list = field(default_factory=list)
    areas: dict = field(default_factory=dict)
    places: dict = field(default_factory=dict)
    crossings: int = 0
    status: str = "VALIDATED"
    armed: int = 0
    pending: int = 0
    stable: int = 0

    @property
    def center(self):
        return (self.a + self.b) / 2

    @property
    def width(self):
        return float(np.linalg.norm(self.b - self.a))

    @property
    def tangent(self):
        return (self.b - self.a) / self.width

    @property
    def normal(self):
        t = self.tangent
        return np.array([-t[1], t[0]])


def jambs(cue, depth, K, transform, grid, clearance):
    """Find near-to-far and far-to-near depth jumps supported by occupied jambs."""
    if cue.get("cue_type") not in ("doorway", "opening"):
        return None
    x0, y0, x1, y1 = (
        np.asarray(cue["bbox"], dtype=float)
        / 1000
        * [K.width, K.height, K.width, K.height]
    )
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    candidates = []
    # Use actual pixels on several horizontal slices, rather than the background
    # depth in the middle of the opening. Vote across rows for consistent jambs.
    rows = np.unique(
        np.clip(
            np.rint(np.linspace(y0 + 0.35 * (y1 - y0), y0 + 0.8 * (y1 - y0), 11)),
            0,
            K.height - 1,
        ).astype(int)
    )
    for v in rows:
        z = depth[v]
        ok = np.isfinite(z) & (z > 0.10) & (z < 7.9)
        jumps = np.diff(z)

        def edges(sign, expected):
            idx = np.flatnonzero(ok[:-1] & ok[1:] & (sign * jumps > 0.4)) + (
                1 if sign < 0 else 0
            )
            limit = max(8, 0.35 * (x1 - x0))
            return sorted(
                (int(j) for j in idx if abs(j - expected) <= limit),
                key=lambda j: abs(j - expected),
            )[:3]

        def point(u):
            cam = np.array([(u - K.cx) * z[u] / K.fx, (K.cy - v) * z[u] / K.fy, -z[u]])
            return (transform[:3, :3] @ cam + transform[:3, 3])[[0, 2]]

        for left in edges(1, x0):
            for right in edges(-1, x1):
                if right <= left + 4:
                    continue
                a, b = point(left), point(right)
                width = np.linalg.norm(a - b)
                if not 2 * clearance + 0.10 <= width <= 2.8:
                    continue
                # Similar jamb depths exclude many unrelated foreground/background pairs.
                if abs(z[left] - z[right]) > 0.7 * width:
                    continue
                candidates.append((v, a, b, abs(left - x0) + abs(right - x1)))
    best = None
    for _, a, b, _ in candidates:
        votes = {}
        for v, pa, pb, error in candidates:
            if max(np.linalg.norm(pa - a), np.linalg.norm(pb - b)) <= 0.25 and (
                v not in votes or error < votes[v][2]
            ):
                votes[v] = (pa, pb, error)
        score = (len(votes), -sum(x[2] for x in votes.values()) / max(1, len(votes)))
        if best is None or score > best[0]:
            best = (score, votes)
    if best is None or best[0][0] < 3:
        return None
    a, b = [np.median([p[k] for p in best[1].values()], axis=0) for k in (0, 1)]

    def occupied(pt):
        r, c = grid.cell(pt)
        return any(
            grid.contains((r + i, c + j)) and grid.log_odds[r + i, c + j] >= 0.8
            for i in range(-1, 2)
            for j in range(-1, 2)
        )

    if not all(occupied(pt) for pt in (a, b)):
        return None
    return a, b


class RegionEntryDetector:
    def __init__(self, graph, robot_radius=0.12, margin=0.06):
        self.graph = graph
        self.clearance = robot_radius + margin
        if robot_radius <= 0 or margin < 0:
            raise ValueError("Invalid robot clearance")
        self.portals = {}
        self.proposals = {}
        self.proposal_versions = {}
        self.area = "A01"
        self.area_count = 1
        self.place = None
        self.previous = None
        self.timestamp = None
        self.path_length = 0.0
        self.continuous = True
        self.used_observations = set()

    def start(self, xy, timestamp, observation):
        if self.place is not None:
            raise ValueError("Already initialized")
        self.place = self.graph.visit(
            "region-start", xy, timestamp, "start", [observation], area_id=self.area
        )
        self.previous = np.array(xy, dtype=float)
        self.timestamp = timestamp
        self.used_observations.add(observation)

    def perceive(
        self, perception, depth, intrinsics, world_from_camera, grid, *, base_xy=None
    ):
        obs = perception["observation_id"]
        depth = np.asarray(depth)
        if depth.shape != (intrinsics.height, intrinsics.width):
            raise ValueError("Depth calibration mismatch")
        reports = []
        cues = list(perception["visual_cues"])
        # Two tall narrow detections can be the two posts of one visible door.
        posts = [
            c
            for c in cues
            if c["cue_type"] == "doorway"
            and c["bbox"][2] - c["bbox"][0] < 140
            and c["bbox"][3] - c["bbox"][1] > 550
        ]
        posts.sort(key=lambda c: c["bbox"][0])
        for left, right in zip(posts, posts[1:]):
            a, b = left["bbox"], right["bbox"]
            if 150 < b[0] - a[2] < 600 and min(a[3], b[3]) - max(a[1], b[1]) > 500:
                cues.append(
                    dict(
                        cue_type="opening",
                        local_id=left["local_id"] + "+" + right["local_id"],
                        bbox=[a[2], max(a[1], b[1]), b[0], min(a[3], b[3])],
                    )
                )
        for cue in cues:
            kind = cue["cue_type"]
            if kind not in ("doorway", "opening", "blocked_passage", "corridor"):
                continue
            key = (obs, cue["local_id"])
            if key in self.proposals and (
                self.proposal_versions[key] == grid.map_version
                or self.proposals[key]["status"] in ("VALIDATED", "CROSSED")
            ):
                reports.append(self.proposals[key])
                continue
            report = dict(
                observation_id=obs, cue_id=cue["local_id"], kind=kind, status="PROPOSED"
            )
            xy = world_from_camera[:3, 3][[0, 2]]
            pair = jambs(
                cue, depth, intrinsics, world_from_camera, grid, self.clearance
            )
            if pair is None and kind in ("corridor", "doorway", "opening"):
                pair = passage_throat(
                    cue, depth, intrinsics, world_from_camera, grid, self.clearance
                )
                if pair is not None:
                    report["geometry"] = "verified_narrowing"
            if kind == "corridor" and pair is None:
                continue
            if kind == "blocked_passage":
                report["status"] = "BLOCKED"
                x0, y0, x1, y1 = (
                    np.asarray(cue["bbox"])
                    * [
                        intrinsics.width,
                        intrinsics.height,
                        intrinsics.width,
                        intrinsics.height,
                    ]
                    / 1000.0
                )
                blocked = []
                for q in self.portals.values():
                    eye = world_from_camera[:3, 3]
                    local = world_from_camera[:3, :3].T @ (
                        np.array([q.center[0], eye[1], q.center[1]]) - eye
                    )
                    if local[2] >= -0.1:
                        continue
                    u = int(round(intrinsics.cx - intrinsics.fx * local[0] / local[2]))
                    v = int(round(intrinsics.cy + intrinsics.fy * local[1] / local[2]))
                    if not (
                        0 <= u < intrinsics.width
                        and 0 <= v < intrinsics.height
                        and x0 <= u < x1
                        and y0 <= v < y1
                    ):
                        continue
                    z = depth[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3]
                    if (
                        np.count_nonzero(
                            np.isfinite(z) & (z > 0) & (np.abs(z + local[2]) < 0.3)
                        )
                        > z.size // 2
                    ):
                        blocked.append(q)
                if len(blocked) == 1:
                    q = blocked[0]
                    q.status = "BLOCKED"
                    q.armed = q.pending = q.stable = 0
                    report["portal_id"] = q.portal_id
            elif pair is not None:
                p = Portal("", *pair)
                side = np.sign(np.dot(xy - p.center, p.normal)) or 1
                far = p.center - side * 0.55 * p.normal
                base = xy if base_xy is None else np.asarray(base_xy, dtype=float)
                if (base - p.center) @ p.normal * side < 0:
                    continue
                states = {
                    footprint_state(grid, s, self.clearance)
                    for a, b in ((base, p.center), (p.center, far))
                    for s in np.linspace(
                        a,
                        b,
                        max(
                            2,
                            math.ceil(np.linalg.norm(b - a) / (grid.resolution / 2))
                            + 1,
                        ),
                    )
                }
                status = (
                    "BLOCKED"
                    if "BLOCKED" in states
                    else ("PROPOSED" if "PROPOSED" in states else "VALIDATED")
                )
                matches = [
                    q
                    for q in self.portals.values()
                    if np.linalg.norm(q.center - p.center) < 0.5
                    and abs(np.dot(q.normal, p.normal)) > math.cos(math.radians(20))
                    and abs(q.width - p.width) < 0.4
                ]
                if len(matches) <= 1:
                    q = matches[0] if matches else p
                    if not matches:
                        q.portal_id = "D%02d" % (len(self.portals) + 1)
                        self.portals[q.portal_id] = q
                    if obs not in q.observations:
                        q.observations.append(obs)
                    q.status = status
                    if status != "VALIDATED":
                        q.armed = q.pending = q.stable = 0
                    report.update(status=q.status, portal_id=q.portal_id)
            self.proposals[key] = report
            self.proposal_versions[key] = grid.map_version
            reports.append(report)
        return reports

    def advance(self, xy, timestamp, grid, observation=None):
        """Feed successive executed base poses. Only a fresh observation can create a place."""
        xy = np.asarray(xy, dtype=float)
        if (
            self.previous is None
            or xy.shape != (2,)
            or not np.isfinite(xy).all()
            or not math.isfinite(timestamp)
            or timestamp <= self.timestamp
        ):
            raise ValueError("Expected ordered, finite base poses after start")
        previous = self.previous
        self.previous = xy.copy()
        self.timestamp = timestamp
        delta = float(np.linalg.norm(xy - previous))
        fresh = bool(observation) and observation not in self.used_observations
        if fresh:
            self.used_observations.add(observation)
        if delta > 0.55:
            for p in self.portals.values():
                p.armed = p.pending = p.stable = 0
            self.continuous = False
            return None
        self.path_length += delta
        for p in self.portals.values():
            if p.status not in ("VALIDATED", "CROSSED"):
                continue
            n = p.normal
            old = float((previous - p.center) @ n)
            new = float((xy - p.center) @ n)
            if not p.pending and abs(old) >= 0.15:
                p.armed = 1 if old > 0 else -1
            side = p.armed
            if not side:
                continue
            if not p.pending and side * old >= 0 and side * new < 0:
                hit = previous + (xy - previous) * old / (old - new)
                if abs((hit - p.center) @ p.tangent) <= p.width / 2 - self.clearance:
                    p.pending = -side
                else:
                    p.armed = 0
            if not p.pending:
                continue
            progress = p.pending * new
            if progress <= 0:
                p.pending = p.stable = 0
                continue
            if not free_footprint(grid, xy, self.clearance) or not free_footprint(
                grid, p.center, self.clearance
            ):
                p.armed = p.pending = p.stable = 0
                continue
            p.stable = p.stable + 1 if progress >= 0.30 else 0
            if p.stable < 2 or progress < 0.45 or not fresh:
                continue
            dest = p.pending
            if -dest in p.areas and p.areas[-dest] != self.area:
                p.armed = p.pending = p.stable = 0
                continue
            p.areas[-dest] = self.area
            if dest not in p.areas:
                self.area_count += 1
                p.areas[dest] = "A%02d" % self.area_count
            self.area = p.areas[dest]
            target = self.graph.visit(
                "region:" + observation,
                xy,
                timestamp,
                "new_region",
                [observation],
                area_id=self.area,
                portal_id=p.portal_id,
                reuse_place=p.places.get(dest),
            )
            p.places[dest] = target
            if self.continuous and self.place != target:
                edge = self.graph.record_navigation(
                    "cross:" + observation,
                    self.place,
                    target,
                    "succeeded",
                    self.path_length,
                )
                if p.portal_id not in edge.portal_ids:
                    edge.portal_ids.append(p.portal_id)
            self.place = target
            self.path_length = 0.0
            self.continuous = True
            p.crossings += 1
            p.status = "CROSSED"
            p.armed, p.pending, p.stable = dest, 0, 0
            return dict(
                portal_id=p.portal_id,
                area=self.area,
                place=target,
                xy=xy.tolist(),
                observation_id=observation,
            )
        return None
