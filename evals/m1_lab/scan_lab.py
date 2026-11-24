"""Full-revolution fixtures and independent visible-surface reference maps."""
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from record import Recorder
from vln_memory.geometry_memory import OccupancyMemory
from vln_memory.topology_memory import TopologyMemory

N = 720
LIMIT = 6.0
STEP = 2 * math.pi / N
ANGLES = np.arange(N) * STEP


def directions(angles):
    return np.column_stack((np.cos(angles), np.sin(angles)))


def segment_ranges(xy, angles, seg, front_only=False):
    """First intersection of each 2D ray with the reference line segments."""
    a = seg[:, 0] - xy
    v = seg[:, 1] - seg[:, 0]
    num = a[:, 0] * v[:, 1] - a[:, 1] * v[:, 0]
    result = []
    for group in np.array_split(directions(angles), math.ceil(len(angles) / 48)):
        dx, dy = group[:, 0, None], group[:, 1, None]
        den = dx * v[:, 1] - dy * v[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
            u = (a[:, 0] * dy - a[:, 1] * dx) / den
        valid = (abs(den) > 1e-12) & (u >= 0) & (u <= 1) & (t > 0) & (t < LIMIT)
        if front_only:
            valid &= den < -1e-12
        result.extend(np.where(valid, t, np.inf).min(axis=1))
    return np.asarray(result)


class ScanExperiment:
    def __init__(self, folder, mode, grid):
        self.grid = grid
        self.graph = TopologyMemory()
        self.rec = Recorder(folder, mode, grid, self.graph)
        self.ref_hits = np.zeros(grid.log_odds.shape, bool)
        self.ref_free = np.zeros_like(self.ref_hits)
        self.ideal_free = np.zeros_like(self.ref_hits)
        self.prev = None
        self.route = []

    def mark(self, mask, pts):
        index = np.floor(
            (pts - [self.grid.origin_x, self.grid.origin_y]) / self.grid.resolution
        ).astype(int)
        c, r = index.T
        valid = (r >= 0) & (c >= 0) & (r < mask.shape[0]) & (c < mask.shape[1])
        mask[r[valid], c[valid]] = True

    def sweep(
        self,
        xy,
        heading,
        ranges,
        segments,
        title,
        detail,
        checks=None,
        path=None,
        graph_update=None,
    ):
        xy = np.asarray(xy, dtype=float)
        i = len(self.rec.steps)
        visits_before = int(self.grid.visit_count.sum())
        self.grid.update_scan(
            ranges,
            0,
            STEP,
            xy,
            heading,
            xy,
            i,
            range_min=0.05,
            range_max=LIMIT,
            base_path=path,
        )
        if graph_update:
            graph_update(self.graph)
        # Twice the angular resolution of the input. Mark free samples every 2.5 cm.
        angles = heading + np.arange(2 * N) * STEP / 2
        ref = segment_ranges(
            xy, angles, segments, front_only=self.rec.mode == "laser360"
        )
        dirs = directions(angles)
        good = np.isfinite(ref)
        self.mark(self.ref_hits, xy + ref[good, None] * dirs[good])
        offsets = np.arange(0, LIMIT, self.grid.resolution / 4)
        valid = offsets[None, :] < np.minimum(ref, LIMIT)[:, None]
        pts = xy + dirs[:, None, :] * offsets[None, :, None]
        self.mark(self.ideal_free, pts[valid])
        # Missing depth in a mesh hole is not a certified clear ray. Preserve the
        # ideal-visibility score separately; score integration on observable rays.
        supported = ~np.isnan(ranges)
        ray_support = np.repeat(supported, 2)
        ray_support[1::2] &= np.roll(supported, -1)
        self.mark(self.ref_free, pts[valid & ray_support[:, None]])
        pred = self.grid.grid() == 100
        free = self.grid.grid() == 0
        match = float((distance_transform_edt(~self.ref_hits)[pred] <= 1.5).mean())
        cover = (
            float((distance_transform_edt(~pred)[self.ref_hits] <= 1.5).mean())
            if pred.any()
            else 0.0
        )
        union = np.count_nonzero(pred | self.ref_hits)
        iou = np.count_nonzero(pred & self.ref_hits) / max(1, union)
        free_ref = self.ref_free & ~self.ref_hits
        free_iou = float(
            np.count_nonzero(free & free_ref)
            / max(1, np.count_nonzero(free | free_ref))
        )
        ideal = self.ideal_free & ~self.ref_hits
        ideal_free_iou = float(
            np.count_nonzero(free & ideal) / max(1, np.count_nonzero(free | ideal))
        )
        leak = float(
            (
                distance_transform_edt(~(self.ref_free | self.ref_hits))[free] > 1.5
            ).mean()
        )
        hits = np.isfinite(ranges) & (ranges >= 0.05) & (ranges < LIMIT)
        finite = hits & np.isfinite(ref[::2])
        err = np.abs(ranges[finite] - ref[::2][finite])
        agreement = float(np.mean(hits == np.isfinite(ref[::2])))
        metrics = dict(
            precision=match,
            recall=cover,
            iou=iou,
            free_iou=free_iou,
            ideal_free_iou=ideal_free_iou,
            unsupported_free=leak,
            range_p95=float(np.percentile(err, 95)) if len(err) else None,
            paired=int(finite.sum()),
            agreement=agreement,
        )
        evidence = [
            (
                "720 个方向，角间隔 0.5°，作为一帧提交",
                len(ranges) == 720 and self.grid.map_version == i + 1,
            ),
            ("与参考命中状态一致的方向达到 95%", agreement >= 0.95),
            (
                "匹配回波不少于 180 束且测距误差 P95 ≤ 0.10 m",
                len(err) >= 180 and metrics["range_p95"] <= 0.1,
            ),
            ("严格逐格 IoU：占用 ≥ 75%，有效束空闲 ≥ 90%", iou >= 0.75 and free_iou >= 0.90),
            ("累计可见表面的精确率、召回率均 ≥ 95%（容差 0.15 m）", min(match, cover) >= 0.95),
            ("参考可见空间之外的空闲格占比 ≤ 1%（容差 0.15 m）", leak <= 0.01),
        ]
        if self.prev is not None and np.array_equal(xy, self.prev):
            evidence.append(
                ("原地扫描没有再次计入访问次数", visits_before == self.grid.visit_count.sum())
            )
        if checks:
            evidence.extend(checks(self.grid))
        self.route.append(xy.tolist())
        self.rec.step(
            title,
            detail,
            evidence,
            xy,
            scan=dict(
                heading=float(heading),
                ranges=[
                    round(float(d), 4) if h else ("clear" if np.isposinf(d) else None)
                    for d, h in zip(ranges, hits)
                ],
                reference=np.flatnonzero(self.ref_hits).tolist(),
                limit=LIMIT,
                trajectory=self.route,
                hits=int(hits.sum()),
                clear=int(np.isposinf(ranges).sum()),
                invalid=int((~hits & ~np.isposinf(ranges)).sum()),
                metrics=metrics,
            ),
        )
        self.prev = xy.copy()


def rectangle(x0, y0, x1, y1):
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)


def room(folder):
    walls = (-4.0, -3.0, 4.0, 3.0)
    box = (-0.65, -0.8, 0.65, 0.8)
    seg = np.concatenate([rectangle(*walls), rectangle(*box)])
    grid = OccupancyMemory((81, 101), (-5.05, -4.05), 0.1)
    ex = ScanExperiment(folder, "room360", grid)
    poses = [
        (-2.0, 0.0, 0.0),
        (-2.0, 0.0, 90.0),
        (-2.0, 1.8, 90.0),
        (1.5, 1.8, 0.0),
        (2.0, 0.0, -90.0),
    ]
    for i, (x, y, yaw) in enumerate(poses):
        xy = np.array([x, y])
        heading = math.radians(yaw)
        # Analytic slab intersections, independently of the segment-based evaluator.
        dr = directions(ANGLES + heading)
        dr[np.abs(dr) < 1e-12] = 1e-12

        def interval(bounds):
            t0 = (np.array(bounds[:2]) - xy) / dr
            t1 = (np.array(bounds[2:]) - xy) / dr
            return np.minimum(t0, t1).max(axis=1), np.maximum(t0, t1).min(axis=1)

        _, distance = interval(walls)
        enter, leave = interval(box)
        distance = np.minimum(
            distance, np.where((enter > 0) & (leave >= enter), enter, np.inf)
        )

        def probes(g):
            read = lambda p: int(g.grid()[g.cell(p)])
            result = [
                ("实心柱内部保持未知", read((0, 0)) == -1),
                ("房间外侧保持未知", read((4.4, 0)) == -1),
            ]
            if i == 0:
                result.append(("柱后遮挡区保持未知", read((1.3, 0)) == -1))
                result.append(("已知正前方柱面距离为 1.35 m", abs(distance[0] - 1.35) < 1e-7))
            if i == 4:
                result.append(("绕到背面后，原遮挡区成为空闲", read((1.3, 0)) == 0))
            return result

        ex.sweep(
            xy,
            heading,
            np.where(distance < LIMIT, distance, np.inf),
            seg,
            ["首圈扫描", "原地转向 90°", "沿柱侧移动", "从另一侧观察", "验证原遮挡区"][i],
            "8 × 6 米解析房间，中央为实心矩形柱。每个位置同时融合完整一圈。",
            probes,
            None if i == 0 else [ex.prev, xy],
            lambda graph: graph.visit("r" + str(i), xy, i, "observation_place")
            if i != 1
            else None,
        )
