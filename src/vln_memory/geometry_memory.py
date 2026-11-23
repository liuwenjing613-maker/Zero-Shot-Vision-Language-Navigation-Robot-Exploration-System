"""Metric Habitat world X/Z -> map x/y. Arrays are indexed [row(z), col(x)]."""
from dataclasses import dataclass
import math
import operator

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if not all(operator.index(n) > 0 for n in (self.width, self.height)):
            raise ValueError("Invalid image dimensions")
        if not np.isfinite((self.fx, self.fy, self.cx, self.cy)).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("Invalid pinhole calibration")

    @classmethod
    def from_hfov(cls, width, height, degrees):
        if not 0 < degrees < 180:
            raise ValueError("Horizontal field of view must be in (0, 180)")
        focal = 0.5 * width / math.tan(math.radians(degrees) / 2)
        return cls(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2)


def _ray_cells(start, end):
    """Traverse positive-length sections, and include the measured endpoint."""
    col, row = map(math.floor, start)
    end_col, end_row = map(math.floor, end)
    dx, dy = end[0] - start[0], end[1] - start[1]
    sx, sy = int(np.sign(dx)), int(np.sign(dy))
    tx = (col + (sx > 0) - start[0]) / dx if sx else math.inf
    ty = (row + (sy > 0) - start[1]) / dy if sy else math.inf
    stepx = 1 / abs(dx) if sx else math.inf
    stepy = 1 / abs(dy) if sy else math.inf
    while (col, row) != (end_col, end_row):
        crossing = min(tx, ty)
        if crossing > 0:
            yield row, col
        if crossing >= 1:
            break
        if tx == crossing:
            col += sx
            tx += stepx
        if ty == crossing:
            row += sy
            ty += stepy
    yield end_row, end_col


class OccupancyMemory:
    """A fixed, half-open map. Allocate once per scene and floor."""

    def __init__(self, shape, origin, resolution=0.1):
        rows, cols = map(operator.index, shape)
        self.origin_x, self.origin_y = map(float, origin)
        self.resolution = float(resolution)
        if min(rows, cols) <= 0 or not np.isfinite(
            (self.origin_x, self.origin_y, self.resolution)
        ).all() or self.resolution <= 0:
            raise ValueError("Invalid grid dimensions, origin, or resolution")
        self.log_odds = np.zeros((rows, cols), dtype=np.float32)
        self.visit_count = np.zeros((rows, cols), dtype=np.uint64)
        self._last_cell = None
        self._last_xy = None
        self.map_version = 0
        self.timestamp = None

    def cell(self, xy):
        x, y = xy
        col = math.floor((x - self.origin_x) / self.resolution)
        row = math.floor((y - self.origin_y) / self.resolution)
        return row, col

    def contains(self, cell):
        row, col = cell
        rows, cols = self.log_odds.shape
        return 0 <= row < rows and 0 <= col < cols

    def grid(self):
        result = np.full(self.log_odds.shape, -1, dtype=np.int8)
        result[self.log_odds <= -0.4] = 0
        result[self.log_odds >= 0.8] = 100
        return result

    def _visit(self, cell):
        if cell != self._last_cell:
            self.visit_count[cell] += 1
            self._last_cell = cell

    def update_depth(self, depth, intrinsics, world_from_camera, robot_xyz, timestamp,
                     *, floor_y=None, obstacle_heights=(0.12, 1.65),
                     depth_range=(0.05, 8.0), stride=4, base_path=None):
        """Integrate one metric axial depth frame. Habitat camera looks down -Z.

        Image v grows downwards. floor_y is in world meters, defaulting to the
        robot base height. Samples outside the open depth range carry no evidence.
        A scene stream must supply a strictly increasing acquisition timestamp.
        Optional base_path contains verified continuous map-XY polyline vertices
        from the previous base sample to this one. With poses alone only sampled
        centers are counted, so a reset or teleport cannot invent coverage.
        """
        image = np.asarray(depth)
        T = np.asarray(world_from_camera, dtype=float)
        base = np.asarray(robot_xyz, dtype=float)
        if image.shape != (intrinsics.height, intrinsics.width) or image.dtype.kind not in "fiu":
            raise ValueError("Expected calibrated H x W numeric depth in meters")
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError("Expected a homogeneous camera-to-world transform")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError("Camera rotation must be proper and orthonormal")
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("Invalid robot base position")
        cell = self.cell(base[[0, 2]])
        if not self.contains(cell):
            raise ValueError("Robot is outside this scene grid")
        if not math.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError("Expected a fresh sensor timestamp")
        floor = base[1] if floor_y is None else float(floor_y)
        low, high = obstacle_heights
        near, far = depth_range
        stride = operator.index(stride)
        if not np.isfinite((floor, low, high, near, far)).all() or not (
            0 <= low < high and 0 <= near < far and stride > 0
        ):
            raise ValueError("Invalid height band, depth range, or stride")

        xy = base[[0, 2]]
        visits = self._base_visits(xy, cell, base_path)

        v, u = np.mgrid[0:image.shape[0]:stride, 0:image.shape[1]:stride]
        z = image[::stride, ::stride]
        valid = np.isfinite(z) & (z > near) & (z < far)
        z, u, v = z[valid], u[valid], v[valid]
        camera_points = np.column_stack(((u - intrinsics.cx) * z / intrinsics.fx,
                                         (intrinsics.cy - v) * z / intrinsics.fy, -z.astype(float)))
        eye = T[:3, 3]
        world = camera_points @ R.T + eye
        # Clip each free-space segment to the obstacle-height slab. This avoids
        # clearing cells behind high/low surfaces based on an overhead ray.
        lower, upper = floor + low, floor + high
        frees, hits = set(), set()
        origin = np.array([self.origin_x, self.origin_y])
        start = (eye[[0, 2]] - origin) / self.resolution
        for point in world:
            dy = point[1] - eye[1]
            entry, leave = 0.0, 1.0
            if dy:
                a, b = sorted(((lower - eye[1]) / dy, (upper - eye[1]) / dy))
                entry, leave = max(entry, a), min(leave, b)
            elif not lower <= eye[1] <= upper:
                continue
            if entry > leave:
                continue
            delta = (point[[0, 2]] - eye[[0, 2]]) / self.resolution
            for crossed in _ray_cells(start + entry * delta, start + leave * delta):
                if self.contains(crossed):
                    frees.add(crossed)
            if lower <= point[1] <= upper:
                hit = self.cell(point[[0, 2]])
                if self.contains(hit):
                    hits.add(hit)

        self._commit_frame(frees, hits, visits, xy, timestamp)

    def _base_visits(self, xy, cell, base_path):
        visits = [cell]
        if base_path is not None:
            path = np.asarray(base_path, dtype=float)
            if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
                raise ValueError("Expected at least two finite map-XY polyline vertices")
            if self._last_xy is None or not np.array_equal(path[0], self._last_xy) or not np.array_equal(path[-1], xy):
                raise ValueError("Path must connect previous and current sampled base positions")
            if not all(self.contains(self.cell(p)) for p in path):
                raise ValueError("Base path leaves the scene grid")
            grid_path = (path - [self.origin_x, self.origin_y]) / self.resolution
            visits = [c for a, b in zip(grid_path, grid_path[1:]) for c in _ray_cells(a, b)]
        return visits

    def _commit_frame(self, frees, hits, visits, xy, timestamp):
        # One vote per cell and frame. Hit wins if another ray crosses it.
        for cells, change in ((frees - hits, -0.45), (hits, 0.9)):
            if cells:
                indices = tuple(zip(*cells))
                self.log_odds[indices] = np.clip(self.log_odds[indices] + change, -4, 4)
        for visited in visits:
            self._visit(visited)
        self._last_xy = xy.copy()
        self.timestamp = float(timestamp)
        self.map_version += 1

    def update_scan(self, ranges, angle_min, angle_increment, sensor_xy, sensor_heading,
                    base_xy, timestamp, *, range_min, range_max, base_path=None):
        """Fuse one synchronized planar scan, using radians and meters in map X/Z.

        Heading zero points along +X, positive angles turn toward +Z. Finite
        values in [range_min, range_max) are hits. +inf is a confirmed clear
        beam to range_max. NaN, -inf and out-of-range finite readings are
        discarded. The driver must translate its own no-return convention.
        """
        distances = np.asarray(ranges, dtype=float)
        eye, base = np.asarray(sensor_xy, dtype=float), np.asarray(base_xy, dtype=float)
        if distances.ndim != 1 or not distances.size:
            raise ValueError("Expected a nonempty vector of planar ranges")
        if eye.shape != (2,) or base.shape != (2,) or not np.isfinite([eye, base]).all():
            raise ValueError("Invalid sensor or robot map position")
        if not np.isfinite([angle_min, angle_increment, sensor_heading, range_min, range_max, timestamp]).all():
            raise ValueError("Invalid scan metadata")
        if angle_increment == 0 or not 0 <= range_min < range_max:
            raise ValueError("Invalid angular increment or range limits")
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("Expected a fresh sensor timestamp")
        cell = self.cell(base)
        if not self.contains(cell) or not self.contains(self.cell(eye)):
            raise ValueError("Robot or sensor outside scene grid")
        visits = self._base_visits(base, cell, base_path)
        origin = np.array([self.origin_x, self.origin_y])
        start = (eye-origin) / self.resolution
        frees, hits = set(), set()
        for i, distance in enumerate(distances):
            hit = range_min <= distance < range_max and distance > 0
            if not hit and not np.isposinf(distance):
                continue
            theta = sensor_heading + angle_min + i * angle_increment
            length = distance if hit else range_max
            end = eye + length * np.array([math.cos(theta), math.sin(theta)])
            for crossed in _ray_cells(start, (end-origin) / self.resolution):
                if self.contains(crossed):
                    frees.add(crossed)
            if hit and self.contains(self.cell(end)):
                hits.add(self.cell(end))
        self._commit_frame(frees, hits, visits, base, timestamp)
