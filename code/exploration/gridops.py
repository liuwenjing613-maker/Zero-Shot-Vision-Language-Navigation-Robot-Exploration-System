"""栅格运算工具：掩码、距离变换、测地距离、射线扫描。

坐标约定（与队长 `geometry_memory.py` 保持一致）：
  · 地图平面是世界 (X, Z)，数组索引是 [Z 行, X 列]
  · 世界坐标 (x, z) → 栅格 (row, col)
  · grid 取值：-1 未知、0 空闲、100 占用

所有函数只依赖 numpy + scipy.ndimage，不依赖 Habitat。
"""

from __future__ import annotations

import heapq
import math
from typing import List, Sequence, Set, Tuple

import numpy as np
from scipy import ndimage as ndi

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# 8 邻域
_NEIGHBORS_8 = np.ones((3, 3), dtype=bool)
# Dijkstra 的 8 个方向及代价
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ══════════════════════════════════════════════════════════════════
# 掩码
# ══════════════════════════════════════════════════════════════════


def free_mask(values: np.ndarray) -> np.ndarray:
    return values == FREE


def unknown_mask(values: np.ndarray) -> np.ndarray:
    return values == UNKNOWN


def occupied_mask(values: np.ndarray) -> np.ndarray:
    """占用判定用阈值而非精确等于 100 —— 兼容 log-odds 之类的表示。"""
    return values >= 50


def dilate(mask: np.ndarray) -> np.ndarray:
    """8 邻域膨胀一格。"""
    return ndi.binary_dilation(mask, structure=_NEIGHBORS_8)


# ══════════════════════════════════════════════════════════════════
# 距离
# ══════════════════════════════════════════════════════════════════


def clearance_map(values: np.ndarray, resolution: float) -> np.ndarray:
    """每个格子到最近「非空闲」格子的距离（米）。

    非空闲 = 占用 或 未知 或 地图外（distance_transform 把零值当目标）。
    注意：这里把未知也当成障碍，保证观察点不会贴着未探测区域。
    """
    free = free_mask(values)
    return ndi.distance_transform_edt(free) * resolution


def geodesic_map(values: np.ndarray, start_cell: Tuple[int, int],
                 resolution: float) -> np.ndarray:
    """从 start_cell 出发、只在空闲格上走的测地距离图（米）。

    8 邻域 Dijkstra：直边代价 1，斜边代价 sqrt(2)。
    不可达的格子为 inf。
    """
    rows, cols = values.shape
    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    sr, sc = start_cell
    if not (0 <= sr < rows and 0 <= sc < cols) or values[sr, sc] != FREE:
        return dist

    dist[sr, sc] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, sr, sc)]
    step = resolution
    diag = resolution * math.sqrt(2.0)

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if values[nr, nc] != FREE:
                continue
            nd = d + (diag if dr and dc else step)
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


# ══════════════════════════════════════════════════════════════════
# 射线
# ══════════════════════════════════════════════════════════════════


def visible_unknown_cells(values: np.ndarray, cell: Tuple[int, int],
                          resolution: float, sensor_range: float,
                          num_rays: int = 72) -> Set[Tuple[int, int]]:
    """从 cell 环视一圈，返回可见的未知格集合（被占用格挡住）。

    这是信息增益的核心估计：站在这儿能看到多少还没探索的地方。
    """
    rows, cols = values.shape
    r0, c0 = cell
    max_steps = max(1, int(round(sensor_range / resolution)))
    step = resolution * 0.9  # 略小于 1 格，避免跨格漏检
    seen: Set[Tuple[int, int]] = set()

    for k in range(num_rays):
        angle = 2.0 * math.pi * k / num_rays
        dx, dy = math.cos(angle), math.sin(angle)
        for i in range(1, max_steps + 1):
            r = int(round(r0 + i * step / resolution * dy))
            c = int(round(c0 + i * step / resolution * dx))
            if not (0 <= r < rows and 0 <= c < cols):
                break
            v = values[r, c]
            if v >= 50:          # 占用，射线终止
                break
            if v == UNKNOWN:
                seen.add((r, c))
    return seen


def normalize(values: Sequence[float]) -> List[float]:
    """min-max 归一化到 [0,1]；全相同时返回 0.5（中性值）。

    ⚠️ 慎用：min-max 会把微小差异放大到 0/1。
    例如 1301 vs 1219 个可见未知格会变成 1.0 vs 0.0，
    排序没错但数值会误导。打分请优先用 `ratio()`。
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def ratio(values: Sequence[float]) -> List[float]:
    """比值归一化：v / max(v)，保留相对差异，结果在 [0,1]。

    1301 vs 1219 → 1.0 vs 0.937（而不是 1.0 vs 0.0）。
    全为 0 或负数时返回全 0。
    """
    if not values:
        return []
    top = max(values)
    if top <= 0:
        return [0.0 for _ in values]
    return [max(0.0, v) / top for v in values]
