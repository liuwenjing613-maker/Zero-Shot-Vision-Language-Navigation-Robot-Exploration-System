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
from typing import Dict, List, Sequence, Set, Tuple

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


# 射线采样点相对起点的偏移缓存：(射线数, 步数) → (sin 分量, cos 分量)
_RAY_OFFSET_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
_RAY_STEP = 0.9      # 步长（格），略小于 1 格，避免跨格漏检


def _ray_offsets(num_rays: int, max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """第 i 步、第 k 条射线的行/列偏移（单位：格），形状均为 (num_rays, max_steps)。

    与原始实现保持一致：行方向用 sin、列方向用 cos，步长 0.9 格。
    """
    key = (num_rays, max_steps)
    cached = _RAY_OFFSET_CACHE.get(key)
    if cached is None:
        angles = 2.0 * np.pi * np.arange(num_rays) / num_rays
        steps = np.arange(1, max_steps + 1, dtype=np.float64) * _RAY_STEP
        sin_off = np.sin(angles)[:, None] * steps[None, :]
        cos_off = np.cos(angles)[:, None] * steps[None, :]
        cached = (sin_off, cos_off)
        _RAY_OFFSET_CACHE[key] = cached
    return cached


def _visible_cells_of(values: np.ndarray, r0: int, c0: int,
                      sensor_range: float, resolution: float,
                      num_rays: int) -> Tuple[np.ndarray, np.ndarray]:
    """单个起点：返回可见未知格的 (行数组, 列数组)，已去重。"""
    rows, cols = values.shape
    max_steps = max(1, int(round(sensor_range / resolution)))
    sin_off, cos_off = _ray_offsets(num_rays, max_steps)

    # 注意：必须先做 r0 + 偏移 再取整，不能先取整偏移再加上 r0 ——
    # 银行家舍入对奇偶敏感，两种写法在 x.5 处结果不同。
    rr = np.rint(r0 + sin_off).astype(np.int32)
    cc = np.rint(c0 + cos_off).astype(np.int32)

    inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    r_clip = np.clip(rr, 0, rows - 1)
    c_clip = np.clip(cc, 0, cols - 1)
    v = values[r_clip, c_clip]

    # 越界或占用 → 该采样点及其之后全部被挡住
    blocked = (~inside) | (v >= 50)
    blocked = np.maximum.accumulate(blocked, axis=1)

    seen = (v == UNKNOWN) & (~blocked) & inside
    if not seen.any():
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    r_hit = rr[seen]
    c_hit = cc[seen]
    packed = r_hit.astype(np.int64) * cols + c_hit
    uniq, idx = np.unique(packed, return_index=True)
    return r_hit[idx], c_hit[idx]


def visible_unknown_cells(values: np.ndarray, cell: Tuple[int, int],
                          resolution: float, sensor_range: float,
                          num_rays: int = 72) -> Set[Tuple[int, int]]:
    """从 cell 环视一圈，返回可见的未知格集合（被占用格挡住）。

    这是信息增益的核心估计：站在这儿能看到多少还没探索的地方。
    """
    r_hit, c_hit = _visible_cells_of(values, int(cell[0]), int(cell[1]),
                                     sensor_range, resolution, num_rays)
    return set(zip(r_hit.tolist(), c_hit.tolist()))


def visible_unknown_cell_counts(values: np.ndarray, cells: Sequence[Tuple[int, int]],
                                resolution: float, sensor_range: float,
                                num_rays: int = 72) -> List[int]:
    """批量版本：一次算出多个起点的可见未知格数量。

    `generate_viewpoints` 每轮要评估几十到几百个候选观察点，逐个调用
    `visible_unknown_cells` 时 Python 循环占了大头；这里把取整、索引、
    遮挡累积都交给 numpy，实测比逐个调用快一个数量级。
    """
    if not cells:
        return []

    rows, cols = values.shape
    max_steps = max(1, int(round(sensor_range / resolution)))
    sin_off, cos_off = _ray_offsets(num_rays, max_steps)     # (R, S)

    # 分块：让每块的 (chunk × 射线 × 步数) 落在几万元素的量级，
    # 太小则 numpy 调用开销占比高，太大则缓存不友好。实测 16 附近最优。
    per_ray = num_rays * max_steps
    chunk = max(1, min(len(cells), 50000 // max(1, per_ray)))

    counts: List[int] = []
    buf = np.zeros(rows * cols, dtype=bool)                   # 复用去重缓冲

    for start in range(0, len(cells), chunk):
        part = np.asarray(cells[start:start + chunk], dtype=np.float64)
        r0 = part[:, 0:1, None]
        c0 = part[:, 1:2, None]

        rr = np.rint(r0 + sin_off[None, :, :]).astype(np.int32)   # (N, R, S)
        cc = np.rint(c0 + cos_off[None, :, :]).astype(np.int32)

        inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        v = values[np.clip(rr, 0, rows - 1), np.clip(cc, 0, cols - 1)]

        blocked = np.maximum.accumulate((~inside) | (v >= 50), axis=2)
        seen = (v == UNKNOWN) & (~blocked) & inside               # (N, R, S)

        for n in range(seen.shape[0]):
            mask_n = seen[n]
            if not mask_n.any():
                counts.append(0)
                continue
            packed = rr[n][mask_n].astype(np.int64) * cols + cc[n][mask_n]
            buf[packed] = True
            counts.append(int(buf.sum()))
            buf[packed] = False
    return counts


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
