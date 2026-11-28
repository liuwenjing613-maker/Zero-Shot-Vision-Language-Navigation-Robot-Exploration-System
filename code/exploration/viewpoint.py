"""Viewpoint 生成与评分。

为什么要单独做 Viewpoint（队长设计文档第 4 节）：

  · 区域质心可能落在未知空间 / 贴着障碍 / 不可达
  · 机器人需要一个「安全、可达、视野好」的观察点
  · 所以：**区域质心只描述位置，导航目标是 viewpoint**

评分（设计文档 4.3 节）：

    J(v, F) = α·G(v,F) − β·D(v) − γ·R(v)

    G = 预期信息增益（射线能看到的未知格数）
    D = 从机器人到 v 的测地距离
    R = 风险（离障碍越近越高）

比基线强在哪：
  · 基线只取「离质心最近的可视格」；这里在区域内采样多个候选并打分
  · 信息增益用**射线遮挡**估计，而不是简单数格子
  · 路径成本用**测地距离**（沿可通行区域），而不是欧氏距离
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detector import FrontierRegion
from .gridops import FREE, free_mask, visible_unknown_cell_counts


@dataclass
class Viewpoint:
    cell: Tuple[int, int]          # (row, col)
    xy: Tuple[float, float]        # 世界 (x, z)
    gain_cells: int                # 可见未知格数量（原始值）
    clearance: float               # 离最近障碍/未知的距离（米）
    path_length: float             # 从机器人的测地距离（米）
    risk: float                    # 0..1，越高越危险
    score: float                   # 综合得分


@dataclass
class ViewpointConfig:
    sample_radius: float = 1.5        # 在质心周围多少米内采样候选（米）
    min_view_clearance: float = 0.30  # 观察点最小安全距离（米）
    safe_clearance: float = 0.60      # 风险归一化的参考安全距离（米）
    sensor_range: float = 4.0         # 射线长度（米）
    num_rays: int = 72                # 环视射线数（5° 一条）
    max_candidates: int = 400         # 每区域最多采样多少候选，控制耗时

    # J = alpha*G - beta*D - gamma*R
    alpha: float = 0.55
    beta: float = 0.30
    gamma: float = 0.15


def _to_world(row: float, col: float, resolution: float,
              origin: Tuple[float, float]) -> Tuple[float, float]:
    return (origin[0] + (col + 0.5) * resolution,
            origin[1] + (row + 0.5) * resolution)


def _candidate_cells(region: FrontierRegion, values: np.ndarray,
                     resolution: float, origin: Tuple[float, float],
                     geodesic: np.ndarray, clearance: np.ndarray,
                     cfg: ViewpointConfig) -> List[Tuple[int, int]]:
    """在区域质心周围采样候选观察点。

    条件：空闲、测地可达、离障碍足够远。
    """
    rows, cols = values.shape
    cr, cc = region.centroid_rc
    radius_cells = int(math.ceil(cfg.sample_radius / resolution))

    r_lo = max(0, int(cr) - radius_cells)
    r_hi = min(rows, int(cr) + radius_cells + 1)
    c_lo = max(0, int(cc) - radius_cells)
    c_hi = min(cols, int(cc) + radius_cells + 1)

    free = free_mask(values)
    candidates: List[Tuple[int, int]] = []
    for r in range(r_lo, r_hi):
        for c in range(c_lo, c_hi):
            if not free[r, c]:
                continue
            if not np.isfinite(geodesic[r, c]):
                continue
            if clearance[r, c] < cfg.min_view_clearance:
                continue
            candidates.append((r, c))

    # 候选太多时按「离质心近」优先截断（区域中心通常视野更完整）
    if len(candidates) > cfg.max_candidates:
        candidates.sort(key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - cc) ** 2)
        candidates = candidates[: cfg.max_candidates]
    return candidates


def generate_viewpoints(
    region: FrontierRegion,
    values: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    geodesic: np.ndarray,
    clearance: np.ndarray,
    cfg: Optional[ViewpointConfig] = None,
) -> List[Viewpoint]:
    """为单个区域生成并评分所有候选观察点（按 score 降序）。"""
    cfg = cfg or ViewpointConfig()
    candidates = _candidate_cells(region, values, resolution, origin,
                                  geodesic, clearance, cfg)

    # 没有安全候选时，退回到区域自身的格子（Frontier Cell 本身是空闲的）
    if not candidates:
        candidates = [(int(r), int(c)) for r, c in region.cells[: cfg.max_candidates]]
        candidates = [rc for rc in candidates if free_mask(values)[rc]]

    if not candidates:
        return []

    # 批量计算所有候选的信息增益（一次 numpy 调用，替代逐个 Python 循环）
    gains = visible_unknown_cell_counts(
        values, candidates, resolution, cfg.sensor_range, cfg.num_rays)

    raw = []
    for (r, c), gain in zip(candidates, gains):
        dist = float(geodesic[r, c])
        clr = float(clearance[r, c])
        risk = max(0.0, 1.0 - clr / cfg.safe_clearance)
        raw.append((r, c, int(gain), dist, clr, risk))

    # 这里用 min-max，而 scorer.py 用 ratio —— 两者不是笔误：
    #   · 观察点只在**本区域内部**比较（挑一个代表点），min-max 的绝对值不参与跨区域比较，
    #     且它偏好"本区域内增益最高"的候选，正是设计意图；
    #   · 区域之间要横向排序，尺度差异大，必须用 ratio 保留绝对差异（见 4.5 节）。
    gains_v = [x[2] for x in raw]
    dists_v = [x[3] for x in raw]
    g_lo, g_hi = (min(gains_v), max(gains_v)) if gains_v else (0, 0)
    d_lo, d_hi = (min(dists_v), max(dists_v)) if dists_v else (0, 0)

    viewpoints: List[Viewpoint] = []
    for (r, c, gain, dist, clr, risk) in raw:
        g_n = 0.5 if g_hi - g_lo < 1e-12 else (gain - g_lo) / (g_hi - g_lo)
        d_n = 0.5 if d_hi - d_lo < 1e-12 else (dist - d_lo) / (d_hi - d_lo)
        score = cfg.alpha * g_n - cfg.beta * d_n - cfg.gamma * risk
        viewpoints.append(Viewpoint(
            cell=(r, c),
            xy=_to_world(r, c, resolution, origin),
            gain_cells=int(gain),
            clearance=clr,
            path_length=dist,
            risk=risk,
            score=float(score),
        ))

    viewpoints.sort(key=lambda v: v.score, reverse=True)
    return viewpoints


def best_viewpoint(
    region: FrontierRegion,
    values: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    geodesic: np.ndarray,
    clearance: np.ndarray,
    cfg: Optional[ViewpointConfig] = None,
) -> Optional[Viewpoint]:
    """区域的最优观察点。第一版每个区域只保留一个（队长文档第 4.3 节）。"""
    views = generate_viewpoints(region, values, resolution, origin,
                                geodesic, clearance, cfg)
    return views[0] if views else None
