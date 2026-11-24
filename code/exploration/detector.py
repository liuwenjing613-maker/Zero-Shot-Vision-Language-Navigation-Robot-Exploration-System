"""Frontier Cell 检测 + 区域聚类。

算法（对应队长设计文档第 3 节）：

  1. Frontier Cell：空闲格中，至少有一个 8 邻域格子是未知的。
  2. 安全过滤：离障碍/未知区域太近的格子不算（避免贴着墙）。
  3. 区域聚类：对 Frontier Cell 做 8 邻域连通分量，得到 Frontier Region。
  4. 小区域过滤：格子数少于阈值（默认 12）的区域丢弃。

注意：这里产出的是 **Region**（记忆单位），不是逐个 Cell。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from .gridops import FREE, clearance_map, dilate, free_mask, unknown_mask

# 8 邻域
_STRUCT_8 = np.ones((3, 3), dtype=bool)


@dataclass
class FrontierRegion:
    """一个聚类后的前沿区域（检测层内部结构）。"""

    cells: np.ndarray              # (N, 2) 的 [row, col] 数组
    centroid_rc: Tuple[float, float]
    centroid_xy: Tuple[float, float]
    cluster_size: int


def detect_frontier_cells(
    values: np.ndarray,
    resolution: float,
    min_clearance: float = 0.0,
) -> np.ndarray:
    """返回布尔掩码：哪些格子是 Frontier Cell。

    定义：空闲格中，8 邻域内至少有一个未知格。

    ⚠️ 注意 `min_clearance`：
        Frontier Cell 按定义就紧邻未知区域，`clearance_map` 只有约 1 格（0.1m）。
        所以**不要**用它来过滤 Frontier Cell —— 会全部筛掉。
        安全距离只在 `viewpoint.py` 里用于筛选**观察点**。
        这里保留参数仅为兼容，默认 0（不过滤）。

    Args:
        values: 占据栅格，-1 未知 / 0 空闲 / 100 占用
        resolution: 每格边长（米）
        min_clearance: 仅当明确需要时使用（默认 0）
    """
    free = free_mask(values)
    unknown = unknown_mask(values)

    # 空闲 且 8 邻域内有未知
    adjacent_unknown = dilate(unknown)
    frontier = free & adjacent_unknown

    if min_clearance > 0:
        frontier &= clearance_map(values, resolution) >= min_clearance

    return frontier


def cluster_frontier_cells(
    frontier_mask: np.ndarray,
    resolution: float,
    min_cells: int = 12,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> List[FrontierRegion]:
    """8 邻域连通分量聚类，过滤过小的区域。

    Args:
        frontier_mask: detect_frontier_cells 的输出
        resolution: 每格边长（米）
        min_cells: 最小格子数（默认 12，与队长 evaluate() 的阈值一致）
        origin: (origin_x, origin_z)，用于把栅格坐标转世界坐标
    """
    labels, count = ndi.label(frontier_mask, structure=_STRUCT_8)
    regions: List[FrontierRegion] = []

    for idx in range(1, count + 1):
        cells = np.argwhere(labels == idx)
        if len(cells) < min_cells:
            continue

        rc = cells.mean(axis=0)
        x = origin[0] + (rc[1] + 0.5) * resolution
        z = origin[1] + (rc[0] + 0.5) * resolution

        regions.append(
            FrontierRegion(
                cells=cells,
                centroid_rc=(float(rc[0]), float(rc[1])),
                centroid_xy=(float(x), float(z)),
                cluster_size=int(len(cells)),
            )
        )

    # 大区域优先（探索价值通常更高）
    regions.sort(key=lambda r: r.cluster_size, reverse=True)
    return regions


def detect_regions(
    values: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    min_clearance: float = 0.0,
    min_cells: int = 12,
) -> List[FrontierRegion]:
    """检测 + 聚类的组合入口。"""
    mask = detect_frontier_cells(values, resolution, min_clearance)
    return cluster_frontier_cells(mask, resolution, min_cells, origin)
