"""FrontierExplorer —— 模块三的对外入口。

职责边界（队长设计文档第 9 节）：

    Exploration（本模块）：Frontier Cell 检测 → 区域聚类 → Viewpoint 生成 → 打分排序
    Memory（队长）：       区域身份分配、历史状态、访问/失败计数

所以本模块**不维护全局 frontier ID**。它产出 snapshot（带 local id），
由 `FrontierMemory.update_snapshot()` 分配全局 ID 并返回映射，
再用 `rank(snapshot, records, mapping)` 把历史信息接回打分。

第一版流程：
    grid ──► detect() ──► snapshot ──► FrontierMemory.update_snapshot()
                                            │
                                            ▼ mapping + records
                                       rank() ──► list[FrontierCandidate]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.schemas import FrontierCandidate, Position3D
from .detector import detect_frontier_cells, cluster_frontier_cells
from .gridops import clearance_map, geodesic_map
from .scorer import UtilityWeights, score_candidates
from .viewpoint import ViewpointConfig, best_viewpoint


class FrontierExplorer:
    def __init__(
        self,
        viewpoint_config: Optional[ViewpointConfig] = None,
        weights: Optional[UtilityWeights] = None,
        min_cells: int = 12,
        min_clearance: float = 0.0,
    ) -> None:
        self.viewpoint_config = viewpoint_config or ViewpointConfig()
        self.weights = weights or UtilityWeights()
        self.min_cells = min_cells
        self.min_clearance = min_clearance

    # ══════════════════════════════════════════════════════════════
    # 检测
    # ══════════════════════════════════════════════════════════════

    def detect(
        self,
        values: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        robot_cell: Tuple[int, int],
        map_version: int,
        complete: bool = True,
        floor_y: float = 0.0,
    ) -> Dict:
        """检测前沿、聚类、为每个区域选最优 Viewpoint，产出 snapshot。

        Args:
            values: 占据栅格（-1 未知 / 0 空闲 / 100 占用）
            resolution: 每格边长（米）
            origin: (origin_x, origin_z)
            robot_cell: 机器人所在栅格 (row, col)
            map_version: 地图版本号
            complete: 本轮扫描是否完整结束
            floor_y: 地面高度（世界 Y），用于补全 3D 坐标

        返回的 snapshot 可直接传给 `FrontierMemory.update_snapshot()`；
        额外的字段（cluster_size / gain_cells / path_length …）它不认识，会被忽略。
        """
        values = np.asarray(values)
        clearance = clearance_map(values, resolution)
        geodesic = geodesic_map(values, robot_cell, resolution)

        mask = detect_frontier_cells(values, resolution, self.min_clearance)
        regions = cluster_frontier_cells(
            mask, resolution, self.min_cells, origin)

        candidates: List[Dict] = []
        for idx, region in enumerate(regions, start=1):
            view = best_viewpoint(region, values, resolution, origin,
                                  geodesic, clearance, self.viewpoint_config)
            if view is None:
                continue

            candidates.append({
                # ── FrontierMemory.update_snapshot 认识的四个字段 ──
                "local_candidate_id": str(idx),
                "centroid_xy": region.centroid_xy,
                "safe_viewpoint_xy": view.xy,
                "cluster_cells": region.cells.tolist(),
                # ── 打分用的扩展字段（update_snapshot 会忽略）──
                "cluster_size": region.cluster_size,
                "gain_cells": view.gain_cells,
                "path_length": view.path_length,
                "clearance": view.clearance,
                "reachable": bool(np.isfinite(view.path_length)),
                "floor_y": float(floor_y),
            })

        return {
            "snapshot_id": str(map_version),
            "map_version": int(map_version),
            "complete": bool(complete),
            "candidates": candidates,
        }

    # ══════════════════════════════════════════════════════════════
    # 打分排序
    # ══════════════════════════════════════════════════════════════

    def rank(
        self,
        snapshot: Dict,
        records: Optional[Dict[str, Dict]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ) -> List[FrontierCandidate]:
        """把 snapshot 转成 FrontierCandidate 并打分排序。

        Args:
            snapshot: detect() 的产物
            records: `FrontierMemory.records`（全局 ID → 记录），可空
            mapping: `update_snapshot()` 返回的 local → global 映射，可空

        没有 records / mapping 时（单元测试场景），用 local id 当全局 id。
        """
        records = records or {}
        mapping = mapping or {}

        candidates: List[FrontierCandidate] = []
        for item in snapshot.get("candidates", []):
            local_id = item["local_candidate_id"]
            global_id = mapping.get(local_id, f"F{local_id}")
            record = records.get(global_id, {})

            y = float(item.get("floor_y", 0.0))
            centroid_xy = item["centroid_xy"]
            view_xy = item["safe_viewpoint_xy"]

            candidates.append(FrontierCandidate(
                frontier_id=global_id,
                centroid=_as_3d(centroid_xy, y),
                safe_viewpoint=_as_3d(view_xy, y),
                cluster_size=int(item.get("cluster_size", 0)),
                reachable=bool(item.get("reachable", True)),
                path_length=float(item.get("path_length", 0.0)),
                clearance=float(item.get("clearance", 0.0)),
                information_gain=float(item.get("gain_cells", 0)),
                novelty=0.0,                       # 由 scorer 统一算
                visit_count=int(record.get("visit_count", 0)),
                failed_count=int(record.get("failed_count", 0)),
                status=str(record.get("status", "NEW")),
            ))

        return score_candidates(candidates, self.weights)

    # ══════════════════════════════════════════════════════════════
    # 组合入口
    # ══════════════════════════════════════════════════════════════

    def explore(
        self,
        values: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        robot_cell: Tuple[int, int],
        map_version: int,
        records: Optional[Dict[str, Dict]] = None,
        mapping: Optional[Dict[str, str]] = None,
        complete: bool = True,
        floor_y: float = 0.0,
    ) -> List[FrontierCandidate]:
        """detect + rank 一步到位（不需要记忆模块时用）。"""
        snapshot = self.detect(values, resolution, origin, robot_cell,
                               map_version, complete, floor_y)
        return self.rank(snapshot, records, mapping)

    @staticmethod
    def select_best(candidates: Sequence[FrontierCandidate]) -> Optional[FrontierCandidate]:
        """选出得分最高的候选（第一版每个决策周期只去一个）。"""
        return candidates[0] if candidates else None


def _as_3d(xy: Sequence[float], y: float) -> Position3D:
    return (float(xy[0]), float(y), float(xy[1]))
