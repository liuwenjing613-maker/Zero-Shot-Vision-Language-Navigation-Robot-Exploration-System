"""模块三 × 队长 vln_memory 的集成测试。

验证两件事：
  1. 我的 `explorer.detect()` 产出的 snapshot 能被
     `vln_memory.frontier_memory.FrontierMemory.update_snapshot()` 直接接受；
  2. 记忆分配出的全局 ID 和历史计数（visit/failed）能正确传回打分。

依赖队长的 `src/vln_memory` 和 `src/vln_perception`。
找不到时整类测试自动跳过（不会让 CI 失败）。

跑法（仓库根目录）：
    PYTHONPATH=src python -m unittest code.tests.test_memory_integration -v

本地调试时可以用环境变量指定 src 位置：
    VLN_SRC=/path/to/repo/src python code/tests/test_memory_integration.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _cand in (ROOT / "src", pathlib.Path(os.environ.get("VLN_SRC", "/__none__"))):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))
        break

sys.path.insert(0, str(ROOT / "code"))

from exploration import FrontierExplorer  # noqa: E402
from exploration.viewpoint import ViewpointConfig  # noqa: E402

try:
    from vln_memory.frontier_memory import FrontierMemory  # noqa: E402
    from vln_memory.geometry_memory import OccupancyMemory  # noqa: E402

    HAVE_MEMORY = True
    SKIP_REASON = ""
except Exception as exc:  # noqa: BLE001
    HAVE_MEMORY = False
    SKIP_REASON = f"无法导入队长 vln_memory：{type(exc).__name__}: {exc}"


def make_memory() -> "OccupancyMemory":
    """构造一个和 demo 相同的场景，用队长的 OccupancyMemory 表示。

    log_odds 约定（见 geometry_memory.grid()）：
        <= -0.4 → 0 空闲
        >=  0.8 → 100 占用
        其余    → -1 未知
    """
    mem = OccupancyMemory((80, 80), (0.0, 0.0), 0.1)
    lo = mem.log_odds
    lo[:] = 0.0                    # 未知
    lo[25:60, 20:65] = -1.0        # 房间
    lo[25:38, 42] = 1.0            # 隔断墙 1
    lo[52:60, 42] = 1.0            # 隔断墙 2
    lo[40:46, 50:54] = 1.0         # 柱子
    mem.map_version = 1
    return mem


def fast_explorer() -> FrontierExplorer:
    return FrontierExplorer(
        viewpoint_config=ViewpointConfig(sample_radius=2.0, sensor_range=5.0,
                                         num_rays=72, max_candidates=200),
        min_cells=12,
    )


@unittest.skipUnless(HAVE_MEMORY, SKIP_REASON)
class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.mem = make_memory()
        self.values = self.mem.grid()
        self.explorer = fast_explorer()
        self.origin = (self.mem.origin_x, self.mem.origin_y)
        self.robot_cell = (55, 30)

    # ── 1. 栅格格式兼容 ──────────────────────────────────────

    def test_grid_values_match_expected_encoding(self):
        unique = set(np.unique(self.values).tolist())
        self.assertTrue(unique.issubset({-1, 0, 100}))
        self.assertIn(0, unique)
        self.assertIn(-1, unique)

    def test_explorer_accepts_occupancy_grid(self):
        result = self.explorer.explore(
            self.values, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        self.assertGreater(len(result), 0)

    # ── 2. snapshot 被 FrontierMemory 接受 ───────────────────

    def test_snapshot_accepted_by_frontier_memory(self):
        snap = self.explorer.detect(
            self.values, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        fm = FrontierMemory()
        mapping = fm.update_snapshot(snap)

        local_ids = {c["local_candidate_id"] for c in snap["candidates"]}
        self.assertEqual(set(mapping), local_ids)
        self.assertTrue(all(v.startswith("F") for v in mapping.values()))
        self.assertEqual(len(set(mapping.values())), len(mapping))   # 全局 ID 唯一

    def test_memory_records_created(self):
        snap = self.explorer.detect(
            self.values, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        fm = FrontierMemory()
        mapping = fm.update_snapshot(snap)
        self.assertEqual(len(fm.records), len(mapping))
        for gid in mapping.values():
            rec = fm.records[gid]
            self.assertIn("status", rec)
            self.assertIn("visit_count", rec)
            self.assertIn("failed_count", rec)
            self.assertEqual(rec["status"], "AVAILABLE")

    # ── 3. 全局 ID + 历史回传打分 ────────────────────────────

    def test_rank_uses_global_ids(self):
        snap = self.explorer.detect(
            self.values, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        fm = FrontierMemory()
        mapping = fm.update_snapshot(snap)

        ranked = self.explorer.rank(snap, fm.records, mapping)
        self.assertGreater(len(ranked), 0)
        for c in ranked:
            self.assertIn(c.frontier_id, fm.records)   # 全局 ID 对上了

    def test_history_lowers_utility(self):
        snap = self.explorer.detect(
            self.values, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        fm = FrontierMemory()
        mapping = fm.update_snapshot(snap)

        base = {c.frontier_id: c.local_utility
                for c in self.explorer.rank(snap, fm.records, mapping)}
        target = max(base, key=base.get)

        fm.records[target]["visit_count"] = 10
        fm.records[target]["failed_count"] = 10
        after = {c.frontier_id: c.local_utility
                 for c in self.explorer.rank(snap, fm.records, mapping)}

        self.assertLess(after[target], base[target])

    # ── 4. 第二轮快照：ID 延续 ──────────────────────────────

    def test_second_snapshot_reuses_ids(self):
        """地图更新后再检测，同一个区域应沿用原来的全局 ID。"""
        snap1 = self.explorer.detect(
            self.values, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        fm = FrontierMemory()
        map1 = fm.update_snapshot(snap1)

        # 地图版本 +1，几何不变
        self.mem.map_version = 2
        values2 = self.mem.grid()
        snap2 = self.explorer.detect(
            values2, self.mem.resolution, self.origin,
            self.robot_cell, self.mem.map_version,
        )
        map2 = fm.update_snapshot(snap2)

        self.assertEqual(set(map1.values()), set(map2.values()),
                         "同一区域应复用已有 frontier ID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
