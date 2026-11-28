"""模块三 Semantic Exploration 的单元测试。

不依赖 Habitat / GPU / 网络，用合成的占据栅格跑。

跑法：
    cd tls
    python code/tests/test_exploration.py

覆盖：
  · Frontier Cell 检测（只在空闲格、必须邻接未知）
  · 区域聚类 + 小区域过滤
  · Viewpoint 生成（在空闲区、安全距离、可达）
  · Utility 打分（信息增益 / 历史惩罚 / 风险）
  · 排序与输出契约（FrontierCandidate）
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.schemas import FrontierCandidate  # noqa: E402
from exploration.detector import (  # noqa: E402
    cluster_frontier_cells,
    detect_frontier_cells,
    detect_regions,
)
from exploration.explorer import FrontierExplorer  # noqa: E402
from exploration.gridops import (  # noqa: E402
    FREE,
    OCCUPIED,
    UNKNOWN,
    clearance_map,
    geodesic_map,
    normalize,
    visible_unknown_cell_counts,
    visible_unknown_cells,
)
from exploration.scorer import (  # noqa: E402
    UtilityWeights,
    history_penalty,
    novelty,
    score_candidates,
)
from exploration.viewpoint import ViewpointConfig, best_viewpoint  # noqa: E402

RES = 0.1  # 每格 0.1 米


def room_grid() -> np.ndarray:
    """一个 60×60 的栅格：中间是空闲房间，四周是未知区域。

    行索引 = Z，列索引 = X（与队长 geometry_memory 的约定一致）。
    """
    values = np.full((60, 60), UNKNOWN, dtype=np.int16)
    values[20:40, 10:50] = FREE
    return values


def two_rooms_grid() -> np.ndarray:
    """两个互不相邻的空闲区域 → 两个独立的前沿区域。"""
    values = np.full((60, 60), UNKNOWN, dtype=np.int16)
    values[10:20, 10:20] = FREE
    values[40:50, 40:50] = FREE
    return values


def fast_config() -> ViewpointConfig:
    """测试用的轻量配置：射线少、采样少，保证秒级跑完。"""
    return ViewpointConfig(
        sample_radius=1.0,
        sensor_range=2.0,
        num_rays=16,
        max_candidates=60,
    )


# ══════════════════════════════════════════════════════════════════
# 栅格工具
# ══════════════════════════════════════════════════════════════════


class TestGridOps(unittest.TestCase):
    def test_clearance_zero_on_obstacle(self):
        values = np.zeros((10, 10), dtype=np.int16)
        values[5, 5] = OCCUPIED
        clearance = clearance_map(values, RES)
        self.assertEqual(clearance[5, 5], 0.0)
        self.assertGreater(clearance[0, 0], 0.0)

    def test_geodesic_blocked_by_obstacle(self):
        """一排障碍把地图分成两半，另一半应不可达。"""
        values = np.zeros((10, 10), dtype=np.int16)
        values[:, 5] = OCCUPIED           # 中间一堵墙
        geo = geodesic_map(values, (0, 0), RES)
        self.assertTrue(np.isfinite(geo[0, 0]))
        self.assertTrue(np.isinf(geo[0, 9]))   # 墙后不可达
        # 绕不过去（墙贯穿整列）
        self.assertTrue(np.isinf(geo[9, 9]))

    def test_geodesic_around_wall(self):
        """墙留一个缺口时，应能绕过去。"""
        values = np.zeros((10, 10), dtype=np.int16)
        values[:8, 5] = OCCUPIED           # 墙只到第 8 行，下方留缺口
        geo = geodesic_map(values, (0, 0), RES)
        self.assertTrue(np.isfinite(geo[0, 9]))
        self.assertGreater(geo[0, 9], 10 * RES)   # 绕路，比直线长

    def test_visible_unknown_stops_at_obstacle(self):
        values = np.full((30, 30), UNKNOWN, dtype=np.int16)
        values[10:20, 10:20] = FREE
        values[10, 15] = OCCUPIED          # 一个遮挡物
        seen = visible_unknown_cells(values, (15, 15), RES, 2.0, 16)
        self.assertGreater(len(seen), 0)
        self.assertTrue(all(values[r, c] == UNKNOWN for r, c in seen))

    def test_normalize(self):
        self.assertEqual(normalize([1, 2, 3]), [0.0, 0.5, 1.0])
        self.assertEqual(normalize([5, 5, 5]), [0.5, 0.5, 0.5])
        self.assertEqual(normalize([]), [])


# ══════════════════════════════════════════════════════════════════
# 检测与聚类
# ══════════════════════════════════════════════════════════════════


class TestDetector(unittest.TestCase):
    def test_frontier_cells_exist_on_boundary(self):
        values = room_grid()
        mask = detect_frontier_cells(values, RES)
        self.assertGreater(int(mask.sum()), 0)

    def test_frontier_cells_are_free(self):
        values = room_grid()
        mask = detect_frontier_cells(values, RES)
        self.assertTrue(np.all(values[mask] == FREE))

    def test_no_frontier_when_everything_known(self):
        values = np.zeros((40, 40), dtype=np.int16)
        mask = detect_frontier_cells(values, RES)
        self.assertEqual(int(mask.sum()), 0)

    def test_clusters_are_separate(self):
        """两个互不相邻的空闲区域应产生两个独立 cluster。"""
        values = np.full((60, 60), UNKNOWN, dtype=np.int16)
        values[10:20, 10:20] = FREE
        values[40:50, 40:50] = FREE
        regions = detect_regions(values, RES, min_cells=8)
        self.assertEqual(len(regions), 2)

    def test_small_cluster_filtered(self):
        """只有 2 个格子的前沿应被过滤掉。"""
        values = np.full((20, 20), UNKNOWN, dtype=np.int16)
        values[10, 10:12] = FREE          # 极小空闲块 → 前沿很小
        regions = detect_regions(values, RES, min_cells=12)
        self.assertEqual(regions, [])

    def test_cluster_cells_belong_to_free_space(self):
        values = room_grid()
        regions = detect_regions(values, RES, min_cells=8)
        self.assertGreater(len(regions), 0)
        for region in regions:
            for r, c in region.cells:
                self.assertEqual(values[r, c], FREE)


# ══════════════════════════════════════════════════════════════════
# Viewpoint
# ══════════════════════════════════════════════════════════════════


class TestViewpoint(unittest.TestCase):
    def _prepare(self):
        values = room_grid()
        clearance = clearance_map(values, RES)
        geo = geodesic_map(values, (30, 30), RES)
        regions = detect_regions(values, RES, min_cells=8)
        self.assertGreater(len(regions), 0)
        return values, clearance, geo, regions

    def test_viewpoint_is_in_free_space(self):
        values, clearance, geo, regions = self._prepare()
        view = best_viewpoint(regions[0], values, RES, (0.0, 0.0),
                              geo, clearance, fast_config())
        self.assertIsNotNone(view)
        r, c = view.cell
        self.assertEqual(values[r, c], FREE)

    def test_viewpoint_is_reachable(self):
        values, clearance, geo, regions = self._prepare()
        view = best_viewpoint(regions[0], values, RES, (0.0, 0.0),
                              geo, clearance, fast_config())
        self.assertTrue(np.isfinite(view.path_length))
        self.assertGreaterEqual(view.path_length, 0.0)

    def test_viewpoint_respects_safety(self):
        values, clearance, geo, regions = self._prepare()
        cfg = fast_config()
        view = best_viewpoint(regions[0], values, RES, (0.0, 0.0),
                              geo, clearance, cfg)
        self.assertGreaterEqual(view.clearance, cfg.min_view_clearance)

    def test_viewpoint_has_information_gain(self):
        values, clearance, geo, regions = self._prepare()
        view = best_viewpoint(regions[0], values, RES, (0.0, 0.0),
                              geo, clearance, fast_config())
        self.assertGreater(view.gain_cells, 0)


# ══════════════════════════════════════════════════════════════════
# 打分
# ══════════════════════════════════════════════════════════════════


class TestScorer(unittest.TestCase):
    def _candidate(self, **kw) -> FrontierCandidate:
        base = dict(
            frontier_id="F1",
            centroid=(0.0, 0.0, 0.0),
            safe_viewpoint=(0.0, 0.0, 0.0),
            reachable=True,
            path_length=1.0,
            clearance=0.5,
            information_gain=10.0,
        )
        base.update(kw)
        return FrontierCandidate(**base)

    def test_novelty_decays_with_visits(self):
        self.assertEqual(novelty(0), 1.0)
        self.assertAlmostEqual(novelty(1), 0.5)
        self.assertLess(novelty(3), novelty(1))

    def test_history_penalty(self):
        self.assertEqual(history_penalty(0, 0), 0.0)
        self.assertAlmostEqual(history_penalty(1, 2), 0.5)
        self.assertEqual(history_penalty(3, 3), 1.0)

    def test_score_is_sorted_descending(self):
        a = self._candidate(frontier_id="A", information_gain=100.0, path_length=0.5)
        b = self._candidate(frontier_id="B", information_gain=1.0, path_length=5.0)
        result = score_candidates([a, b])
        self.assertEqual(result[0].frontier_id, "A")
        self.assertGreater(result[0].local_utility, result[1].local_utility)

    def test_failed_frontier_penalized(self):
        good = self._candidate(frontier_id="GOOD")
        bad = self._candidate(frontier_id="BAD", failed_count=5, visit_count=5)
        result = score_candidates([good, bad])
        self.assertEqual(result[0].frontier_id, "GOOD")

    def test_unreachable_is_heavily_penalized(self):
        ok = self._candidate(frontier_id="OK")
        bad = self._candidate(frontier_id="BAD", reachable=False)
        result = score_candidates([ok, bad])
        self.assertEqual(result[0].frontier_id, "OK")

    def test_utility_in_range(self):
        cands = [self._candidate(frontier_id=str(i), information_gain=i * 10.0)
                 for i in range(1, 6)]
        for c in score_candidates(cands):
            self.assertTrue(-1.0 <= c.local_utility <= 1.0)

    def test_empty_input(self):
        self.assertEqual(score_candidates([]), [])


# ══════════════════════════════════════════════════════════════════
# Explorer 端到端
# ══════════════════════════════════════════════════════════════════


class TestExplorer(unittest.TestCase):
    def test_explore_returns_frontier_candidates(self):
        values = room_grid()
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        result = explorer.explore(values, RES, (0.0, 0.0), (30, 30), map_version=7)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], FrontierCandidate)
        self.assertEqual(result[0].status, "NEW")

    def test_results_are_ranked(self):
        values = room_grid()
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        result = explorer.explore(values, RES, (0.0, 0.0), (30, 30), map_version=7)
        scores = [c.local_utility for c in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_snapshot_is_memory_compatible(self):
        """detect() 的输出必须能被 FrontierMemory.update_snapshot 读。"""
        values = room_grid()
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        snap = explorer.detect(values, RES, (0.0, 0.0), (30, 30), map_version=7)

        self.assertEqual(snap["map_version"], 7)
        self.assertTrue(snap["complete"])
        for item in snap["candidates"]:
            for key in ("local_candidate_id", "centroid_xy",
                        "safe_viewpoint_xy", "cluster_cells"):
                self.assertIn(key, item)
            self.assertEqual(len(item["centroid_xy"]), 2)
            self.assertGreater(len(item["cluster_cells"]), 0)

    def test_rank_uses_memory_history(self):
        """同一 frontier 带上失败历史后，分数必须下降（验证机制，不是验证排序结果）。"""
        values = two_rooms_grid()
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        snap = explorer.detect(values, RES, (0.0, 0.0), (15, 15), map_version=7)
        self.assertGreaterEqual(len(snap["candidates"]), 2)

        mapping = {c["local_candidate_id"]: f"F{i}"
                   for i, c in enumerate(snap["candidates"])}

        base = {c.frontier_id: c.local_utility
                for c in explorer.rank(snap, {}, mapping)}

        target_id = max(base, key=base.get)
        records = {target_id: {"visit_count": 10, "failed_count": 10, "status": "STALE"}}
        with_history = {c.frontier_id: c.local_utility
                        for c in explorer.rank(snap, records, mapping)}

        self.assertLess(with_history[target_id], base[target_id])

    def test_3d_coordinates_use_floor_y(self):
        values = room_grid()
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        result = explorer.explore(values, RES, (0.0, 0.0), (30, 30),
                                  map_version=1, floor_y=2.5)
        self.assertEqual(result[0].centroid[1], 2.5)
        self.assertEqual(result[0].safe_viewpoint[1], 2.5)

    def test_select_best(self):
        values = room_grid()
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        result = explorer.explore(values, RES, (0.0, 0.0), (30, 30), map_version=1)
        self.assertIs(explorer.select_best(result), result[0])
        self.assertIsNone(explorer.select_best([]))

    def test_no_frontier_returns_empty(self):
        values = np.zeros((40, 40), dtype=np.int16)   # 全是已知空闲
        explorer = FrontierExplorer(viewpoint_config=fast_config(), min_cells=8)
        self.assertEqual(explorer.explore(values, RES, (0.0, 0.0), (20, 20), 1), [])


class TestRaycastEquivalence(unittest.TestCase):
    """向量化射线扫描必须与逐格实现完全等价（性能优化不能改变语义）。"""

    @staticmethod
    def _reference(values, cell, resolution, sensor_range, num_rays=72):
        rows, cols = values.shape
        r0, c0 = cell
        max_steps = max(1, int(round(sensor_range / resolution)))
        step = resolution * 0.9
        seen = set()
        for k in range(num_rays):
            angle = 2.0 * math.pi * k / num_rays
            dx, dy = math.cos(angle), math.sin(angle)
            for i in range(1, max_steps + 1):
                r = int(round(r0 + i * step / resolution * dy))
                c = int(round(c0 + i * step / resolution * dx))
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                v = values[r, c]
                if v >= 50:
                    break
                if v == UNKNOWN:
                    seen.add((r, c))
        return seen

    def test_matches_reference(self):
        rng = np.random.default_rng(20261128)
        trials = 0
        for shape in ((30, 30), (60, 45)):
            for _ in range(6):
                values = rng.choice([UNKNOWN, FREE, OCCUPIED], size=shape,
                                    p=[0.45, 0.45, 0.10])
                for sensor_range in (2.0, 4.0):
                    cell = (int(rng.integers(0, shape[0])),
                            int(rng.integers(0, shape[1])))
                    if values[cell] >= 50:
                        continue
                    trials += 1
                    self.assertEqual(
                        self._reference(values, cell, RES, sensor_range),
                        visible_unknown_cells(values, cell, RES, sensor_range),
                        msg=f"shape={shape} cell={cell} range={sensor_range}")
        self.assertGreater(trials, 10)

    def test_batch_matches_single(self):
        rng = np.random.default_rng(7)
        values = rng.choice([UNKNOWN, FREE, OCCUPIED], size=(50, 50),
                            p=[0.45, 0.45, 0.10])
        cells = [(int(rng.integers(0, 50)), int(rng.integers(0, 50)))
                 for _ in range(40)]
        batch = visible_unknown_cell_counts(values, cells, RES, 4.0)
        single = [len(visible_unknown_cells(values, c, RES, 4.0)) for c in cells]
        self.assertEqual(batch, single)

    def test_empty_input(self):
        values = np.zeros((10, 10), dtype=np.int16)
        self.assertEqual(visible_unknown_cell_counts(values, [], RES, 4.0), [])

    def test_occlusion_blocks_far_cells(self):
        """一整堵墙后面的未知格不应被计入。"""
        values = np.full((30, 30), UNKNOWN, dtype=np.int16)
        values[:, 15] = OCCUPIED             # 贯穿全高的竖墙，射线绕不过去

        left = visible_unknown_cells(values, (15, 10), RES, 5.0)
        right = visible_unknown_cells(values, (15, 20), RES, 5.0)
        self.assertTrue(left)
        self.assertTrue(right)
        self.assertTrue(all(c < 15 for _, c in left))
        self.assertTrue(all(c > 15 for _, c in right))


if __name__ == "__main__":
    unittest.main(verbosity=2)
