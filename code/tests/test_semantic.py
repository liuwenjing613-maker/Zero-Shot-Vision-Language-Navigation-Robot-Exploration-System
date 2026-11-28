"""语义探索（S0：知识库 + 对象先验）的单元测试。

覆盖：
  · 知识库加载与归一化
  · 物体关联 A(o,g) 的房间路径 / 共现路径
  · 别名字典与未知类别
  · 可靠性映射
  · ObjectPriorProvider 的打分、sigma 影响、历史衰减
  · 通过 FrontierExplorer.rank 注入语义后排序能否翻转
  · 不接线时行为与纯几何完全一致
"""

from __future__ import annotations

import math
import unittest

from exploration import FrontierExplorer, UtilityWeights
from exploration.demo_exploration import build_scene
from exploration.knowledge import KnowledgeBase, normalize_category
from exploration.scorer import score_candidates
from common.schemas import FrontierCandidate

from exploration.semantic import (
    CandidateContext,
    CompositeProvider,
    MemoryObject,
    NullProvider,
    ObjectPriorConfig,
    ObjectPriorProvider,
    euclidean_object_query,
    reliability_from_status,
)

RES = 0.1
ROBOT = (55, 30)


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(normalize_category("TV Monitor"), "tv_monitor")
        self.assertEqual(normalize_category("  Dining-Table "), "dining_table")
        self.assertEqual(normalize_category("a//b"), "a_b")

    def test_empty(self):
        self.assertEqual(normalize_category(""), "")
        self.assertEqual(normalize_category(None), "")


class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.load()

    def test_loaded(self):
        self.assertGreater(len(self.kb.room_prior), 30)
        self.assertGreater(len(self.kb.cooccurrence), 20)

    def test_room_distribution_normalized(self):
        for obj, dist in self.kb.room_prior.items():
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6, msg=obj)
            self.assertTrue(all(v > 0 for v in dist.values()), obj)

    def test_same_category_is_one(self):
        self.assertEqual(self.kb.affinity("chair", "chair"), 1.0)
        self.assertEqual(self.kb.affinity("Chair", "chair"), 1.0)

    def test_room_path(self):
        # 马桶与洗手池词义不近，但同属卫生间，房间路径应给出强关联
        self.assertGreater(self.kb.affinity("toilet", "sink"), 0.4)
        self.assertLess(self.kb.affinity("toilet", "bed"), 0.1)

    def test_cooccurrence_path(self):
        # 遥控器与电视：房间路径已经较高，共现路径更高
        self.assertGreater(self.kb.affinity("remote_control", "tv_monitor"), 0.8)
        self.assertGreater(self.kb.affinity("chair", "dining_table"), 0.8)

    def test_symmetry(self):
        self.assertAlmostEqual(self.kb.affinity("toilet", "sink"),
                               self.kb.affinity("sink", "toilet"), places=9)

    def test_alias(self):
        self.assertEqual(self.kb.canonical("Television"), "tv_monitor")
        self.assertAlmostEqual(self.kb.affinity("tv", "remote_control"), 0.95, places=6)

    def test_unknown_category(self):
        self.assertEqual(self.kb.affinity("spaceship", "chair"), 0.0)
        self.assertEqual(self.kb.affinity("chair", "spaceship"), 0.0)

    def test_empty_kb(self):
        kb = KnowledgeBase.empty()
        self.assertEqual(kb.affinity("chair", "chair"), 1.0)   # 同类仍为 1
        self.assertEqual(kb.affinity("chair", "table"), 0.0)


class TestReliability(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(reliability_from_status("CONFIRMED"), 1.0)
        self.assertEqual(reliability_from_status("CANDIDATE"), 0.7)
        self.assertEqual(reliability_from_status("CANDIDATE", has_position=False), 0.25)
        self.assertEqual(reliability_from_status("REJECTED"), 0.0)
        self.assertEqual(reliability_from_status("confirmed"), 1.0)


def _ctx(fid, vp, visit=0):
    return CandidateContext(frontier_id=fid, viewpoint_xy=vp, centroid_xy=vp,
                            visit_count=visit)


class TestObjectPriorProvider(unittest.TestCase):
    def setUp(self):
        self.provider = ObjectPriorProvider(config=ObjectPriorConfig(sigma=4.0))

    def test_no_objects_gives_zero(self):
        ctxs = [_ctx("F1", (0.0, 0.0)), _ctx("F2", (3.0, 0.0))]
        scores = self.provider.score("chair", ctxs, euclidean_object_query([]))
        self.assertEqual(scores, {"F1": 0.0, "F2": 0.0})

    def test_object_boosts_nearby_frontier(self):
        obj = MemoryObject("O1", "chair", (2.9, 0.0), reliability=1.0)
        ctxs = [_ctx("F1", (0.0, 0.0)), _ctx("F2", (3.0, 0.0))]
        scores = self.provider.score("chair", ctxs, euclidean_object_query([obj]))
        self.assertGreater(scores["F2"], scores["F1"])
        self.assertAlmostEqual(scores["F2"], math.exp(-0.1 / 4.0), places=6)

    def test_unrelated_object_ignored(self):
        obj = MemoryObject("O1", "bed", (3.0, 0.0), reliability=1.0)
        ctxs = [_ctx("F2", (3.0, 0.0))]
        scores = self.provider.score("chair", ctxs, euclidean_object_query([obj]))
        self.assertEqual(scores["F2"], 0.0)

    def test_reliability_scales_score(self):
        strong = MemoryObject("O1", "chair", (0.0, 0.0), reliability=1.0)
        weak = MemoryObject("O1", "chair", (0.0, 0.0), reliability=0.25)
        ctxs = [_ctx("F1", (0.0, 0.0))]
        s1 = self.provider.score("chair", ctxs, euclidean_object_query([strong]))["F1"]
        s2 = self.provider.score("chair", ctxs, euclidean_object_query([weak]))["F1"]
        self.assertAlmostEqual(s1, 1.0, places=6)
        self.assertAlmostEqual(s2, 0.25, places=6)

    def test_sigma_controls_reach(self):
        obj = MemoryObject("O1", "chair", (0.0, 0.0), reliability=1.0)
        ctxs = [_ctx("F1", (6.0, 0.0))]
        near = ObjectPriorProvider(config=ObjectPriorConfig(sigma=2.0)).score(
            "chair", ctxs, euclidean_object_query([obj]))["F1"]
        far = ObjectPriorProvider(config=ObjectPriorConfig(sigma=8.0)).score(
            "chair", ctxs, euclidean_object_query([obj]))["F1"]
        self.assertGreater(far, near)
        self.assertLess(near, 0.1)

    def test_history_decay(self):
        obj = MemoryObject("O1", "chair", (0.0, 0.0), reliability=1.0)
        q = euclidean_object_query([obj])
        fresh = self.provider.score("chair", [_ctx("F1", (0.0, 0.0))], q)["F1"]
        visited = self.provider.score("chair", [_ctx("F1", (0.0, 0.0), visit=1)], q)["F1"]
        self.assertLess(visited, fresh)
        self.assertAlmostEqual(visited, fresh * 0.4, places=6)

    def test_radius_filter(self):
        obj = MemoryObject("O1", "chair", (50.0, 0.0), reliability=1.0)
        ctxs = [_ctx("F1", (0.0, 0.0))]
        scores = self.provider.score("chair", ctxs, euclidean_object_query([obj]))
        self.assertEqual(scores["F1"], 0.0)

    def test_null_provider(self):
        ctxs = [_ctx("F1", (0.0, 0.0))]
        obj = MemoryObject("O1", "chair", (0.0, 0.0), reliability=1.0)
        scores = NullProvider().score("chair", ctxs, euclidean_object_query([obj]))
        self.assertEqual(scores["F1"], 0.0)

    def test_composite_takes_max(self):
        obj = MemoryObject("O1", "chair", (0.0, 0.0), reliability=1.0)
        ctxs = [_ctx("F1", (0.0, 0.0))]
        comp = CompositeProvider([NullProvider(), self.provider])
        self.assertAlmostEqual(comp.score("chair", ctxs, euclidean_object_query([obj]))["F1"],
                               1.0, places=6)


class TestScorerSemanticTerm(unittest.TestCase):
    def _cand(self, fid, semantic):
        return FrontierCandidate(
            frontier_id=fid,
            centroid=(0.0, 0.0, 0.0),
            safe_viewpoint=(0.0, 0.0, 0.0),
            reachable=True,
            path_length=1.0,
            clearance=0.5,
            information_gain=1000.0,
            semantic_utility=semantic,
        )

    def test_semantic_can_flip_ranking(self):
        a = self._cand("A", 0.0)
        b = self._cand("B", 1.0)
        built = [a, b]
        for c in built:
            c.path_length = 1.0
        ranked = score_candidates(built, UtilityWeights())
        self.assertEqual(ranked[0].frontier_id, "B")

    def test_semantic_none_behaves_as_zero(self):
        a = self._cand("A", None)
        b = self._cand("B", 0.0)
        ranked = score_candidates([a, b], UtilityWeights())
        self.assertEqual(len(ranked), 2)
        for c, w in zip(sorted(ranked, key=lambda x: x.frontier_id),
                        [None, None]):
            pass
        self.assertAlmostEqual(ranked[0].local_utility, ranked[1].local_utility, places=6)

    def test_semantic_clipped(self):
        a = self._cand("A", 5.0)      # 越界值应被截断到 1
        ranked = score_candidates([a], UtilityWeights())
        self.assertLessEqual(ranked[0].local_utility, 1.0)


class TestExplorerIntegration(unittest.TestCase):
    """真实的 detect → rank 链路：两个几何上势均力敌的前沿。"""

    @classmethod
    def setUpClass(cls):
        cls.values = build_scene()
        cls.explorer = FrontierExplorer()
        cls.snapshot = cls.explorer.detect(cls.values, RES, (0.0, 0.0), ROBOT, 1)

    def test_no_semantic_keeps_geometric_order(self):
        ranked = self.explorer.rank(self.snapshot)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].frontier_id, "F1")     # 几何上 F1 更优
        for c in ranked:
            self.assertIsNone(c.semantic_utility)

    def test_goal_alone_without_provider_does_nothing(self):
        ranked = self.explorer.rank(self.snapshot, goal="chair")
        for c in ranked:
            self.assertIsNone(c.semantic_utility)

    def test_object_near_weaker_frontier_flips_ranking(self):
        base = self.explorer.rank(self.snapshot)
        f2 = next(c for c in base if c.frontier_id == "F2")
        obj = MemoryObject(
            object_id="O000001",
            category="chair",
            xy=(f2.safe_viewpoint[0], f2.safe_viewpoint[2]),
            reliability=1.0,
            status="CONFIRMED",
        )
        ranked = self.explorer.rank(
            self.snapshot,
            goal="chair",
            semantic=ObjectPriorProvider(),
            object_query=euclidean_object_query([obj]),
        )
        self.assertEqual(ranked[0].frontier_id, "F2")
        self.assertGreater(ranked[0].semantic_utility, 0.9)

    def test_goal_object_with_category_attribute(self):
        """GoalSpec 之类带 .category 的对象也能直接用。"""
        class FakeGoal:
            category = "chair"

        base = self.explorer.rank(self.snapshot)
        f2 = next(c for c in base if c.frontier_id == "F2")
        obj = MemoryObject("O1", "chair", (f2.safe_viewpoint[0], f2.safe_viewpoint[2]),
                           reliability=1.0)
        ranked = self.explorer.rank(
            self.snapshot, goal=FakeGoal(),
            semantic=ObjectPriorProvider(),
            object_query=euclidean_object_query([obj]))
        self.assertEqual(ranked[0].frontier_id, "F2")

    def test_unrelated_goal_does_not_change_order(self):
        base = self.explorer.rank(self.snapshot)
        f2 = next(c for c in base if c.frontier_id == "F2")
        obj = MemoryObject("O1", "bed", (f2.safe_viewpoint[0], f2.safe_viewpoint[2]),
                           reliability=1.0)
        ranked = self.explorer.rank(
            self.snapshot, goal="chair",
            semantic=ObjectPriorProvider(),
            object_query=euclidean_object_query([obj]))
        self.assertEqual(ranked[0].frontier_id, "F1")


if __name__ == "__main__":
    unittest.main()
