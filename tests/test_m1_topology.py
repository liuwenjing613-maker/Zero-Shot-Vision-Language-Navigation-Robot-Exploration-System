import unittest

from vln_memory.topology_memory import TopologyMemory


class TopologyTest(unittest.TestCase):
    def setUp(self):
        self.m = TopologyMemory()
        self.a = self.m.visit("start", (0, 0), 0, "start", ["obs1"])
        self.b = self.m.visit("arrive", (1, 0), 3, "frontier_observed", ["obs2"])

    def test_places_revisit_and_replay(self):
        self.assertEqual(self.m.edges, {})
        for _ in range(2):
            result = self.m.visit("return", (0.3, 0), 7, "observation_place", ["obs1", "obs3"], ["obj1"])
            self.assertEqual(result, self.a)
        p = self.m.places[self.a]
        self.assertEqual((p.visit_count, p.observation_ids, p.nearby_object_ids), (2, ["obs1", "obs3"], ["obj1"]))
        with self.assertRaises(ValueError):
            self.m.visit("return", (2, 0), 7, "observation_place")
        self.assertEqual(len(self.m.places), 2)

    def test_directed_edge_lifecycle(self):
        nav = self.m.record_navigation
        e = nav("route", self.a, self.b, "planned", 1.2)
        self.assertEqual((e.status, e.traversal_count), ("PLANNED", 0))
        for event, expected in [("f1", "PLANNED"), ("f1", "PLANNED"), ("f2", "BLOCKED")]:
            self.assertEqual(nav(event, self.a, self.b, "failed").status, expected)
        self.assertEqual(nav("rep", self.a, self.b, "planned", 1.1).status, "BLOCKED")
        for _ in range(2):
            nav("ok", self.a, self.b, "succeeded", 1.3)
        self.assertEqual((e.status, e.traversal_count, e.failure_count, e.consecutive_failures),
                         ("TRAVERSED", 1, 2, 0))
        self.assertEqual(e.path_cost, 1.3)
        self.assertNotIn((self.b, self.a), self.m.edges)
        self.assertEqual(nav("block", self.a, self.b, "blocked").status, "BLOCKED")

    def test_invalid_evidence_leaves_graph_intact(self):
        for a, b, outcome, cost in [(self.a, self.a, "succeeded", 0), (self.a, "bogus", "planned", 1),
                                    (self.a, self.b, "planned", -1), (self.a, self.b, "failed", None)]:
            with self.assertRaises(ValueError):
                self.m.record_navigation("bad", a, b, outcome, cost)
        self.assertEqual(self.m.edges, {})

    def test_frontier_reassign_remove(self):
        self.m.link_frontier("F01", self.a)
        self.m.link_frontier("F01", self.b)
        self.assertEqual(self.m.frontier_places, {"F01": self.b})
        with self.assertRaises(ValueError):
            self.m.link_frontier("F01", "missing")
        self.m.link_frontier("F01", None)
        self.assertEqual(self.m.frontier_places, {})
