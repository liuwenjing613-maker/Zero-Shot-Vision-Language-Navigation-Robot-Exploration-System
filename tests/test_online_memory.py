import unittest
import numpy as np
from vln_memory.online_memory import OnlineMemory
from vln_memory.topology_memory import TopologyMemory


class OnlineMemoryCases(unittest.TestCase):
    config = dict(
        shape=[120, 120],
        origin=[-4.8, -4.8],
        resolution=0.08,
        radius=0.1,
        camera=[32, 24, 80],
    )

    @staticmethod
    def sample(n, x, action, frontier=None):
        oid = f"run-{n}"
        frame = dict(
            session="run",
            step=n,
            xy=[x, 0],
            action=action,
            frontier=frontier,
            observation_id=oid,
        )
        angles = np.arange(720) * 2 * np.pi / 720
        with np.errstate(divide="ignore"):
            scan = np.minimum(
                np.where(
                    np.cos(angles) > 0,
                    (3 - x) / np.cos(angles),
                    (-3 - x) / np.cos(angles),
                ),
                3 / np.abs(np.sin(angles)),
            )
        data = dict(depth=np.full((24, 32), 3.0), ranges=scan, camera_pose=np.eye(4))
        result = dict(
            task_id="run",
            observation_id=oid,
            target_entities=[],
            context_entities=[dict(local_id="E01")],
            visual_cues=[],
        )
        return frame, data, result

    def test_ordered_replay_and_frontier_arrival(self):
        samples = [self.sample(0, 0, "start")]
        target = dict(id="F1", xy=[1.0, 0])
        samples.extend(
            self.sample(n, x, "forward", target)
            for n, x in enumerate((0.25, 0.5, 0.75, 1.0), 1)
        )
        samples.extend(self.sample(n, 1.0, "observe", target) for n in (5, 6))
        a = OnlineMemory(self.config)
        records = [a.apply(*s) for s in samples]
        self.assertEqual([r["events"] for r in records[1:5]], [[], [], [], []])
        self.assertEqual(records[5]["events"][0]["reason"], "FRONTIER_OBSERVED")
        self.assertEqual(records[6]["events"], [])
        self.assertEqual(records[-1]["edges"][0]["path_cost"], 1.0)
        node = records[-1]["places"][-1]
        self.assertIn("run-6/E01", node["nearby_object_ids"])
        self.assertEqual(node["visit_count"], 1)
        self.assertEqual(records[-1]["frontier_places"], {"F1": node["place_id"]})
        b = OnlineMemory(self.config)
        self.assertEqual([b.apply(*s) for s in samples], records)
        np.testing.assert_array_equal(a.grid.visit_count, b.grid.visit_count)

    def test_wrong_frame_and_out_of_order_no_mutation(self):
        m = OnlineMemory(self.config)
        frame, data, perception = self.sample(0, 0, "start")
        with self.assertRaises(ValueError):
            m.apply(frame, data, dict(perception, observation_id="old"))
        with self.assertRaises(ValueError):
            m.apply(frame, data, dict(perception, task_id="other"))
        with self.assertRaises(ValueError):
            m.apply(*self.sample(1, 0, "forward"))
        self.assertEqual(m.grid.map_version, 0)
        self.assertEqual(m.graph.places, {})
        m.apply(frame, data, perception)
        with self.assertRaises(ValueError):
            m.apply(frame, data, perception)

    def test_same_frame_adds_type_without_counting_two_visits(self):
        g = TopologyMemory()
        p = g.visit("a", (0, 0), 4, "new_region", ["obs"], area_id="A")
        q = g.visit(
            "b",
            (0, 0),
            4,
            "junction",
            ["obs"],
            area_id="A",
            branches=[(0, 2, 0.8), (90, 2, 0.8), (180, 2, 0.8)],
        )
        self.assertEqual(p, q)
        self.assertEqual(g.places[p].visit_count, 1)
        self.assertEqual(g.places[p].node_types, ["REGION_ENTRY", "JUNCTION"])
