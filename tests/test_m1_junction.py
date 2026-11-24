import math
import unittest
import numpy as np
from vln_memory import JunctionDetector
from vln_memory.junction_memory import junctions, angle_gap
from vln_memory.geometry_memory import OccupancyMemory
from vln_memory.topology_memory import TopologyMemory


def layout(segments, res=0.05):
    side = math.ceil(10 / res)
    g = OccupancyMemory((side, side), (-5.0, -5.0), res)
    rr, cc = np.indices((side, side))
    pts = np.stack(((cc + 0.5) * res - 5, (rr + 0.5) * res - 5), axis=-1)
    g.log_odds[:] = 3
    for angle, length, width, value in segments:
        u = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        along = np.clip(pts @ u, 0, length)
        g.log_odds[
            np.linalg.norm(pts - along[:, :, None] * u, axis=2) <= width / 2
        ] = value
    return g


def arms(directions):
    return [(a, 3.8, 0.85, -3) for a in directions]


class JunctionCases(unittest.TestCase):
    def test_t_y_cross_rotation_and_scale(self):
        for headings in (
            (0, 90, 180),
            (35, 155, 275),
            (0, 90, 180, 270),
            (23, 113, 203, 293),
            (19, 109, 199),
        ):
            for res in (0.05, 0.1):
                with self.subTest(headings=headings, res=res):
                    candidates, _ = junctions(layout(arms(headings), res))
                    self.assertEqual(len(candidates), 1)
                    j = candidates[0]
                    self.assertLess(math.hypot(*j.xy), 0.4)
                    self.assertEqual(len(j.branches), len(headings))
                    for h in headings:
                        self.assertLess(
                            min(angle_gap(h, b.direction_deg) for b in j.branches), 18
                        )
                    self.assertTrue(
                        all(
                            b.length_m >= 1 and b.min_width_m >= 0.32
                            for b in j.branches
                        )
                    )

    def test_no_false_decisions(self):
        for mid in (
            [],
            [(90, 0.55, 0.70, -3)],
            [(90, 3.6, 0.25, -3)],
            [(90, 3.6, 0.85, 0)],
        ):
            self.assertEqual(junctions(layout(arms([0, 180]) + mid))[0], [])
        self.assertEqual(junctions(layout(arms([0, 90])))[0], [])
        for width in (2.0, 5.0):
            g = layout([])
            g.log_odds[70:130, 100 - int(width * 10) : 100 + int(width * 10)] = -3
            self.assertEqual(junctions(g)[0], [])

    def test_arrival_revisit_and_multitype(self):
        g = layout(arms([0, 90, 180]))
        graph = TopologyMemory()
        front = graph.visit("f", (0, 0), 0, "frontier_observed", area_id="A")
        det = JunctionDetector(graph)

        def step(i, xy, area="A"):
            g.map_version += 1
            return det.update(g, xy, i, str(i), area_id=area)

        self.assertIsNone(step(1, (2, 0)))
        self.assertTrue(det.candidates)
        self.assertIsNone(step(2, (0, 0)))
        self.assertIsNone(det.update(g, (0, 0), 3, "stale-scan", area_id="A"))
        self.assertEqual(step(4, (0, 0)), front)
        self.assertIsNone(step(5, (0, 0)))
        self.assertIsNone(step(6, (2, 0)))
        self.assertIsNone(step(7, (0, 0)))
        self.assertEqual(step(8, (0, 0)), front)
        p = graph.places[front]
        self.assertEqual(p.visit_count, 3)
        self.assertEqual(p.node_types, ["FRONTIER_OBSERVED", "JUNCTION"])
        self.assertEqual(len(p.branch_directions), 3)
        self.assertFalse(graph.edges)
        self.assertIsNone(step(9, (0, 0), "B"))
        self.assertNotEqual(step(10, (0, 0), "B"), front)

    def test_missing_or_duplicate_scans_and_obstacle_separated_node(self):
        g = layout(arms([0, 90, 180]))
        graph = TopologyMemory()
        neighbor = graph.visit("other", (0, -0.48), 0, "frontier_observed", area_id="A")
        d = JunctionDetector(graph)
        for t, obs in [(1, None), (2, "same"), (3, "same")]:
            g.map_version += 1
            self.assertIsNone(d.update(g, (0, 0), t, obs, area_id="A"))
        g.map_version += 1
        place = d.update(g, (0, 0), 4, "fresh", area_id="A")
        self.assertIsNotNone(place)
        self.assertNotEqual(place, neighbor)
        self.assertEqual(graph.places[neighbor].node_types, ["FRONTIER_OBSERVED"])

    def test_near_parallel_paths_do_not_count_as_three_distinct_directions(self):
        candidates, _ = junctions(layout(arms([180, -12, 12])))
        self.assertEqual(candidates, [])
