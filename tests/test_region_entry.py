import unittest
import numpy as np
from vln_memory.geometry_memory import Intrinsics, OccupancyMemory
from vln_memory.topology_memory import TopologyMemory
from vln_memory.portal_memory import RegionEntryDetector


class RegionEntryTest(unittest.TestCase):
    def setUp(self):
        self.g = OccupancyMemory((100, 100), (-2.525, -3.525), 0.05)
        self.g.log_odds[:] = -2
        K = self.K = Intrinsics(101, 81, 100, 100, 50, 40)
        self.T = np.eye(4)
        self.T[1, 3] = 1
        self.z = np.full((81, 101), 2.0)
        self.z[:, 26:75] = 4.0
        for x in (-0.5, 0.5):
            self.g.log_odds[self.g.cell((x, -2))] = 2
        self.graph = TopologyMemory()
        self.det = RegionEntryDetector(self.graph)
        self.det.start((0, 0), 0, "start")
        self.k = 0

    def see(self, kind="doorway", obs="v1"):
        return self.det.perceive(
            dict(
                observation_id=obs,
                visual_cues=[
                    dict(cue_type=kind, local_id="V", bbox=[250, 0, 750, 1000])
                ],
            ),
            self.z,
            self.K,
            self.T,
            self.g,
        )

    def walk(self, positions, observe=True):
        events = []
        for pt in positions:
            self.k += 1
            e = self.det.advance(pt, self.k, self.g, str(self.k) if observe else None)
            if e:
                events.append(e)
        return events

    def test_see_and_stop_then_cross_and_return(self):
        self.assertEqual(self.see()[0]["status"], "VALIDATED")
        self.assertEqual(self.see("opening", "v2")[0]["portal_id"], "D01")
        self.assertEqual(len(self.graph.places), 1)
        self.assertEqual(
            self.walk([(0, -d) for d in [0.5, 1, 1.5, 1.9, 2.1, 1.9, 2.1, 2.33]]), []
        )
        self.assertEqual(self.walk([(0, -2.50)], False), [])
        self.assertEqual(self.walk([(0, -2.50)])[0]["area"], "A02")
        p = self.det.portals["D01"]
        for expected, ds in [
            ("A01", [2.15, 1.9, 1.65, 1.45]),
            ("A02", [1.8, 2.1, 2.35, 2.55]),
        ]:
            self.assertEqual(self.walk([(0, -z) for z in ds])[0]["area"], expected)
        self.assertEqual(
            (len(self.det.portals), p.crossings, self.det.area_count), (1, 3, 2)
        )
        self.assertEqual(self.graph.places[p.places[-1]].area_id, p.areas[-1])
        self.assertEqual(self.graph.places[self.det.place].visit_count, 2)

    def test_missing_evidence_closed_and_unrelated_cues(self):
        self.assertEqual(self.see("corridor"), [])
        self.assertEqual(self.see("blocked_passage")[0]["status"], "BLOCKED")
        self.z[:] = 2.0
        self.assertEqual(self.see("doorway", "closed")[0]["status"], "PROPOSED")
        self.assertFalse(self.det.portals)

    def test_occupied_or_unknown_throat(self):
        for v in (0.0, 2.0):
            self.g.log_odds[self.g.cell((0, -2))] = v
            self.assertEqual(
                self.see(obs=str(v))[0]["status"], "PROPOSED" if v == 0 else "BLOCKED"
            )
        self.assertEqual(self.walk([(0, -v) for v in np.arange(0.2, 2.8, 0.2)]), [])

    def test_cross_extension_or_teleport(self):
        self.see()
        self.assertEqual(
            self.walk(
                [(0.4, 0), (0.8, 0)] + [(0.8, -v) for v in np.arange(0.2, 2.8, 0.2)]
            ),
            [],
        )
        self.assertEqual(self.walk([(0, -1.4), (0, -2.8), (0, -2.8)]), [])
        self.assertEqual(self.det.area, "A01")

    def test_no_crossing_on_parallel_motion(self):
        self.see()
        self.walk([(0, -0.4), (0, -0.8), (0, -1.2), (0, -1.6)])
        self.assertEqual(self.walk([(x, -1.6) for x in (0.2, 0.4, 0.6, 0.8)]), [])

    def test_close_then_reopen_same_portal(self):
        self.see()
        original = self.z.copy()
        self.z[:] = 2
        self.assertEqual(self.see("blocked_passage", "closed")[0]["portal_id"], "D01")
        self.assertEqual(self.det.portals["D01"].status, "BLOCKED")
        self.z[:] = original
        self.assertEqual(self.see("doorway", "open")[0]["portal_id"], "D01")
        self.assertEqual(self.det.portals["D01"].status, "VALIDATED")

    def test_verified_narrowing_and_constant_corridor(self):
        from vln_memory.portal_throat import passage_throat

        self.z[:] = 4
        self.z[:, 8:23] = 1.3
        self.z[:, 77:93] = 1.3
        cue = dict(cue_type="corridor", bbox=[0, 0, 1000, 1000])
        self.g.log_odds[:] = -2
        for x in (-0.45, 0.45):
            for z in np.arange(1.3, 2.05, 0.02):
                self.g.log_odds[self.g.cell((x, -z))] = 2.0
        self.assertIsNotNone(passage_throat(cue, self.z, self.K, self.T, self.g, 0.18))
        for x in (-0.45, 0.45):
            for z in np.arange(0, 1.31, 0.02):
                self.g.log_odds[self.g.cell((x, -z))] = 2.0
        self.assertIsNone(passage_throat(cue, self.z, self.K, self.T, self.g, 0.18))

    def test_recheck_from_new_safe_base_keeps_original_rgbd(self):
        self.g.log_odds[self.g.cell((0, 0))] = 2
        self.assertEqual(self.see()[0]["status"], "BLOCKED")
        self.g.map_version += 1
        cue = dict(
            observation_id="v1",
            visual_cues=[
                dict(cue_type="doorway", local_id="V", bbox=[250, 0, 750, 1000])
            ],
        )
        self.assertEqual(
            self.det.perceive(cue, self.z, self.K, self.T, self.g, base_xy=(0, -0.4))[
                0
            ]["status"],
            "VALIDATED",
        )
        self.assertEqual(len(self.det.portals), 1)
