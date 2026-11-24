import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from vln_memory.geometry_memory import OccupancyMemory
from vln_memory.frontier_memory import FrontierMemory
from vln_memory.episodic_memory import EpisodicMemory


def snap(version, items, complete=True):
    return dict(
        snapshot_id=str(version),
        map_version=version,
        complete=complete,
        candidates=[
            dict(
                local_candidate_id=str(i),
                centroid_xy=[x, 0],
                safe_viewpoint_xy=[x, 0],
                cluster_cells=[[2, 2 + i]],
            )
            for i, x in enumerate(items)
        ],
    )


def completion_trace():
    grid = OccupancyMemory((10, 10), (0, 0), 0.1)
    fm = FrontierMemory()
    key = fm.update_snapshot(snap(1, [0.1]))["0"]
    cells = {(r, c) for r in range(2, 6) for c in range(2, 7)}
    fm.region("R1", [key], cells, {"V1": (2, 2)})
    grid.map_version = 1
    for c in sorted(cells)[:8]:
        grid.observed_count[c] = 1
    result = []

    def check(label):
        r = fm.evaluate("R1", grid)
        result.append(
            dict(
                label=label,
                observed=r["observed"],
                eligible=r["eligible"],
                remaining=r["remaining"],
                checks=r["checks"],
                status=r["status"],
                view=r["views"]["V1"]["status"],
            )
        )

    check("partial scan")
    fm.transition(key, "selected", "d")
    fm.transition(key, "started", "a")
    fm.transition(key, "reached", "a")
    check("arrival")
    for c in sorted(cells)[:18]:
        grid.observed_count[c] = 1
    grid.last_observed_version[2, 2] = 1
    fm.scan("R1", "V1", grid, (0.25, 0.25), "obs")
    check("scan")
    fm.transition(key, "resolved", "detector", "boundary eliminated")
    grid.map_version = 2
    check("confirmation 1")
    check("repeated version")
    grid.map_version = 3
    check("confirmation 2")
    fm.update_snapshot(snap(4, [0.1]))
    grid.map_version = 4
    check("boundary reappears")
    return result


class ExplorationTests(unittest.TestCase):
    def test_completion(self):
        rows = completion_trace()
        self.assertEqual([r["checks"] for r in rows], [0, 0, 0, 1, 1, 2, 0])
        self.assertEqual(rows[5]["status"], "RESOLVED")
        self.assertEqual(rows[2]["remaining"], 1)
        self.assertEqual(rows[3]["observed"], 18)
        self.assertNotEqual(rows[6]["status"], "RESOLVED")

    def test_matching_failure_and_incomplete_snapshot(self):
        m = FrontierMemory()
        ids = m.update_snapshot(snap(1, [0.1, 0.18]))
        self.assertEqual(len(set(ids.values())), 2)
        self.assertEqual(ids, m.update_snapshot(snap(2, [0.11, 0.19])))
        m.update_snapshot(snap(3, [], False))
        self.assertEqual(m.records[ids["0"]]["missed_snapshots"], 0)
        for i, reason in enumerate(["TIMEOUT", "TIMEOUT", "NO_PATH", "NO_PATH"]):
            key = ids["0"]
            aid = str(i)
            m.transition(key, "selected", aid)
            m.transition(key, "started", aid)
            m.transition(key, "failed", aid, reason)
            m.transition(key, "failed", aid, reason)
        self.assertEqual(m.records[key]["failed_count"], 4)
        self.assertEqual(m.records[key]["status"], "BLOCKED")
        m.update_snapshot(snap(4, []))
        m.update_snapshot(snap(5, []))
        self.assertEqual(m.records[ids["1"]]["status"], "STALE")

    def test_visibility(self):
        grid = OccupancyMemory((8, 8), (0, 0), 1)
        grid.log_odds[2, 2] = 1
        grid.visit_count[:] = 1
        m = FrontierMemory()
        m.region("R", [], [(2, 1), (2, 3), (7, 7)], {"v": (2, 0)})
        r = m.evaluate("R", grid)
        self.assertEqual((r["observed"], r["eligible"]), (0, 1))

    def test_durable_idempotence(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            m = EpisodicMemory("s", p)
            m.record("a", "ACTION_STARTED", "execution", 1, action_id="x")
            m = EpisodicMemory("s", p)
            m.record("a", "ACTION_STARTED", "execution", 1, action_id="x")
            self.assertEqual(len(m.query(action_id="x")), 1)
            with self.assertRaises(ValueError):
                m.record("a", "ACTION_FAILED", "execution", 1)

class IntegrationTests(unittest.TestCase):
    def test_selected_target_scan_links_stable_id_and_place(self):
        from test_online_memory import OnlineMemoryCases
        from vln_memory.online_memory import OnlineMemory
        m=OnlineMemory(dict(OnlineMemoryCases.config,session="run"))
        target=dict(id="detector-local-id",xy=[0.,0.])
        for n,action in enumerate(["start","frontier","observe"]):
            frame,data,perception=OnlineMemoryCases.sample(n,0,action,target if n else None)
            rc=m.grid.cell([0,0])
            frame["frontier_snapshot"]=dict(snapshot_id=str(n),map_version=n+1,complete=True,candidates=[
                dict(local_candidate_id=str(n),cluster_cells=[list(rc)],centroid_xy=[0.,0.],safe_viewpoint_xy=[0.,0.])])
            state=m.apply(frame,data,perception)
        fm=m.exploration.m3
        self.assertEqual(fm.records["F0001"]["visit_count"],1)
        self.assertIn("F0001",state["frontier_places"])
        self.assertEqual(fm.regions["R0001"]["views"]["V1"]["observations"],["run-2"])
        self.assertEqual(len(m.exploration.m4.query(event_type="FRONTIER_REACHED",frontier_id="F0001")),1)
