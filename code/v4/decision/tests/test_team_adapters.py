"""Contract tests for the current lwj/tls -> hyf Decision boundary."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from v4.common.schemas import ActionType, GoalSpec
from v4.decision import (
    HighLevelPlanner,
    frontier_candidates_from_tls,
    planner_context_from_team_outputs,
)


def _tls_frontier(**overrides):
    values = {
        "frontier_id": "F17",
        "centroid": (2.0, 0.0, 4.0),
        "safe_viewpoint": (1.5, 0.0, 3.5),
        "cluster_size": 24,
        "reachable": True,
        "path_length": 3.2,
        "clearance": 0.8,
        "information_gain": 18.0,
        "novelty": 0.7,
        "nearby_object_ids": ("O000001",),
        "associated_place_id": "P1",
        "visual_cues": ("doorway",),
        "visit_count": 0,
        "failed_count": 0,
        "status": "NEW",
        "local_utility": 0.91,
        "semantic_utility": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _online_update():
    return {
        "map_version": 8,
        "events": [{"reason": "JUNCTION", "place": "P1"}],
        "exploration": {"events": [{"reason": "FRONTIER_NEW", "frontier": "F17"}]},
        "object_memory": {
            "objects": {
                "O000001": {
                    "category_scores": {"chair": 0.90},
                    "attribute_scores": {"color": {"red": 0.85}},
                    "map_position_xz": (2.0, 3.0),
                    "position_sigma_m": 0.15,
                    "semantic_status": "SUPPORTED",
                    "observation_ids": ["obs-1/E1"],
                },
            },
            "verification": [{"object_id": "O000001", "status": "VERIFIED"}],
        },
    }


class TeamAdapterTests(unittest.TestCase):
    def setUp(self):
        self.goal = GoalSpec(
            task_id="task-1", target_category="chair", target_attributes={"color": "red"},
        )

    def test_verified_and_reachable_lwj_object_allows_approach(self):
        context = planner_context_from_team_outputs(
            self.goal, _online_update(), [_tls_frontier()], robot_pose=(0.0, 0.0, 0.0),
            goal_matches={"O000001": "MATCH"}, reachable_object_ids={"O000001"},
        )
        obj = context.world_state.objects[0]
        self.assertEqual((2.0, 0.0, 3.0), obj.map_position)
        self.assertEqual("CONFIRMED", obj.status)
        decision = HighLevelPlanner().step(context)
        self.assertEqual(ActionType.APPROACH_TARGET, decision.action_type)
        self.assertEqual("O000001", decision.target_id)

    def test_verified_but_unchecked_path_stays_observe_only(self):
        context = planner_context_from_team_outputs(
            self.goal, _online_update(), [_tls_frontier()], robot_pose=(0.0, 0.0, 0.0),
            goal_matches={"O000001": "MATCH"},
        )
        self.assertEqual("CONFIRMATION", context.world_state.objects[0].status)
        self.assertEqual(
            ActionType.OBSERVE_TARGET,
            HighLevelPlanner().step(context).action_type,
        )

    def test_tls_frontier_is_preserved_and_used_when_no_object_matches(self):
        frontiers = frontier_candidates_from_tls([_tls_frontier()])
        self.assertEqual("F17", frontiers[0].frontier_id)
        self.assertEqual(0.91, frontiers[0].local_utility)
        context = planner_context_from_team_outputs(
            self.goal,
            {"map_version": 9, "object_memory": {"objects": {}, "verification": []}},
            [_tls_frontier()],
            robot_pose=(0.0, 0.0, 0.0),
        )
        decision = HighLevelPlanner().step(context)
        self.assertEqual(ActionType.EXPLORE_FRONTIER, decision.action_type)
        self.assertEqual("F17", decision.frontier_id)


if __name__ == "__main__":
    unittest.main()
