"""Synthetic scenario matrix for the V4 high-level decision policy.

These are deliberately data-only tests: they make the intended policy visible
without requiring Habitat, Qwen, an RGB camera, or a robot backend.
"""

from __future__ import annotations

import unittest

from v4.common.schemas import (
    ActionType,
    ExecutionResult,
    GoalSpec,
    WorldObject,
    WorldState,
)
from v4.decision import HighLevelPlanner, build_planner_context
from v4.common.schemas import FrontierCandidate


GOAL = GoalSpec(
    task_id="sim-red-chair",
    target_category="chair",
    target_attributes={"color": "red"},
    search_budget={"max_navigation_failures": 2, "max_exploration_steps": 3},
)


def frontier(identifier="F1", *, reachable=True, status="NEW"):
    return FrontierCandidate(
        frontier_id=identifier, centroid=(4.0, 0.0, 1.0), safe_viewpoint=(3.0, 0.0, 1.0),
        cluster_size=12, reachable=reachable, path_length=3.0, clearance=0.8,
        information_gain=10.0, novelty=0.8, status=status, local_utility=0.7,
    )


def obj(identifier="O1", *, category="chair", attributes=None, position=(2.0, 0.0, 1.0),
        confidence=0.9, status="CANDIDATE", reachable=True, sigma=0.2):
    return WorldObject(
        object_id=identifier, category=category,
        attributes={} if attributes is None else attributes,
        map_position=position, confidence=confidence, status=status,
        position_uncertainty=sigma, reachable=reachable, evidence_ids=("sim/obs-1",),
    )


def context(*, objects=(), frontiers=(), localization_valid=True, sensor_healthy=True,
            last_execution_result=None, navigation_failures=0, exploration_steps=0):
    state = WorldState(
        task_id=GOAL.task_id, map_version=4, robot_pose=(0.0, 0.0, 0.0),
        objects=objects, frontiers=frontiers, localization_valid=localization_valid,
        sensor_healthy=sensor_healthy,
    )
    return build_planner_context(
        GOAL, state, frontiers, last_execution_result=last_execution_result,
        navigation_failures=navigation_failures, exploration_steps=exploration_steps,
    )


class SimulatedDecisionScenarios(unittest.TestCase):
    def setUp(self):
        self.planner = HighLevelPlanner()

    def assert_action(self, expected, **kwargs):
        self.assertEqual(expected, self.planner.step(context(**kwargs)).action_type)

    def test_confirmed_reachable_match_is_approached(self):
        self.assert_action(
            ActionType.APPROACH_TARGET,
            objects=(obj(attributes={"color": "red"}, status="CONFIRMED"),),
            frontiers=(frontier(),),
        )

    def test_missing_target_attribute_is_observed_not_discarded(self):
        # A chair without a color observation may still be the red chair. It
        # must be checked before spending exploration budget elsewhere.
        self.assert_action(
            ActionType.OBSERVE_TARGET,
            objects=(obj(attributes={}, status="CONFIRMATION"),), frontiers=(frontier(),),
        )

    def test_known_conflicting_attribute_is_not_selected(self):
        self.assert_action(
            ActionType.EXPLORE_FRONTIER,
            objects=(obj(attributes={"color": "blue"}, status="CONFIRMATION"),),
            frontiers=(frontier(),),
        )

    def test_unlocalized_possible_match_requests_scan(self):
        self.assert_action(
            ActionType.SCAN,
            objects=(obj(attributes={"color": "red"}, position=None, status="UNCERTAIN"),),
            frontiers=(frontier(),),
        )

    def test_imprecisely_localized_confirmed_target_is_observed_again(self):
        # Semantics may be verified while geometry remains too uncertain for a
        # safe approach. Refining the observation is safer than abandoning the
        # candidate for an unrelated frontier.
        self.assert_action(
            ActionType.OBSERVE_TARGET,
            objects=(obj(attributes={"color": "red"}, status="CONFIRMED", sigma=2.0),),
            frontiers=(frontier(),),
        )

    def test_rejected_target_is_not_reintroduced_by_category_match(self):
        self.assert_action(
            ActionType.EXPLORE_FRONTIER,
            objects=(obj(attributes={"color": "red"}, status="REJECTED"),), frontiers=(frontier(),),
        )

    def test_stale_reachable_match_is_revisited(self):
        self.assert_action(
            ActionType.REVISIT_TARGET,
            objects=(obj(attributes={"color": "red"}, status="STALE"),), frontiers=(frontier(),),
        )

    def test_invalid_frontiers_are_skipped(self):
        self.assert_action(
            ActionType.EXPLORE_FRONTIER,
            frontiers=(frontier("blocked", status="BLOCKED"), frontier("F2")),
        )

    def test_sensor_fault_stops_navigation_before_any_target(self):
        self.assert_action(
            ActionType.RELOCALIZE,
            objects=(obj(attributes={"color": "red"}, status="CONFIRMED"),),
            frontiers=(frontier(),), sensor_healthy=False,
        )

    def test_no_path_becomes_recover(self):
        failure = ExecutionResult(
            task_id=GOAL.task_id, action_id="a1", status="FAILED", error_code="NO_PATH",
        )
        self.assert_action(ActionType.RECOVER, frontiers=(frontier(),), last_execution_result=failure)

    def test_verification_rejection_requests_scan(self):
        failure = ExecutionResult(
            task_id=GOAL.task_id, action_id="a1", status="FAILED", error_code="TARGET_REJECTED",
        )
        self.assert_action(ActionType.SCAN, frontiers=(frontier(),), last_execution_result=failure)

    def test_second_vlm_timeout_uses_local_fallback_without_retry(self):
        failure = ExecutionResult(
            task_id=GOAL.task_id, action_id="a1", status="TIMEOUT", error_code="VLM_TIMEOUT",
        )
        state = context(frontiers=(frontier(),), last_execution_result=failure).world_state
        state = WorldState(
            task_id=state.task_id, map_version=state.map_version, robot_pose=state.robot_pose,
            frontiers=state.frontiers, recent_events=({"event_type": "VLM_RETRY"},),
        )
        decision = self.planner.step(build_planner_context(GOAL, state, state.frontiers, last_execution_result=failure))
        self.assertEqual(ActionType.SCAN, decision.action_type)
        self.assertFalse(decision.recovery_directives["retry_vlm"])

    def test_budget_exhaustion_stops(self):
        self.assert_action(ActionType.STOP, frontiers=(frontier(),), exploration_steps=3)


if __name__ == "__main__":
    unittest.main()
