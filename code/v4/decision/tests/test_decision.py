import unittest
from types import SimpleNamespace

from v4.common.schemas import (
    ActionType, ExecutionResult, ExecutionStatus, FrontierCandidate, GoalSpec,
    PerceptionResult, WorldObject, WorldState,
)
from v4.decision import (
    HighLevelPlanner, TLSDispatchOperation, TaskState, TaskStateMachine,
    build_planner_context, execution_result_from_tls, goal_spec_from_lwj,
    perception_result_from_lwj, plan_tls_dispatch, to_tls_action_decision,
)


def frontier(identifier="frontier-1", *, status="NEW", reachable=True):
    return FrontierCandidate(
        identifier, (2.0, 0.0, 1.0), (1.8, 0.0, 1.0), 10, reachable, 2.0, 0.7,
        0.8, 0.9, status=status, local_utility=0.8,
    )


def goal():
    return GoalSpec("task-1", "chair", {"color": "red"})


def state(*, objects=(), frontiers=(), localization_valid=True, sensor_healthy=True):
    return WorldState("task-1", 5, (0.0, 0.0, 0.0), objects, frontiers,
                      localization_valid=localization_valid, sensor_healthy=sensor_healthy)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.planner = HighLevelPlanner()

    def test_confirmed_reachable_goal_has_priority_over_frontier(self):
        chair = WorldObject("obj-1", "chair", {"color": "red"}, (2, 0, 1), 0.92,
                            "CONFIRMED", 2, 0.1, True, ("obs-2",))
        decision = self.planner.step(build_planner_context(goal(), state(objects=(chair,), frontiers=(frontier(),)), (frontier(),)))
        self.assertEqual(decision.action_type, ActionType.APPROACH_TARGET)
        self.assertEqual(decision.target_id, "obj-1")
        self.assertIsNone(decision.frontier_id)

    def test_unconfirmed_goal_is_observed_not_navigated(self):
        chair = WorldObject("obj-1", "chair", {"color": "red"}, (2, 0, 1), 0.6,
                            "CANDIDATE", 1, 0.8, True, ("obs-2",))
        decision = self.planner.step(build_planner_context(goal(), state(objects=(chair,))))
        self.assertEqual(decision.action_type, ActionType.OBSERVE_TARGET)
        self.assertEqual(decision.target_id, "obj-1")

    def test_unlocalized_candidate_requests_scan(self):
        chair = WorldObject("obj-1", "chair", {"color": "red"}, None, 0.6, "CANDIDATE")
        decision = self.planner.step(build_planner_context(goal(), state(objects=(chair,))))
        self.assertEqual(decision.action_type, ActionType.SCAN)

    def test_exploration_skips_blocked_frontier(self):
        blocked, available = frontier("f-blocked", status="BLOCKED"), frontier("f-good")
        decision = self.planner.step(build_planner_context(goal(), state(frontiers=(blocked, available)), (blocked, available)))
        self.assertEqual(decision.action_type, ActionType.EXPLORE_FRONTIER)
        self.assertEqual(decision.frontier_id, "f-good")

    def test_no_frontier_stops_search(self):
        decision = self.planner.step(build_planner_context(goal(), state()))
        self.assertEqual(decision.action_type, ActionType.STOP)

    def test_sensor_fault_has_highest_normal_priority(self):
        chair = WorldObject("obj-1", "chair", {"color": "red"}, (2, 0, 1), 0.95, "CONFIRMED")
        decision = self.planner.step(build_planner_context(goal(), state(objects=(chair,), sensor_healthy=False)))
        self.assertEqual(decision.action_type, ActionType.RELOCALIZE)

    def test_no_path_failure_generates_local_recovery(self):
        result = ExecutionResult("task-1", "a-1", ExecutionStatus.FAILED, error_code="NO_PATH")
        decision = self.planner.step(build_planner_context(goal(), state(), last_execution_result=result))
        self.assertEqual(decision.action_type, ActionType.RECOVER)
        self.assertEqual(decision.recovery_directives["mark_frontier"], "BLOCKED")

    def test_budget_exhaustion_stops_before_exploration(self):
        limited_goal = GoalSpec("task-1", "chair", search_budget={"max_navigation_failures": 1, "max_exploration_steps": 3})
        decision = self.planner.step(build_planner_context(limited_goal, state(frontiers=(frontier(),)), (frontier(),), navigation_failures=1))
        self.assertEqual(decision.action_type, ActionType.STOP)

    def test_decision_freshness_requires_same_map_version(self):
        decision = self.planner.step(build_planner_context(goal(), state(frontiers=(frontier(),)), (frontier(),)))
        self.assertTrue(self.planner.is_fresh(decision, map_version=5))
        self.assertFalse(self.planner.is_fresh(decision, map_version=6))

    def test_lwj_adapters_preserve_schema_boundary(self):
        parsed = {"target_category": "chair", "target_attributes": {"color": "red"},
                  "spatial_relations": [], "completion_conditions": {"target_confirmed": True, "approach_required": True}}
        converted_goal = goal_spec_from_lwj(parsed, task_id="task-1")
        scene = {"task_id": "task-1", "observation_id": "obs-1", "target_status": "VISIBLE",
                 "target_entities": [{"local_id": "E01", "category": "chair"}],
                 "context_entities": [], "visual_cues": []}
        converted_scene = perception_result_from_lwj(scene)
        self.assertEqual(converted_goal.target_category, "chair")
        self.assertEqual(converted_scene.target_candidates[0]["local_id"], "E01")

    def test_state_machine_closed_loop(self):
        machine = TaskStateMachine()
        machine.start_task(); machine.task_parsed(); machine.initialized(); machine.observed(); machine.memory_updated()
        self.assertEqual(machine.apply_decision(ActionType.EXPLORE_FRONTIER), TaskState.EXPLORE)
        self.assertEqual(machine.execution_finished(ExecutionStatus.SUCCEEDED), TaskState.OBSERVE)

    def test_state_machine_verification_failure_replans(self):
        machine = TaskStateMachine()
        machine.start_task(); machine.task_parsed(); machine.initialized(); machine.observed(); machine.memory_updated()
        machine.apply_decision(ActionType.VERIFY)
        self.assertEqual(machine.verified(False), TaskState.REPLAN)

    def test_tls_bridge_converts_supported_navigation_payload(self):
        decision = self.planner.step(build_planner_context(goal(), state(frontiers=(frontier(),)), (frontier(),)))
        dispatch = plan_tls_dispatch(decision)
        self.assertEqual(dispatch.operation, TLSDispatchOperation.TASK_EXECUTOR)
        self.assertEqual(dispatch.executor_payload.action_type, "EXPLORE_FRONTIER")
        self.assertEqual(to_tls_action_decision(decision).frontier_id, "frontier-1")

    def test_tls_bridge_routes_scan_and_recovery_without_faking_navigation(self):
        candidate = WorldObject("obj-1", "chair", {"color": "red"}, None, 0.6, "CANDIDATE")
        scan = self.planner.step(build_planner_context(goal(), state(objects=(candidate,))))
        self.assertEqual(plan_tls_dispatch(scan).operation, TLSDispatchOperation.BACKEND_SCAN)
        failed = ExecutionResult("task-1", "a-1", "FAILED", error_code="STUCK")
        recover = self.planner.step(build_planner_context(goal(), state(), last_execution_result=failed))
        self.assertEqual(plan_tls_dispatch(recover).operation, TLSDispatchOperation.BACKEND_CANCEL)

    def test_tls_result_is_normalized_before_recovery(self):
        tls_result = SimpleNamespace(task_id="task-1", action_id="a-1", status="FAILED",
                                     final_pose=[1, 0, 2], path_length=1.2, execution_time=0.3,
                                     error_code="NO_PATH", error_message="blocked")
        result = execution_result_from_tls(tls_result)
        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertEqual(result.final_pose, (1.0, 0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
