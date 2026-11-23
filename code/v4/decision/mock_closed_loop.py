"""Deterministic three-branch mock closed-loop demonstration.

This file belongs to ``hyf`` but imports TLS lazily, so the Decision branch
stays independently testable before the team merges the execution module.
Run in an integration checkout with ``PYTHONPATH=code;src``:

    python -m v4.decision.mock_closed_loop
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from v4.common.schemas import FrontierCandidate, WorldObject, WorldState
from v4.decision.adapters import goal_spec_from_lwj, perception_result_from_lwj
from v4.decision.execution_bridge import (
    TLSDispatchOperation,
    execution_result_from_tls,
    plan_tls_dispatch,
)
from v4.decision.high_level_planner import HighLevelPlanner
from v4.decision.planner_context import build_planner_context


@dataclass(frozen=True)
class MockLoopEvent:
    stage: str
    action: str
    outcome: str
    detail: str


def run_mock_closed_loop() -> List[MockLoopEvent]:
    """Exercise lwj-shaped input → hyf Decision → actual TLS MockBackend.

    The VLM itself is intentionally not called.  Its output is deterministic
    mock JSON, while TLS execution uses its real in-repository MockBackend.
    """
    try:
        from execution import MockBackend, TaskExecutor
    except ImportError as error:
        raise RuntimeError(
            "TLS execution is required. Merge/copy code/execution and code/common, "
            "then run with PYTHONPATH=code;src."
        ) from error

    parsed_goal = {
        "target_category": "chair",
        "target_attributes": {"color": "red"},
        "spatial_relations": [{"relation": "near", "reference_object": "window"}],
        "completion_conditions": {"target_confirmed": True, "approach_required": True},
    }
    scene = {
        "task_id": "mock-red-chair",
        "observation_id": "obs-001",
        "target_status": "VISIBLE",
        "target_entities": [{
            "local_id": "E01", "category": "chair", "attributes": {"color": "red"},
            "bbox": [420, 350, 710, 820], "confidence": 0.62,
            "evidence": "A possible red chair is partly occluded.",
        }],
        "context_entities": [{
            "local_id": "E02", "category": "window", "attributes": {},
            "bbox": [10, 120, 280, 500], "confidence": 0.91, "evidence": "Large window on the left.",
        }],
        "visual_cues": [],
    }
    goal = goal_spec_from_lwj(parsed_goal, task_id="mock-red-chair")
    perception = perception_result_from_lwj(scene)
    frontier = FrontierCandidate(
        "frontier-001", (4.0, 0.0, 0.0), (4.0, 0.0, 0.0), 12, True, 4.0, 0.8,
        0.9, 0.9, local_utility=0.83,
    )
    backend = MockBackend(position=(0.0, 0.0, 0.0), reachable={(4.0, 0.0, 0.0), (2.0, 0.0, 1.0)})
    positions: Dict[str, tuple[float, float, float]] = {
        "frontier-001": (4.0, 0.0, 0.0), "obj-chair-001": (2.0, 0.0, 1.0),
    }
    executor = TaskExecutor(backend, lambda _kind, identifier: positions.get(identifier))
    planner = HighLevelPlanner()
    events: List[MockLoopEvent] = []

    # 1. Geometry has no valid depth for E01 yet: scan rather than navigate.
    unlocalized = WorldObject("obj-chair-001", "chair", {"color": "red"}, None, 0.62, "CANDIDATE")
    context = build_planner_context(
        goal, WorldState(goal.task_id, 1, (0.0, 0.0, 0.0), objects=(unlocalized,), frontiers=(frontier,)),
        (frontier,), perception=perception,
    )
    decision = planner.step(context)
    dispatch = plan_tls_dispatch(decision)
    assert dispatch.operation is TLSDispatchOperation.BACKEND_SCAN
    observations = backend.scan()
    events.append(MockLoopEvent("unlocalized candidate", decision.action_type.value, "SCANNED", str(len(observations))))

    # 2. No reliable target: use TLS TaskExecutor to explore the safe frontier.
    context = build_planner_context(
        goal, WorldState(goal.task_id, 2, (0.0, 0.0, 0.0), frontiers=(frontier,)),
        (frontier,), perception=perception,
    )
    decision = planner.step(context)
    dispatch = plan_tls_dispatch(decision)
    assert dispatch.operation is TLSDispatchOperation.TASK_EXECUTOR
    explore_result = execution_result_from_tls(executor.execute(dispatch.executor_payload))
    events.append(MockLoopEvent("target absent", decision.action_type.value, explore_result.status.value, decision.frontier_id or ""))

    # 3. Simulated Geometry + World Memory confirm a 3-D target; approach wins.
    confirmed = WorldObject(
        "obj-chair-001", "chair", {"color": "red"}, (2.0, 0.0, 1.0), 0.93,
        "CONFIRMED", 2, 0.15, True, ("obs-001:E01", "verify-001"),
    )
    context = build_planner_context(
        goal, WorldState(goal.task_id, 3, (4.0, 0.0, 0.0), objects=(confirmed,), frontiers=(frontier,)),
        (frontier,), perception=perception,
    )
    decision = planner.step(context)
    dispatch = plan_tls_dispatch(decision)
    approach_result = execution_result_from_tls(executor.execute(dispatch.executor_payload))
    events.append(MockLoopEvent("target confirmed", decision.action_type.value, approach_result.status.value, decision.target_id or ""))

    # 4. A deterministic NO_PATH failure must become cancel-and-replan recovery.
    backend.next_failure = "NO_PATH"
    failed_tls = executor.execute(dispatch.executor_payload)
    failed = execution_result_from_tls(failed_tls)
    recovery_context = build_planner_context(goal, context.world_state, (frontier,), last_execution_result=failed)
    recovery = planner.step(recovery_context)
    recovery_dispatch = plan_tls_dispatch(recovery)
    assert recovery_dispatch.operation is TLSDispatchOperation.BACKEND_CANCEL
    canceled = execution_result_from_tls(backend.cancel_navigation())
    events.append(MockLoopEvent("no path", recovery.action_type.value, canceled.status.value,
                                str(recovery.recovery_directives.get("mark_frontier"))))
    return events


def main() -> None:
    for event in run_mock_closed_loop():
        print(asdict(event))


if __name__ == "__main__":
    main()
