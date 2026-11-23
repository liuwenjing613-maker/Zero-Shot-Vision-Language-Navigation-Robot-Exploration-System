"""Build a bounded Decision input from World Memory and exploration output."""

from __future__ import annotations

from typing import Iterable, Optional

from v4.common.schemas import (
    ExecutionResult,
    FrontierCandidate,
    GoalSpec,
    PerceptionResult,
    PlannerContext,
    WorldState,
)


def build_planner_context(
    goal_spec: GoalSpec,
    world_state: WorldState,
    ranked_frontiers: Iterable[FrontierCandidate] = (),
    *,
    perception: Optional[PerceptionResult] = None,
    last_execution_result: Optional[ExecutionResult] = None,
    navigation_failures: int = 0,
    exploration_steps: int = 0,
    max_frontiers: int = 8,
    max_events: int = 20,
) -> PlannerContext:
    """Limit context size so Decision stays deterministic and prompt-ready."""
    if goal_spec.task_id != world_state.task_id:
        raise ValueError("GoalSpec and WorldState must belong to the same task")
    if perception is not None and perception.task_id != goal_spec.task_id:
        raise ValueError("PerceptionResult belongs to another task")
    if last_execution_result is not None and last_execution_result.task_id != goal_spec.task_id:
        raise ValueError("ExecutionResult belongs to another task")
    ranked = tuple(ranked_frontiers)[:max_frontiers]
    bounded_state = WorldState(
        task_id=world_state.task_id,
        map_version=world_state.map_version,
        robot_pose=world_state.robot_pose,
        objects=tuple(world_state.objects),
        frontiers=tuple(world_state.frontiers),
        recent_events=tuple(world_state.recent_events[-max_events:]),
        localization_valid=world_state.localization_valid,
        sensor_healthy=world_state.sensor_healthy,
    )
    return PlannerContext(
        goal_spec=goal_spec,
        world_state=bounded_state,
        ranked_frontiers=ranked,
        perception=perception,
        last_execution_result=last_execution_result,
        navigation_failures=navigation_failures,
        exploration_steps=exploration_steps,
    )
