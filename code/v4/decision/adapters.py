"""Explicit adapters for the public ``lwj`` perception package schemas."""

from __future__ import annotations

from typing import Any, Mapping

from v4.common.schemas import GoalSpec, PerceptionResult


def goal_spec_from_lwj(parsed_goal: Mapping[str, Any], *, task_id: str,
                       search_budget: Mapping[str, int] | None = None) -> GoalSpec:
    """Convert ``vln_perception.VLMClient.parse_goal`` output to V4 GoalSpec."""
    required = {"target_category", "target_attributes", "spatial_relations", "completion_conditions"}
    missing = required.difference(parsed_goal)
    if missing:
        raise ValueError(f"lwj GoalSpec missing fields: {sorted(missing)}")
    return GoalSpec(
        task_id=task_id,
        target_category=str(parsed_goal["target_category"]),
        target_attributes=dict(parsed_goal["target_attributes"]),
        spatial_relations=tuple(parsed_goal["spatial_relations"]),
        completion_conditions=dict(parsed_goal["completion_conditions"]),
        search_budget=dict(search_budget or {"max_navigation_failures": 5, "max_exploration_steps": 30}),
    )


def perception_result_from_lwj(scene_result: Mapping[str, Any], *, task_id: str | None = None) -> PerceptionResult:
    """Convert ``perceive_scene`` output without claiming its local IDs are global.

    The resulting candidates still need geometry fusion and World Memory
    association before Decision can navigate to them.
    """
    required = {"task_id", "observation_id", "target_status", "target_entities", "context_entities", "visual_cues"}
    missing = required.difference(scene_result)
    if missing:
        raise ValueError(f"lwj PerceptionResult missing fields: {sorted(missing)}")
    source_task = str(scene_result["task_id"])
    if task_id is not None and source_task != task_id:
        raise ValueError("lwj scene result task_id does not match the active task")
    return PerceptionResult(
        task_id=source_task,
        observation_id=str(scene_result["observation_id"]),
        target_state=str(scene_result["target_status"]),
        target_candidates=tuple(scene_result["target_entities"]),
        scene_objects=tuple(scene_result["context_entities"]),
        visual_cues=tuple(scene_result["visual_cues"]),
    )
