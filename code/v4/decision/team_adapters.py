"""Adapters from the current lwj Memory and tls Exploration outputs to V4.

They use structural fields rather than importing either sibling module.  This
keeps Decision importable alone while making the merged ``hyf`` branch consume
the actual contracts emitted by ``vln_memory.OnlineMemory`` and
``exploration.FrontierExplorer``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

from v4.common.schemas import (
    FrontierCandidate,
    GoalSpec,
    PerceptionResult,
    PlannerContext,
    WorldObject,
    WorldState,
)

from .planner_context import build_planner_context


def frontier_candidates_from_tls(candidates: Iterable[Any]) -> tuple[FrontierCandidate, ...]:
    """Copy TLS explorer candidates into the V4 immutable planner contract."""
    result = []
    for candidate in candidates:
        result.append(FrontierCandidate(
            frontier_id=str(_field(candidate, "frontier_id")),
            centroid=_position(_field(candidate, "centroid")),
            safe_viewpoint=_position(_field(candidate, "safe_viewpoint")),
            cluster_size=int(_field(candidate, "cluster_size", 0)),
            reachable=bool(_field(candidate, "reachable", False)),
            path_length=float(_field(candidate, "path_length", 0.0)),
            clearance=float(_field(candidate, "clearance", 0.0)),
            information_gain=float(_field(candidate, "information_gain", 0.0)),
            novelty=float(_field(candidate, "novelty", 0.0)),
            nearby_object_ids=tuple(_field(candidate, "nearby_object_ids", ())),
            associated_place_id=_field(candidate, "associated_place_id", None),
            visual_cues=tuple(_field(candidate, "visual_cues", ())),
            visit_count=int(_field(candidate, "visit_count", 0)),
            failed_count=int(_field(candidate, "failed_count", 0)),
            status=str(_field(candidate, "status", "NEW")),
            local_utility=float(_field(candidate, "local_utility", 0.0)),
            semantic_utility=_optional_float(_field(candidate, "semantic_utility", None)),
        ))
    return tuple(result)


def world_state_from_online_memory(
    goal_spec: GoalSpec,
    online_update: Mapping[str, Any],
    *,
    robot_pose: Sequence[float],
    ranked_frontiers: Iterable[Any] = (),
    goal_matches: Optional[Mapping[str, str]] = None,
    reachable_object_ids: Iterable[str] = (),
    floor_y: float = 0.0,
    localization_valid: bool = True,
    sensor_healthy: bool = True,
) -> WorldState:
    """Build a conservative WorldState from ``OnlineMemory.apply`` output.

    ``OnlineMemory`` correctly owns association and evidence, while the planner
    needs normalized category/attributes and a task-local target status.
    Crucially, geometry alone does *not* imply safe navigation: callers must
    supply ``reachable_object_ids`` after a backend/path check before an object
    can become an APPROACH_TARGET candidate.
    """
    snapshot = online_update.get("object_memory", {})
    records = snapshot.get("objects", {})
    verifications = snapshot.get("verification", ())
    verified = {item.get("object_id") for item in verifications if item.get("status") == "VERIFIED"}
    rejected = {item.get("object_id") for item in verifications if item.get("status") == "REJECTED"}
    reachable = set(reachable_object_ids)
    matches = dict(goal_matches or {})
    objects = tuple(
        _world_object(identifier, record, identifier in verified, identifier in rejected,
                      matches.get(identifier), identifier in reachable, floor_y)
        for identifier, record in records.items()
    )
    frontiers = frontier_candidates_from_tls(ranked_frontiers)
    events = list(online_update.get("events", ()))
    exploration = online_update.get("exploration")
    if isinstance(exploration, Mapping):
        events.extend(exploration.get("events", ()))
    return WorldState(
        task_id=goal_spec.task_id,
        map_version=int(online_update.get("map_version", 0)),
        # ``OnlineMemory.apply`` stores the navigation frame as ``frame["xy"]``
        # and deliberately does not return a 3-D pose.  Do not manufacture an
        # origin here: the V4 loop must supply its current backend pose.
        robot_pose=_position(robot_pose),
        objects=objects,
        frontiers=frontiers,
        recent_events=tuple(events[-20:]),
        localization_valid=localization_valid,
        sensor_healthy=sensor_healthy,
    )


def planner_context_from_team_outputs(
    goal_spec: GoalSpec,
    online_update: Mapping[str, Any],
    ranked_frontiers: Iterable[Any],
    *,
    robot_pose: Sequence[float],
    perception: PerceptionResult | None = None,
    goal_matches: Optional[Mapping[str, str]] = None,
    reachable_object_ids: Iterable[str] = (),
    floor_y: float = 0.0,
    **context_kwargs: Any,
) -> PlannerContext:
    """One public entry point for the current lwj/tls → Decision boundary."""
    frontiers = frontier_candidates_from_tls(ranked_frontiers)
    state = world_state_from_online_memory(
        goal_spec, online_update, robot_pose=robot_pose, ranked_frontiers=frontiers, goal_matches=goal_matches,
        reachable_object_ids=reachable_object_ids, floor_y=floor_y,
    )
    return build_planner_context(
        goal_spec, state, frontiers, perception=perception, **context_kwargs,
    )


def _world_object(identifier: str, record: Mapping[str, Any], is_verified: bool,
                  is_rejected: bool, goal_match: str | None, is_reachable: bool,
                  floor_y: float) -> WorldObject:
    category, category_strength = _best_vote(record.get("category_scores", {}), "unknown")
    attributes = {
        key: _best_vote(values, "unknown")[0]
        for key, values in record.get("attribute_scores", {}).items()
        if _best_vote(values, "unknown")[0] != "unknown"
    }
    position_xz = record.get("map_position_xz")
    position = None if position_xz is None else (float(position_xz[0]), float(floor_y), float(position_xz[1]))
    if is_rejected or goal_match == "NON_MATCH":
        status = "REJECTED"
    elif is_verified and position is not None and is_reachable:
        status = "CONFIRMED"
    elif position is None:
        status = "UNCERTAIN"
    elif record.get("semantic_status") == "CONFLICTED" or goal_match == "POSSIBLE_MATCH":
        status = "UNCERTAIN"
    else:
        status = "CONFIRMATION"
    observation_ids = tuple(record.get("observation_ids", ()))
    return WorldObject(
        object_id=str(identifier), category=category, attributes=attributes,
        map_position=position, confidence=category_strength, status=status,
        observation_count=len(observation_ids),
        position_uncertainty=_optional_float(record.get("position_sigma_m")),
        reachable=is_reachable, evidence_ids=observation_ids,
    )


def _best_vote(scores: Mapping[str, Any], default: str) -> tuple[str, float]:
    if not scores:
        return default, 0.0
    key, value = max(scores.items(), key=lambda item: float(item[1]))
    total = sum(max(0.0, float(weight)) for weight in scores.values())
    # Evidence weights can grow over multiple views; keep confidence bounded.
    strength = min(1.0, max(0.0, float(value))) * (float(value) / total if total else 0.0)
    return str(key), strength


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _position(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"expected a 3-D position, got {value!r}")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
