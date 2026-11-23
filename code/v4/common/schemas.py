"""Frozen, dependency-free contracts shared by V4 modules.

The contracts deliberately contain compact, serialisable data only.  A memory
implementation owns map arrays and full event logs; Decision receives a bounded
snapshot through :class:`PlannerContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

Position3D = Tuple[float, float, float]


class ActionType(str, Enum):
    SCAN = "SCAN"
    APPROACH_TARGET = "APPROACH_TARGET"
    OBSERVE_TARGET = "OBSERVE_TARGET"
    REVISIT_TARGET = "REVISIT_TARGET"
    EXPLORE_FRONTIER = "EXPLORE_FRONTIER"
    RELOCALIZE = "RELOCALIZE"
    RECOVER = "RECOVER"
    VERIFY = "VERIFY"
    STOP = "STOP"


class ExecutionStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class GoalSpec:
    task_id: str
    target_category: str
    target_attributes: Mapping[str, str] = field(default_factory=dict)
    spatial_relations: Sequence[Mapping[str, str]] = field(default_factory=tuple)
    completion_conditions: Mapping[str, bool] = field(default_factory=lambda: {
        "target_confirmed": True,
        "approach_required": True,
    })
    task_type: str = "object_navigation"
    search_budget: Mapping[str, int] = field(default_factory=lambda: {
        "max_navigation_failures": 5,
        "max_exploration_steps": 30,
    })

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.target_category.strip():
            raise ValueError("GoalSpec requires task_id and target_category")


@dataclass(frozen=True)
class PerceptionResult:
    task_id: str
    observation_id: str
    target_state: str
    target_candidates: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    scene_objects: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    visual_cues: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.target_state not in {"VISIBLE", "NOT_VISIBLE", "UNCERTAIN", "AMBIGUOUS"}:
            raise ValueError("invalid perception target_state")


@dataclass(frozen=True)
class WorldObject:
    """Decision-facing projection of one object-memory record."""

    object_id: str
    category: str
    attributes: Mapping[str, str] = field(default_factory=dict)
    map_position: Optional[Position3D] = None
    confidence: float = 0.0
    status: str = "CANDIDATE"
    observation_count: int = 1
    position_uncertainty: Optional[float] = None
    reachable: bool = True
    evidence_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.object_id or not self.category:
            raise ValueError("WorldObject requires object_id and category")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("object confidence must be in [0, 1]")


@dataclass(frozen=True)
class FrontierCandidate:
    """Shared Frontier Detection → Exploration → Decision contract."""

    frontier_id: str
    centroid: Position3D
    safe_viewpoint: Position3D
    cluster_size: int
    reachable: bool
    path_length: float
    clearance: float
    information_gain: float
    novelty: float
    nearby_object_ids: Sequence[str] = field(default_factory=tuple)
    associated_place_id: Optional[str] = None
    visual_cues: Sequence[str] = field(default_factory=tuple)
    visit_count: int = 0
    failed_count: int = 0
    status: str = "NEW"
    local_utility: float = 0.0
    semantic_utility: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.frontier_id:
            raise ValueError("FrontierCandidate requires frontier_id")


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    action_id: str
    status: ExecutionStatus | str
    final_pose: Optional[Position3D] = None
    path_length: float = 0.0
    execution_time: float = 0.0
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        status = ExecutionStatus(self.status)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True)
class WorldState:
    task_id: str
    map_version: int
    robot_pose: Position3D
    objects: Sequence[WorldObject] = field(default_factory=tuple)
    frontiers: Sequence[FrontierCandidate] = field(default_factory=tuple)
    recent_events: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    localization_valid: bool = True
    sensor_healthy: bool = True


@dataclass(frozen=True)
class PlannerContext:
    """Bounded snapshot consumed by Decision; no raw images or map arrays."""

    goal_spec: GoalSpec
    world_state: WorldState
    ranked_frontiers: Sequence[FrontierCandidate] = field(default_factory=tuple)
    perception: Optional[PerceptionResult] = None
    last_execution_result: Optional[ExecutionResult] = None
    navigation_failures: int = 0
    exploration_steps: int = 0


@dataclass(frozen=True)
class ActionDecision:
    task_id: str
    decision_id: str
    action_type: ActionType | str
    target_id: Optional[str] = None
    frontier_id: Optional[str] = None
    reason: str = ""
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    map_version: int = 0
    observation_id: Optional[str] = None
    recovery_directives: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = ActionType(self.action_type)
        object.__setattr__(self, "action_type", action)
        if bool(self.target_id) and bool(self.frontier_id):
            raise ValueError("an ActionDecision can target an object or frontier, not both")
