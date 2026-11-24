"""Deterministic high-level action selection for the V4 closed loop."""

from __future__ import annotations

from itertools import count
from typing import Iterable

from v4.common.schemas import (
    ActionDecision,
    ActionType,
    FrontierCandidate,
    PlannerContext,
    WorldObject,
)

from .recovery_manager import RecoveryManager


class HighLevelPlanner:
    """Apply the frozen Decision priority order without direct VLM/backend calls.

    Preconditions such as path safety remain authoritative: a semantic object
    candidate is never selected unless Memory reports a world position and
    reachability.  Frontier ranking is owned by Semantic Exploration; this
    class only accepts or rejects its already-ranked candidates.
    """

    def __init__(self, *, confirmed_confidence: float = 0.75,
                 candidate_confidence: float = 0.35,
                 max_position_uncertainty: float = 1.5,
                 recovery_manager: RecoveryManager | None = None) -> None:
        self.confirmed_confidence = confirmed_confidence
        self.candidate_confidence = candidate_confidence
        self.max_position_uncertainty = max_position_uncertainty
        self.recovery_manager = recovery_manager or RecoveryManager()
        self._decision_numbers = count(1)

    def step(self, context: PlannerContext) -> ActionDecision:
        """Return the next high-level action according to the architecture spec."""
        self._validate_context(context)
        recovery = self.recovery_manager.plan(context)
        if recovery is not None:
            return self._decision(context, recovery.action_type, recovery.reason,
                                  recovery_directives=recovery.directives)

        state = context.world_state
        if not state.localization_valid or not state.sensor_healthy:
            return self._decision(context, ActionType.RELOCALIZE,
                                  "定位无效或传感器状态异常，禁止继续导航")
        if self._budget_exhausted(context):
            return self._decision(context, ActionType.STOP,
                                  "搜索预算已耗尽，停止任务并报告未找到目标")

        goal_objects = tuple(obj for obj in state.objects if self._matches_goal(obj, context))
        confirmed = self._best_confirmed(goal_objects)
        if confirmed is not None:
            return self._decision(
                context, ActionType.APPROACH_TARGET,
                "存在可达且已确认的目标，优先接近目标而非继续探索",
                target_id=confirmed.object_id, evidence_ids=confirmed.evidence_ids,
            )

        uncertain = self._best_uncertain(goal_objects)
        if uncertain is not None:
            action = ActionType.OBSERVE_TARGET if uncertain.map_position is not None else ActionType.SCAN
            reason = "存在目标候选但证据或几何位置不足，先观察/验证而不盲目导航"
            return self._decision(context, action, reason, target_id=uncertain.object_id,
                                  evidence_ids=uncertain.evidence_ids)

        historical = self._best_historical(goal_objects)
        if historical is not None:
            return self._decision(context, ActionType.REVISIT_TARGET,
                                  "存在可信度下降的历史目标，回访并重新观察",
                                  target_id=historical.object_id, evidence_ids=historical.evidence_ids)

        frontier = self._next_frontier(context.ranked_frontiers)
        if frontier is not None:
            return self._decision(context, ActionType.EXPLORE_FRONTIER,
                                  "未发现可靠目标，选择可达且有效的最高排名前沿",
                                  frontier_id=frontier.frontier_id)
        return self._decision(context, ActionType.STOP,
                              "没有有效前沿，搜索空间已耗尽")

    def is_fresh(self, decision: ActionDecision, *, map_version: int,
                 observation_id: str | None = None) -> bool:
        """Reject delayed decisions computed from an earlier map/observation."""
        return decision.map_version == map_version and (
            observation_id is None or decision.observation_id == observation_id
        )

    def _decision(self, context: PlannerContext, action: ActionType, reason: str,
                  *, target_id: str | None = None, frontier_id: str | None = None,
                  evidence_ids: Iterable[str] = (), recovery_directives=None) -> ActionDecision:
        observation_id = context.perception.observation_id if context.perception else None
        return ActionDecision(
            task_id=context.goal_spec.task_id,
            decision_id=f"{context.goal_spec.task_id}:decision:{next(self._decision_numbers):04d}",
            action_type=action,
            target_id=target_id,
            frontier_id=frontier_id,
            reason=reason,
            evidence_ids=tuple(evidence_ids),
            map_version=context.world_state.map_version,
            observation_id=observation_id,
            recovery_directives={} if recovery_directives is None else recovery_directives,
        )

    def _matches_goal(self, obj: WorldObject, context: PlannerContext) -> bool:
        # A missing attribute is unknown evidence, not negative evidence.  For
        # example, a detected chair whose color was occluded can still be a
        # requested red chair and should be observed before we spend the search
        # budget exploring elsewhere.  Only an explicit conflicting attribute
        # rules it out.  ``CONFIRMED`` remains protected by _best_confirmed's
        # status/reachability/geometry checks.
        if obj.status == "REJECTED":
            return False
        if obj.category.strip().casefold() != context.goal_spec.target_category.strip().casefold():
            return False
        return all(
            key not in obj.attributes
            or obj.attributes[key].strip().casefold() == value.strip().casefold()
            for key, value in context.goal_spec.target_attributes.items()
        )

    def _best_confirmed(self, objects: Iterable[WorldObject]) -> WorldObject | None:
        items = [obj for obj in objects if obj.status == "CONFIRMED" and obj.map_position is not None
                 and obj.reachable and obj.confidence >= self.confirmed_confidence
                 and self._position_is_precise(obj)]
        return max(items, key=lambda obj: obj.confidence, default=None)

    def _best_uncertain(self, objects: Iterable[WorldObject]) -> WorldObject | None:
        items = [obj for obj in objects if obj.status in {"CANDIDATE", "UNCERTAIN", "CONFIRMATION"}
                 and obj.confidence >= self.candidate_confidence]
        return max(items, key=lambda obj: obj.confidence, default=None)

    def _best_historical(self, objects: Iterable[WorldObject]) -> WorldObject | None:
        items = [obj for obj in objects if obj.status in {"OCCLUDED", "OUT_OF_VIEW", "STALE"}
                 and obj.map_position is not None and obj.reachable
                 and obj.confidence >= self.candidate_confidence]
        return max(items, key=lambda obj: obj.confidence, default=None)

    def _next_frontier(self, candidates: Iterable[FrontierCandidate]) -> FrontierCandidate | None:
        invalid = {"BLOCKED", "RESOLVED", "STALE", "COOLDOWN"}
        return next((candidate for candidate in candidates if candidate.reachable
                     and candidate.status not in invalid), None)

    def _position_is_precise(self, obj: WorldObject) -> bool:
        return obj.position_uncertainty is None or obj.position_uncertainty <= self.max_position_uncertainty

    @staticmethod
    def _budget_exhausted(context: PlannerContext) -> bool:
        budget = context.goal_spec.search_budget
        return (context.navigation_failures >= budget.get("max_navigation_failures", 5)
                or context.exploration_steps >= budget.get("max_exploration_steps", 30))

    @staticmethod
    def _validate_context(context: PlannerContext) -> None:
        if context.goal_spec.task_id != context.world_state.task_id:
            raise ValueError("PlannerContext contains mixed task IDs")
