"""Rules for failures that must never be delegated to a VLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from v4.common.schemas import ActionType, ExecutionResult, ExecutionStatus, PlannerContext


@dataclass(frozen=True)
class RecoveryPlan:
    action_type: ActionType
    reason: str
    directives: Mapping[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """Maps execution failures to safe local recovery actions and memory hints."""

    def plan(self, context: PlannerContext) -> RecoveryPlan | None:
        result = context.last_execution_result
        if result is None or result.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.RUNNING}:
            return None
        code = (result.error_code or "").upper()
        if code in {"LOCALIZATION_INVALID", "SENSOR_ERROR", "POSE_UNAVAILABLE"}:
            return RecoveryPlan(ActionType.RELOCALIZE, "定位或传感器异常，先重新定位", {
                "record_event": "RECOVERY_TRIGGERED", "cancel_navigation": True,
            })
        if code in {"NO_PATH", "UNREACHABLE"}:
            return RecoveryPlan(ActionType.RECOVER, "当前路径不可达，屏蔽失败前沿后重选", {
                "record_event": "NAVIGATION_FAILED", "mark_frontier": "BLOCKED",
                "exclude_last_frontier": True, "cancel_navigation": True,
            })
        if result.status is ExecutionStatus.TIMEOUT or code in {"NAVIGATION_TIMEOUT", "TIMEOUT"}:
            return RecoveryPlan(ActionType.RECOVER, "导航超时，取消当前导航并检查位姿", {
                "record_event": "NAVIGATION_FAILED", "cancel_navigation": True,
                "mark_frontier": "COOLDOWN", "require_reobservation": True,
            })
        if code in {"STUCK", "COLLISION"}:
            return RecoveryPlan(ActionType.RECOVER, "机器人受阻，停止并重新观察后规划", {
                "record_event": "RECOVERY_TRIGGERED", "cancel_navigation": True,
                "require_reobservation": True, "require_replan": True,
            })
        if code in {"TARGET_REJECTED", "VERIFICATION_REJECTED"}:
            return RecoveryPlan(ActionType.SCAN, "候选目标被否定，降级证据并重新观察", {
                "record_event": "TARGET_REJECTED", "downgrade_target": True,
            })
        if code in {"VLM_TIMEOUT", "PERCEPTION_TIMEOUT"}:
            retry = not any(event.get("event_type") == "VLM_RETRY" for event in context.world_state.recent_events)
            return RecoveryPlan(ActionType.SCAN, "视觉服务超时，执行一次受限重试" if retry else "视觉服务再次超时，使用本地规则继续", {
                "record_event": "VLM_RETRY" if retry else "VLM_TIMEOUT", "retry_vlm": retry,
            })
        return RecoveryPlan(ActionType.RECOVER, "执行失败，停止当前动作并重新规划", {
            "record_event": "RECOVERY_TRIGGERED", "cancel_navigation": True,
        })
