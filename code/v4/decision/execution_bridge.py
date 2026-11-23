"""Bridge V4 Decision outputs to the independently developed ``tls`` executor.

The bridge has no import from TLS source. The three branches can therefore be
developed separately and integrated only through their documented contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from v4.common.schemas import ActionDecision, ActionType, ExecutionResult


class TLSDispatchOperation(str, Enum):
    TASK_EXECUTOR = "TASK_EXECUTOR"
    BACKEND_SCAN = "BACKEND_SCAN"
    BACKEND_CANCEL = "BACKEND_CANCEL"
    EXTERNAL_REQUIRED = "EXTERNAL_REQUIRED"


@dataclass(frozen=True)
class TLSActionDecisionPayload:
    """Field-compatible payload for ``tls.common.schemas.ActionDecision``."""

    task_id: str
    decision_id: str
    action_type: str
    target_id: Optional[str] = None
    frontier_id: Optional[str] = None
    reason: str = ""
    map_version: int = 0


@dataclass(frozen=True)
class TLSDispatchPlan:
    operation: TLSDispatchOperation
    reason: str
    executor_payload: Optional[TLSActionDecisionPayload] = None
    directives: Mapping[str, Any] = field(default_factory=dict)


_TLS_EXECUTOR_ACTIONS = {
    ActionType.APPROACH_TARGET,
    ActionType.EXPLORE_FRONTIER,
    ActionType.STOP,
}


def to_tls_action_decision(decision: ActionDecision) -> TLSActionDecisionPayload:
    """Make the ID-only payload expected by TLS ``TaskExecutor``."""
    if decision.action_type not in _TLS_EXECUTOR_ACTIONS:
        raise ValueError(f"TLS TaskExecutor does not support {decision.action_type.value}")
    return TLSActionDecisionPayload(
        task_id=decision.task_id,
        decision_id=decision.decision_id,
        action_type=decision.action_type.value,
        target_id=decision.target_id,
        frontier_id=decision.frontier_id,
        reason=decision.reason,
        map_version=decision.map_version,
    )


def plan_tls_dispatch(decision: ActionDecision) -> TLSDispatchPlan:
    """Map every Decision action to TLS or its true owning module.

    The integration runner invokes ``TaskExecutor.execute(payload)`` for
    TASK_EXECUTOR, ``RobotBackend.scan()`` for BACKEND_SCAN, and
    ``RobotBackend.cancel_navigation()`` for BACKEND_CANCEL.  It must not
    misroute verification or relocalisation to navigation.
    """
    if decision.action_type in _TLS_EXECUTOR_ACTIONS:
        return TLSDispatchPlan(
            TLSDispatchOperation.TASK_EXECUTOR,
            "TLS TaskExecutor supports this ID-resolved navigation action",
            to_tls_action_decision(decision),
        )
    if decision.action_type in {ActionType.SCAN, ActionType.OBSERVE_TARGET, ActionType.REVISIT_TARGET}:
        return TLSDispatchPlan(
            TLSDispatchOperation.BACKEND_SCAN,
            "需要新的 RGB-D 观测；由 TLS RobotBackend.scan() 执行",
            directives={"target_id": decision.target_id, "next_stage": "PERCEPTION_AND_MEMORY"},
        )
    if decision.action_type is ActionType.RECOVER:
        return TLSDispatchPlan(
            TLSDispatchOperation.BACKEND_CANCEL,
            "恢复前取消当前导航；随后由 Memory 应用恢复指令并重新决策",
            directives=decision.recovery_directives,
        )
    if decision.action_type is ActionType.RELOCALIZE:
        return TLSDispatchPlan(
            TLSDispatchOperation.EXTERNAL_REQUIRED,
            "TLS 第一版没有定位重建器；必须由 Localization 模块处理",
            directives=decision.recovery_directives,
        )
    if decision.action_type is ActionType.VERIFY:
        return TLSDispatchPlan(
            TLSDispatchOperation.EXTERNAL_REQUIRED,
            "目标验证由 lwj VLM 的 verify_target() 负责，不应送入导航执行器",
        )
    raise ValueError(f"unhandled Decision action: {decision.action_type}")


def execution_result_from_tls(result: Any) -> ExecutionResult:
    """Convert TLS's string-status result to the V4 enum-backed contract."""
    final_pose = getattr(result, "final_pose", None)
    return ExecutionResult(
        task_id=str(result.task_id),
        action_id=str(result.action_id),
        status=str(result.status),
        final_pose=None if final_pose is None else tuple(float(value) for value in final_pose),
        path_length=float(getattr(result, "path_length", 0.0)),
        execution_time=float(getattr(result, "execution_time", 0.0)),
        error_code=getattr(result, "error_code", None),
        error_message=getattr(result, "error_message", None),
    )
