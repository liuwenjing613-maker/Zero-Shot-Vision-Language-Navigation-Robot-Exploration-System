"""TaskExecutor —— 把 ActionDecision 翻译成 RobotBackend 调用。

职责边界：
  · **只负责"执行"**，不负责决策、不负责记忆
  · ActionDecision 里只有 ID，坐标通过注入的 pose_resolver 解析
    （决策见 docs/模块五_执行_方案.md Q3）
  · 任何失败都返回 ExecutionResult，不向调用方抛异常
    （调用方是状态机，要靠 error_code 进入恢复流程）

第一版支持的动作：
    EXPLORE_FRONTIER  → navigate_to(frontier_pose)
    APPROACH_TARGET   → navigate_to(target_pose)
    STOP              → stop()
其余返回 FAILED + UNSUPPORTED_ACTION。
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional, Sequence

from common.enums import ActionType, EntityKind, ErrorCode
from common.schemas import ActionDecision, ExecutionResult
from .robot_backend import RobotBackend

# pose_resolver(kind, entity_id) -> (x, y, z) 或 None
PoseResolver = Callable[[str, str], Optional[Sequence[float]]]


class TaskExecutor:
    def __init__(self, backend: RobotBackend, pose_resolver: Optional[PoseResolver] = None):
        """
        Args:
            backend: 执行后端（HabitatBackend / MockBackend / M1Backend）
            pose_resolver: 把 target_id / frontier_id 解析成世界坐标。
                           第一版可以传一个查字典的 lambda；
                           正式版由 Memory 模块提供。
        """
        self.backend = backend
        self.pose_resolver = pose_resolver

    # ══════════════════════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════════════════════

    def execute(self, decision: ActionDecision) -> ExecutionResult:
        """执行一个决策，返回结构化结果。"""
        start = perf_counter()
        try:
            result = self._dispatch(decision)
        except Exception as exc:  # 后端实现有 bug 也不能拖垮主循环
            result = ExecutionResult.failed(
                decision.task_id,
                decision.decision_id,
                ErrorCode.INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}",
            )

        # 统一补齐标识与耗时（后端方法不感知 task/action 语义）
        result.task_id = decision.task_id
        result.action_id = decision.decision_id
        result.execution_time = round(perf_counter() - start, 3)
        return result

    # ══════════════════════════════════════════════════════════════
    # 分发
    # ══════════════════════════════════════════════════════════════

    def _dispatch(self, decision: ActionDecision) -> ExecutionResult:
        action = decision.action_type

        if action == ActionType.STOP:
            return self.backend.stop()

        if action == ActionType.EXPLORE_FRONTIER:
            return self._navigate(decision, EntityKind.FRONTIER, decision.frontier_id)

        if action == ActionType.APPROACH_TARGET:
            return self._navigate(decision, EntityKind.TARGET, decision.target_id)

        return ExecutionResult.failed(
            decision.task_id,
            decision.decision_id,
            ErrorCode.UNSUPPORTED_ACTION,
            f"第一版不支持的动作类型：{action}",
        )

    def _navigate(self, decision: ActionDecision, kind: str,
                  entity_id: Optional[str]) -> ExecutionResult:
        if not entity_id:
            return ExecutionResult.failed(
                decision.task_id, decision.decision_id,
                ErrorCode.UNKNOWN_ID, f"{kind} 类型决策缺少 ID",
            )

        pose = self._resolve(kind, entity_id)
        if pose is None:
            return ExecutionResult.failed(
                decision.task_id, decision.decision_id,
                ErrorCode.UNKNOWN_ID, f"无法解析 {kind} 坐标：{entity_id}",
            )

        return self.backend.navigate_to(pose)

    def _resolve(self, kind: str, entity_id: str) -> Optional[Sequence[float]]:
        if self.pose_resolver is None:
            return None
        pose = self.pose_resolver(kind, entity_id)
        if pose is None:
            return None
        if len(pose) != 3:
            raise ValueError(f"pose 必须是 3 维，收到 {pose!r}")
        return pose
