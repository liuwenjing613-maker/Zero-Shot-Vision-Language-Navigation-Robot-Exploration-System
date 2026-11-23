"""MockBackend —— 不依赖 Habitat / GPU / 网络的假执行后端。

存在的意义：
  1. **本地可测**：Decision / TaskExecutor 的逻辑可以脱离仿真器验证
  2. **故障注入**：可以精确制造 NO_PATH / TIMEOUT / STUCK 等失败
  3. **契约参考**：它是 RobotBackend 的最小完整实现，
     新写 M1Backend 时可以照它实现

设计成**完全确定性**的（不用随机数），测试结果可复现。
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from common.enums import ErrorCode, ExecutionStatus
from common.schemas import ExecutionResult, ObservationPacket
from .robot_backend import RobotBackend

IMG_H, IMG_W = 8, 8  # 故意用极小尺寸，测试跑得快


class MockBackend(RobotBackend):
    def __init__(self, position: Sequence[float] = (0.0, 0.0, 0.0),
                 reachable: Optional[set] = None) -> None:
        """
        Args:
            position: 初始位姿 (x, y, z)
            reachable: 可到达的点集合（元组）。None 表示"全部可到达"。
        """
        self.position = tuple(float(v) for v in position)
        self.reachable = reachable
        self.canceled = False

        # ── 测试控制开关（下一个 navigate_to 的行为）─────────────
        self.next_failure: Optional[str] = None   # None / "NO_PATH" / "TIMEOUT" / "STUCK"
        self.next_path_length: float = 1.0

        # ── 调用记录，便于断言 ────────────────────────────────
        self.calls: List[tuple] = []

        self._closed = False

    # ══════════════════════════════════════════════════════════════
    # 感知
    # ══════════════════════════════════════════════════════════════

    def observe(self) -> ObservationPacket:
        self.calls.append(("observe",))
        return ObservationPacket(
            task_id="mock",
            observation_id=f"obs_{len(self.calls)}",
            timestamp=perf_counter(),
            rgb=np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
            depth=np.full((IMG_H, IMG_W), 2.0, dtype=np.float32),
            robot_pose=((self.position), (1.0, 0.0, 0.0, 0.0)),
            source="mock",
        )

    def scan(self) -> List[ObservationPacket]:
        self.calls.append(("scan",))
        return [self.observe() for _ in range(4)]

    # ══════════════════════════════════════════════════════════════
    # 动作
    # ══════════════════════════════════════════════════════════════

    def navigate_to(self, pose: Sequence[float]) -> ExecutionResult:
        self.calls.append(("navigate_to", tuple(pose)))
        start = perf_counter()
        self.canceled = False

        target = tuple(float(v) for v in pose)

        # 1) 故障注入：优先于一切
        if self.next_failure == "NO_PATH":
            self.next_failure = None
            return ExecutionResult(
                task_id="", action_id="",
                status=ExecutionStatus.FAILED,
                error_code=ErrorCode.NO_PATH,
                error_message="mock: 目标不可达",
                execution_time=round(perf_counter() - start, 3),
            )
        if self.next_failure == "TIMEOUT":
            self.next_failure = None
            return ExecutionResult(
                task_id="", action_id="",
                status=ExecutionStatus.TIMEOUT,
                error_code=ErrorCode.TIMEOUT,
                error_message="mock: 导航超时",
                final_pose=self.position,
                execution_time=round(perf_counter() - start, 3),
            )
        if self.next_failure == "STUCK":
            self.next_failure = None
            return ExecutionResult(
                task_id="", action_id="",
                status=ExecutionStatus.FAILED,
                error_code=ErrorCode.STUCK,
                error_message="mock: 卡住",
                final_pose=self.position,
                execution_time=round(perf_counter() - start, 3),
            )

        # 2) 可达性检查
        if self.reachable is not None and target not in self.reachable:
            return ExecutionResult(
                task_id="", action_id="",
                status=ExecutionStatus.FAILED,
                error_code=ErrorCode.NO_PATH,
                error_message=f"mock: {target} 不在可到达集合内",
                final_pose=self.position,
                execution_time=round(perf_counter() - start, 3),
            )

        # 3) 正常到达
        path_length = float(np.linalg.norm(np.array(target) - np.array(self.position)))
        self.position = target
        return ExecutionResult(
            task_id="", action_id="",
            status=ExecutionStatus.SUCCEEDED,
            final_pose=self.position,
            path_length=self.next_path_length,
            execution_time=round(perf_counter() - start, 3),
        )

    def cancel_navigation(self) -> ExecutionResult:
        self.calls.append(("cancel_navigation",))
        self.canceled = True
        return ExecutionResult(
            task_id="", action_id="",
            status=ExecutionStatus.CANCELED,
            final_pose=self.position,
            error_message="mock: 已取消",
        )

    def stop(self) -> ExecutionResult:
        self.calls.append(("stop",))
        return ExecutionResult(
            task_id="", action_id="",
            status=ExecutionStatus.SUCCEEDED,
            final_pose=self.position,
        )

    # ══════════════════════════════════════════════════════════════

    def close(self) -> None:
        self._closed = True
