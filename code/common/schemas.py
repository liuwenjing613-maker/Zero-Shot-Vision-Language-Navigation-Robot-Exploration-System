"""V4 第一版共享数据结构。

只定义 Execution 用得到的三个结构（ObservationPacket / ActionDecision /
ExecutionResult），其余（GoalSpec / PerceptionResult / WorldState）
由队长那边确定后再合并，避免重复定义产生分歧。

字段与 docs/模块五_执行_方案.md 第三节一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ObservationPacket:
    """一帧观测。由 RobotBackend.observe() / scan() 产出。

    rgb / depth 第一版直接放 numpy 数组，不做网络序列化
    （方案第七节："图像和地图不需要全部网络序列化"）。
    """

    task_id: str
    observation_id: str
    timestamp: float

    rgb: Any = None
    depth: Optional[Any] = None

    # (x, y, z) 世界坐标 + 朝向四元数 (w, x, y, z)
    robot_pose: Optional[Tuple[Sequence[float], Sequence[float]]] = None

    map_version: int = 0
    source: str = ""

    def position(self) -> Optional[Sequence[float]]:
        """取位置部分，方便上层只关心坐标的场景。"""
        return None if self.robot_pose is None else self.robot_pose[0]


@dataclass
class ActionDecision:
    """决策模块（成员 B）产出的动作决策。

    ⚠️ 注意：这里只有 ID，没有坐标。坐标由 TaskExecutor 通过
    pose_resolver 解析（方案 Q3 的决策）。
    """

    task_id: str
    decision_id: str
    action_type: str

    target_id: Optional[str] = None
    frontier_id: Optional[str] = None

    reason: str = ""
    map_version: int = 0


@dataclass
class ExecutionResult:
    """执行结果。由 TaskExecutor.execute() 产出，回传给 Memory / Decision。"""

    task_id: str
    action_id: str
    status: str

    final_pose: Optional[Sequence[float]] = None

    path_length: float = 0.0
    execution_time: float = 0.0

    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的 dict（numpy 标量会被转成 Python 数字）。"""
        return {
            "task_id": self.task_id,
            "action_id": self.action_id,
            "status": self.status,
            "final_pose": None if self.final_pose is None else [float(v) for v in self.final_pose],
            "path_length": float(self.path_length),
            "execution_time": float(self.execution_time),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def succeeded(cls, task_id: str, action_id: str, **kw: Any) -> "ExecutionResult":
        return cls(task_id=task_id, action_id=action_id, status="SUCCEEDED", **kw)

    @classmethod
    def failed(cls, task_id: str, action_id: str, error_code: str,
               error_message: str = "", **kw: Any) -> "ExecutionResult":
        return cls(task_id=task_id, action_id=action_id, status="FAILED",
                   error_code=error_code, error_message=error_message, **kw)
