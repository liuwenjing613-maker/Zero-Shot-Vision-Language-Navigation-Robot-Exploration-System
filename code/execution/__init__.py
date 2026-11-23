"""执行模块（模块五）。

    robot_backend.py    RobotBackend 抽象接口
    habitat_backend.py  Habitat-Sim 实现
    mock_backend.py     假后端（本地测试用）
    task_executor.py    ActionDecision → ExecutionResult
"""

from .mock_backend import MockBackend
from .robot_backend import RobotBackend
from .task_executor import PoseResolver, TaskExecutor

__all__ = [
    "RobotBackend",
    "TaskExecutor",
    "PoseResolver",
    "MockBackend",
]


def __getattr__(name):
    """HabitatBackend 延迟导出 —— 没装 habitat_sim 时不报错。"""
    if name == "HabitatBackend":
        from .habitat_backend import HabitatBackend
        return HabitatBackend
    raise AttributeError(name)
