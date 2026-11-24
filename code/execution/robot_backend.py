"""RobotBackend —— 机器人执行后端的统一抽象。

这是"仿真 → 实机"解耦的唯一替换点：
上层（Decision / TaskExecutor）只依赖这个类，
换成 M1 实机时只替换实现，上层代码一行不动。

    Decision ──► TaskExecutor ──► RobotBackend
                                     ├── HabitatBackend   （第一版）
                                     └── M1Backend        （后续，接口相同）

第一版约定（docs/模块五_执行_方案.md Q1/Q2）：
  · navigate_to() 是**阻塞**的：走完才返回，status 不会是 RUNNING。
    M1 阶段需要异步时，再增加 poll 接口，不影响现有调用方。
  · scan() 返回**多帧**观测（默认 4 个朝向）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from common.schemas import ExecutionResult, ObservationPacket


class RobotBackend(ABC):
    """机器人执行后端。

    约定：
      · 所有坐标是世界坐标系下的 3D 点 (x, y, z)，单位米
      · 失败通过 ExecutionResult.status/error_code 表达，不抛异常
        （调用方是状态机，需要看到失败原因才能进入恢复流程）
    """

    # ══════════════════════════════════════════════════════════════
    # 感知
    # ══════════════════════════════════════════════════════════════

    @abstractmethod
    def observe(self) -> ObservationPacket:
        """取一帧观测（RGB-D + 位姿）。

        仿真：sim.get_sensor_observations() + agent.get_state()
        实机：相机驱动取帧 + 里程计/定位系统取位姿
        """

    @abstractmethod
    def scan(self) -> List[ObservationPacket]:
        """原地环视一圈，返回多帧观测。

        第一版实现：从当前朝向开始，每转 90° 取一帧，共 4 帧
        （朝向量可配置，见 HabitatBackend.scan）。

        仿真：agent.act("turn_right") 重复若干次 + 每档取一帧
        实机：底盘原地旋转 + 连续取帧
        """

    # ══════════════════════════════════════════════════════════════
    # 动作
    # ══════════════════════════════════════════════════════════════

    @abstractmethod
    def navigate_to(self, pose: Sequence[float]) -> ExecutionResult:
        """导航到世界坐标 pose。**阻塞**，走完才返回。

        Args:
            pose: (x, y, z) 目标点。不可达时返回 FAILED + NO_PATH，
                  不能抛异常、不能死循环。

        仿真：pathfinder.snap_point → ShortestPath → find_path → 逐帧 set_state
        实机：Nav2 / 底盘导航，阻塞直到到位或超时
        """

    @abstractmethod
    def cancel_navigation(self) -> ExecutionResult:
        """取消当前导航。

        用于 Decision 的恢复流程（方案 5.7 节）：
        导航超时或发现目标已失效时，主动叫停。

        仿真：置取消标志，navigate_to 的循环检测到后退出
        实机：向导航栈发取消指令
        """

    @abstractmethod
    def stop(self) -> ExecutionResult:
        """立即停止（不做路径清理以外的任何动作）。

        与 cancel_navigation 的区别：
          · cancel_navigation 是"放弃当前导航任务"
          · stop 是"停下"，通常用于任务结束（ActionType.STOP）

        仿真：清空目标与路径
        实机：发零速指令
        """

    # ══════════════════════════════════════════════════════════════
    # 生命周期（有默认实现，子类可选重写）
    # ══════════════════════════════════════════════════════════════

    def close(self) -> None:
        """释放资源。默认空操作。"""
        return None
