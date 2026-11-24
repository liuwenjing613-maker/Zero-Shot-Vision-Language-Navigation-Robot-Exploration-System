"""HabitatBackend —— RobotBackend 的 Habitat-Sim 实现。

第一版只做基础能力（方案 M1）：能观察、能环视、能移动、能停、能返回结果。

代码来源：复用之前在真 Habitat 上验证过的 A* / snap_point / set_state 逻辑
（见旧版 code/runtime/habitat_backend.py），只把接口改成 V4 方案要求的形状。

⚠️ 仿真限制（必须在文档和答辩里说明）：
    tick 的移动是「读位姿 → 算新坐标 → set_state 写回」，本质是**瞬移**，
    没有碰撞检测、没有动力学。真机上必须换成底盘运动控制。

⚠️ 依赖说明：
    habitat_sim 只在 __init__ 里导入（延迟导入），
    这样没装仿真器的机器也能 import 本模块、跑 Mock 单测。
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import List, Optional, Sequence

import numpy as np

from common.enums import ErrorCode, ExecutionStatus
from common.schemas import ExecutionResult, ObservationPacket
from .robot_backend import RobotBackend

# ── 第一版常量 ────────────────────────────────────────────────
IMG_H, IMG_W = 480, 640
HFOV_DEG = 90.0
MOVE_STEP = 0.15          # 每步位移（米）
ARRIVAL_THRESHOLD = 0.6   # 到达判定（米）
WAYPOINT_EPS = 0.1        # 距路点小于此值即认为到达该路点
MAX_STEPS = 300           # 单次导航最大步数（防死循环，也即超时保护）
TURN_DEG_PER_ACTION = 10.0


class HabitatBackend(RobotBackend):
    def __init__(
        self,
        scene_path: str,
        agent_height: float = 1.5,
        camera_offset: float = 0.5,
        start_position: Optional[Sequence[float]] = None,
        hfov_deg: float = HFOV_DEG,
        max_steps: int = MAX_STEPS,
        scan_views: int = 4,
        seed: Optional[int] = None,
        *,
        enable_physics: bool = True,
        scene_dataset: Optional[str] = None,
        navmesh: Optional[str] = None,
        gpu_device_id: int = 0,
        extra_sensors: Sequence = (),
    ) -> None:
        import habitat_sim  # 延迟导入（见模块 docstring）
        import magnum as mn

        self._hs = habitat_sim
        self._mn = mn
        self.max_steps = max_steps
        self.hfov_deg = hfov_deg
        # scan 的朝向量放在构造参数里，不放进 scan() ——
        # 抽象接口 scan(self) 保持最小，不暴露实现细节。
        if scan_views < 1:
            raise ValueError("scan_views 必须 >= 1")
        self.scan_views = scan_views

        if seed is not None:
            np.random.seed(seed)

        # ── 1. 构造配置 ───────────────────────────────────────
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = enable_physics
        sim_cfg.gpu_device_id = gpu_device_id
        if scene_dataset:
            sim_cfg.scene_dataset_config_file = scene_dataset

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = agent_height

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [IMG_H, IMG_W]
        rgb_spec.position = [0.0, camera_offset, 0.0]
        rgb_spec.hfov = hfov_deg

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [IMG_H, IMG_W]
        depth_spec.position = [0.0, camera_offset, 0.0]
        depth_spec.hfov = hfov_deg

        # ⚠️ 属性名是 sensor_specifications
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec, *extra_sensors]

        # ── 2. 创建仿真器 ─────────────────────────────────────
        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

        if navmesh is None:
            nav = habitat_sim.NavMeshSettings()
            nav.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, nav)
        elif not self._sim.pathfinder.load_nav_mesh(navmesh):
            self.close()
            raise ValueError(f"Could not load navmesh {navmesh}")

        self._agent = self._sim.initialize_agent(0)

        if start_position is not None:
            self._set_position(np.asarray(start_position, dtype=np.float64))

        # ── 3. 执行状态 ───────────────────────────────────────
        self._cancel_requested = False
        self._observation_counter = 0

    # ══════════════════════════════════════════════════════════════
    # 感知
    # ══════════════════════════════════════════════════════════════

    def observe(self) -> ObservationPacket:
        raw = self._sim.get_sensor_observations()
        rgb = np.ascontiguousarray(raw["color_sensor"][:, :, :3])
        depth = np.ascontiguousarray(raw["depth_sensor"]).astype(np.float32)

        state = self._agent.get_state()
        self._observation_counter += 1
        return ObservationPacket(
            task_id="",                       # 由上层（Executor/Decision）填
            observation_id=f"obs_{self._observation_counter:04d}",
            timestamp=perf_counter(),
            rgb=rgb,
            depth=depth,
            robot_pose=(
                tuple(float(v) for v in state.position),
                _quat_to_tuple(state.rotation),
            ),
            source="habitat",
        )

    def scan(self) -> List[ObservationPacket]:
        """原地环视：从当前朝向开始，每 360/scan_views 度取一帧。

        scan_views 在构造时指定（默认 4，即每 90° 一帧）。
        扫描结束后朝向回到起点方向（转满 360°）。
        """
        step_deg = 360.0 / self.scan_views
        frames: List[ObservationPacket] = []
        for _ in range(self.scan_views):
            frames.append(self.observe())
            self._rotate(step_deg)
        return frames

    # ══════════════════════════════════════════════════════════════
    # 动作
    # ══════════════════════════════════════════════════════════════

    def navigate_to(self, pose: Sequence[float]) -> ExecutionResult:
        """阻塞导航到 pose。走完才返回（Q1 的决策）。"""
        start_t = perf_counter()
        self._cancel_requested = False

        target = np.asarray(pose, dtype=np.float64)
        snapped = self._snap(target)
        if snapped is None:
            return self._failed(ErrorCode.NO_PATH, "目标点不可导航", start_t)

        path = self._plan(snapped)
        if path is None or len(path.points) < 1:
            return self._failed(ErrorCode.NO_PATH, "A* 未找到可行路径", start_t)

        length = float(path.geodesic_distance)
        points = [np.asarray(p, dtype=np.float64) for p in path.points]

        for _ in range(self.max_steps):
            if self._cancel_requested:
                self._clear_goal()
                return ExecutionResult(
                    task_id="", action_id="",
                    status=ExecutionStatus.CANCELED,
                    final_pose=self._position(),
                    execution_time=round(perf_counter() - start_t, 3),
                    error_message="导航被取消",
                )

            pos = self._position()
            dist = float(np.linalg.norm(snapped - pos))
            if dist < ARRIVAL_THRESHOLD:
                return ExecutionResult(
                    task_id="", action_id="",
                    status=ExecutionStatus.SUCCEEDED,
                    final_pose=tuple(pos),
                    path_length=length,
                    execution_time=round(perf_counter() - start_t, 3),
                )

            if points:
                next_pt = points[0] if len(points) == 1 else points[1]
                if float(np.linalg.norm(next_pt - pos)) < WAYPOINT_EPS:
                    points.pop(0)
                else:
                    direction = next_pt - pos
                    direction[1] = 0.0
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-6:
                        self._set_position(pos + direction / norm * MOVE_STEP)
            else:
                # 路点走完但没到（阈值内就该到）——视为卡住
                return self._failed(ErrorCode.STUCK, "路点耗尽但未到达", start_t,
                                    path_length=length)

        return self._failed(ErrorCode.TIMEOUT,
                            f"超过 {self.max_steps} 步未到达", start_t,
                            path_length=length)

    def cancel_navigation(self) -> ExecutionResult:
        self._cancel_requested = True
        return ExecutionResult(
            task_id="", action_id="",
            status=ExecutionStatus.CANCELED,
            final_pose=self._position(),
            error_message="已请求取消导航",
        )

    def stop(self) -> ExecutionResult:
        self._clear_goal()
        return ExecutionResult(
            task_id="", action_id="",
            status=ExecutionStatus.SUCCEEDED,
            final_pose=self._position(),
        )

    # ══════════════════════════════════════════════════════════════
    # 生命周期
    # ══════════════════════════════════════════════════════════════

    def close(self) -> None:
        if getattr(self, "_sim", None) is not None:
            self._sim.close()
            self._sim = None

    # ══════════════════════════════════════════════════════════════
    # 内部工具
    # ══════════════════════════════════════════════════════════════

    def _failed(self, code: str, message: str, start_t: float,
                path_length: float = 0.0) -> ExecutionResult:
        return ExecutionResult(
            task_id="", action_id="",
            status=ExecutionStatus.FAILED,
            final_pose=self._position(),
            path_length=path_length,
            execution_time=round(perf_counter() - start_t, 3),
            error_code=code,
            error_message=message,
        )

    def _snap(self, point: np.ndarray) -> Optional[np.ndarray]:
        if not self._sim.pathfinder.is_loaded:
            return None
        snapped = np.asarray(self._sim.pathfinder.snap_point(self._vec(point)),
                             dtype=np.float64)
        if not np.all(np.isfinite(snapped)):
            return None
        if float(np.linalg.norm(snapped - point)) > 3.0:
            return None
        return snapped

    def _plan(self, target: np.ndarray):
        path = self._hs.ShortestPath()
        path.requested_start = np.asarray(self._position(), dtype=np.float32)
        path.requested_end = target.astype(np.float32)
        if not self._sim.pathfinder.find_path(path):
            return None
        return path

    def _position(self) -> np.ndarray:
        return np.asarray(self._agent.get_state().position, dtype=np.float64)

    def _set_position(self, position: np.ndarray) -> None:
        snapped = self._snap(np.asarray(position, dtype=np.float64))
        if snapped is not None:
            position = snapped
        state = self._agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        self._agent.set_state(state)

    def _rotate(self, degrees: float) -> None:
        action = "turn_right" if degrees > 0 else "turn_left"
        steps = int(round(abs(degrees) / TURN_DEG_PER_ACTION))
        for _ in range(steps):
            self._agent.act(action)

    def _clear_goal(self) -> None:
        self._cancel_requested = False

    def _vec(self, point) -> "object":
        """numpy → magnum.Vector3（pathfinder 只认这个类型）。"""
        p = np.asarray(point, dtype=np.float64).ravel()
        return self._mn.Vector3(float(p[0]), float(p[1]), float(p[2]))


def _quat_to_tuple(q) -> tuple:
    return (float(q.w), float(q.x), float(q.y), float(q.z))
