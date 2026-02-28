# ==============================================================================
# 文件名: evaluate_system.py
# 功能: 自动化评估脚本（无头模式）- 支持 v2 / v3，统一用「终点与标注 gt 距离」判定成功
# 指标: SR (成功率)、SPL (路径效率)、NE (导航误差)；可对比 v2 与 v3
# ==============================================================================

import os
import sys
import json
import time
import math
import argparse
import threading
import signal
import numpy as np
import habitat_sim
import magnum as mn
import cv2
import requests
# 超时机制：使用 signal.SIGALRM（仅 Linux）
class EpisodeTimeoutError(Exception):
    """Episode 超时异常"""
    pass

def episode_timeout_handler(signum, frame):
    raise EpisodeTimeoutError("Episode execution timeout")

# 无头：不弹窗，不依赖 GUI
os.environ["MPLBACKEND"] = "Agg"

# ============== 配置（与 v2 一致）==============
CLOUD_URL = "http://127.0.0.1:5000/plan"
SCENES_BASE = "/home/abc/ZeroShot_VLN/assets/scenes"
SCENE_LIST = [
    "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "MatterPort3D/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    "MatterPort3D/mp3d/8194nk5LbLH/8194nk5LbLH.glb",
    "MatterPort3D/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb",
    "MatterPort3D/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb",
    "MatterPort3D/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb",
    "MatterPort3D/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
    "MatterPort3D/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
    "MatterPort3D/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb",
]

IMG_WIDTH, IMG_HEIGHT, HFOV = 640, 480, 110
CLOUD_SEND_INTERVAL = 0.6  # 与原版 v2/v3 一致
MOVE_STEP_SIZE = 0.06
REPLAN_DISTANCE_THRESHOLD = 1.5
STUCK_DEPTH_THRESHOLD = 0.6
STUCK_FRAMES = 4
DEPTH_MIN, DEPTH_MAX = 0.3, 5.0
TURN_INTERVAL = 12
INFER_SMOOTH_ALPHA = 0.25
INFER_TURN_COMMIT_FRAMES = 15
INFER_DEAD_ZONE = 100
SUCCESS_DEPTH_THRESHOLD_V1_V2 = 0.6  # v1/v2 使用
SUCCESS_DEPTH_THRESHOLD_V3 = 0.5    # v3 使用更严格的阈值
FBE_EXPLORE_RADIUS = 2.5
FBE_MIN_DISTANCE = 1.2
FBE_SEARCHING_FRAMES = 150  # Searching 状态持续 150 帧触发 FBE（约 3 秒）
MEMORY_DISTANCE_CHECK_INTERVAL = 5.0
MEMORY_DISTANCE_IMPROVE_THRESHOLD = 0.15
MEMORY_BLACKLIST_TOLERANCE = 0.5

# 评估专用：
# SR (Success Rate): 最终停止位置距离目标 < 阈值才算成功
# OSR (Oracle Success Rate): 导航过程中曾经有一次距离目标 < 阈值即算成功
EVAL_SUCCESS_DISTANCE = 2.0   # 2.0m：与原论文对齐（论文用 3.0m，这里用 2.0m）
# 单 episode 最大步数，避免死循环
EVAL_MAX_STEPS = 800  # 800 步，给机器人充足时间探索
EPISODE_TIMEOUT = 300  # 单个 episode 最大运行时间（秒）

# 保存成功案例的俯视轨迹图
SAVE_SUCCESS_TOPDOWN = True
SUCCESS_TOPDOWN_DIR = os.path.join(os.path.dirname(__file__), "success_topdown")
EVAL_EPISODES_JSON = os.path.join(os.path.dirname(__file__), "eval_episodes.json")

# 详细结果保存目录
DETAILED_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "detailed_results")

# v3 出生地环视
SPAWN_SCAN_VIEWS = 4
DEGREES_PER_TURN = 10
TURNS_PER_90 = max(1, int(math.ceil(90.0 / DEGREES_PER_TURN)))
FBE_SMOOTH_TURN_FRAMES = 12
STUCK_SMOOTH_TURN_FRAMES = 4
STUCK_TURN_WHEN_TARGET_LOCKED = 0
USE_VERIFY_BEFORE_SUCCESS = True

# ============== 根据 scene_id 解析场景路径 ==============
def get_scene_path_by_id(scene_id):
    for rel in SCENE_LIST:
        if scene_id in rel or rel.endswith(scene_id + ".glb"):
            full = os.path.join(SCENES_BASE, rel)
            if os.path.exists(full):
                return full
    return None

# ============== 共享状态（与 v2 一致）==============
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_rgb = None
        self.current_depth = None
        self.current_agent_state = None
        self.new_image_ready = False
        self.latest_goal_uv = None
        self.latest_goal_depth = None
        self.latest_goal_camera_snapshot = None
        self.latest_status = "Searching..."
        self.current_instruction = ""
        self.target_list = []
        self.target_index = 0
        self.latest_instruction_used = None
        # 云端错误追踪
        self.cloud_error_count = 0
        self.cloud_last_success_time = time.time()
        self.cloud_fatal_error = False  # 严重错误标志（如连续多次失败）

# ============== Habitat 配置 ==============
def make_cfg(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid, rgb_spec.resolution = "color_sensor", [IMG_HEIGHT, IMG_WIDTH]
    rgb_spec.position, rgb_spec.hfov = [0.0, 0.5, 0.0], HFOV
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid, depth_spec.sensor_type = "depth_sensor", habitat_sim.SensorType.DEPTH
    depth_spec.resolution, depth_spec.position, depth_spec.hfov = [IMG_HEIGHT, IMG_WIDTH], [0.0, 0.5, 0.0], HFOV
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# ============== 工具函数（与 v2 一致）==============
def get_depth_at_uv(u, v, depth_img):
    u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
    v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
    patch = depth_img[max(0, v_idx-2):v_idx+3, max(0, u_idx-2):u_idx+3]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))

def get_3d_point(u, v, depth_img, agent_state, sim, camera_snapshot=None):
    u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
    v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
    patch = depth_img[max(0, v_idx-2):v_idx+3, max(0, u_idx-2):u_idx+3]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None, None
    z_depth = float(np.median(valid))
    if z_depth < DEPTH_MIN or z_depth > DEPTH_MAX:
        return None, None
    f = (IMG_WIDTH / 2.0) / np.tan(np.deg2rad(HFOV) / 2.0)
    cx, cy = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0
    x_c = (u_idx - cx) * z_depth / f
    y_c = -(v_idx - cy) * z_depth / f
    z_c = -z_depth
    if camera_snapshot is not None:
        pos_xyz, rot_real, rot_imag = camera_snapshot
        pos = mn.Vector3(pos_xyz)
        mn_q = mn.Quaternion(mn.Vector3(rot_imag), rot_real)
    else:
        cam = agent_state.sensor_states["color_sensor"]
        pos = mn.Vector3(cam.position)
        q = cam.rotation
        mn_q = mn.Quaternion(mn.Vector3(q.imag), q.real)
    mat = mn.Matrix4.from_(mn_q.to_matrix(), pos)
    world = mat.transform_point(mn.Vector3(x_c, y_c, z_c))
    raw_3d = np.array([world.x, world.y, world.z], dtype=np.float32)
    if sim.pathfinder.is_loaded:
        snapped = sim.pathfinder.snap_point(raw_3d)
        if not np.isnan(snapped).any():
            return snapped, z_depth
    return raw_3d, z_depth

def get_explore_waypoint(sim, curr_pos):
    if not sim.pathfinder.is_loaded:
        return None
    try:
        start_snap = sim.pathfinder.snap_point(np.array(curr_pos, dtype=np.float32))
        island_idx = sim.pathfinder.get_island(start_snap)
        for _ in range(10):
            pt = sim.pathfinder.get_random_navigable_point_near(
                start_snap, FBE_EXPLORE_RADIUS, island_index=island_idx
            )
            if pt is not None and not np.isnan(pt).any():
                dist = np.linalg.norm(np.array(pt) - curr_pos)
                if dist >= FBE_MIN_DISTANCE:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = start_snap
                    path.requested_end = np.array(pt, dtype=np.float32)
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        return np.array(pt, dtype=np.float32)
    except Exception:
        pass
    return None

def is_goal_in_blacklist(goal_3d, blacklist):
    for bl in blacklist:
        if np.linalg.norm(np.array(goal_3d) - np.array(bl)) < MEMORY_BLACKLIST_TOLERANCE:
            return True
    return False

def request_scan_confidence(rgb_bgr, instruction, timeout=15):
    """v3 出生地环视：请求云端 scan 模式，返回目标可能性 0~1。"""
    try:
        # 压缩质量降低，减轻云端显存压力
        _, buf = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                          data={'instruction': instruction, 'scan': '1'}, timeout=timeout)
        return float(r.json().get('confidence', 0.0))
    except Exception:
        return 0.0

# ============== 云端 worker（与 v2 一致，instruction 由外部设置）==============
CLOUD_MAX_CONSECUTIVE_ERRORS = 10  # 连续错误超过此数则标记为严重错误
CLOUD_REQUEST_TIMEOUT = 15  # 云端请求超时（秒）- 给模型充足推理时间

def cloud_worker(shared_state):
    last_send = 0.0
    while True:
        # 检查是否已标记为严重错误
        with shared_state.lock:
            if shared_state.cloud_fatal_error:
                time.sleep(1)
                continue
        
        now = time.time()
        if now - last_send < CLOUD_SEND_INTERVAL:
            time.sleep(0.3)
            continue
        img_to_send = None
        depth_snap = None
        cam_snap = None
        with shared_state.lock:
            if shared_state.new_image_ready and shared_state.current_rgb is not None:
                img_to_send = shared_state.current_rgb.copy()
                depth_snap = shared_state.current_depth.copy() if shared_state.current_depth is not None else None
                if shared_state.current_agent_state is not None:
                    c = shared_state.current_agent_state.sensor_states["color_sensor"]
                    cam_snap = (np.array(c.position), float(c.rotation.real), np.array(c.rotation.imag))
                shared_state.new_image_ready = False
                instruction = shared_state.current_instruction
        if img_to_send is not None:
            last_send = time.time()
            # 压缩质量降低，减轻云端显存压力
            _, buf = cv2.imencode('.jpg', img_to_send, [cv2.IMWRITE_JPEG_QUALITY, 70])
            try:
                data = {'instruction': instruction}
                if shared_state.target_list:
                    data['target_list'] = json.dumps(shared_state.target_list)
                    data['target_index'] = str(shared_state.target_index)
                r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')}, data=data, timeout=CLOUD_REQUEST_TIMEOUT)
                j = r.json()
                with shared_state.lock:
                    shared_state.cloud_error_count = 0  # 成功，重置错误计数
                    shared_state.cloud_last_success_time = time.time()
                    shared_state.latest_instruction_used = instruction
                    if j.get('status') == 'success':
                        shared_state.latest_goal_uv = (j['u'], j['v'])
                        shared_state.latest_goal_depth = depth_snap
                        shared_state.latest_goal_camera_snapshot = cam_snap
                        shared_state.latest_status = "Target Locked"
                    elif j.get('message') == 'Inferred' and 'u' in j and 'v' in j:
                        shared_state.latest_goal_uv = (j['u'], j['v'])
                        shared_state.latest_goal_depth = depth_snap
                        shared_state.latest_goal_camera_snapshot = cam_snap
                        shared_state.latest_status = "Inferred"
                    else:
                        shared_state.latest_status = "Searching..."
                        shared_state.latest_goal_uv = None
                        shared_state.latest_goal_depth = None
                        shared_state.latest_goal_camera_snapshot = None
            except Exception as e:
                with shared_state.lock:
                    shared_state.cloud_error_count += 1
                    if shared_state.cloud_error_count >= CLOUD_MAX_CONSECUTIVE_ERRORS:
                        shared_state.cloud_fatal_error = True
                        print(f"\n❌ 云端连续 {CLOUD_MAX_CONSECUTIVE_ERRORS} 次错误，标记为严重错误: {e}")
        time.sleep(0.3)

# ============== 单 episode 评估（无头；version=v1/v2/v3，成功=过程内任意时刻与 gt 距离 < EVAL_SUCCESS_DISTANCE）==============
def run_episode(episode, shared_state, version="v2"):
    scene_path = get_scene_path_by_id(episode["scene_id"])
    if scene_path is None:
        return {"error": f"scene not found: {episode['scene_id']}"}

    start_pos = np.array(episode["start_position"], dtype=np.float32)
    gt_pos = np.array(episode["gt_position"], dtype=np.float32)
    instruction = episode["instruction"]

    sim = habitat_sim.Simulator(make_cfg(scene_path))
    nav = habitat_sim.NavMeshSettings()
    nav.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, nav)
    agent = sim.initialize_agent(0)

    start_snap = sim.pathfinder.snap_point(start_pos)
    if np.isnan(start_snap).any():
        sim.close()
        return {"error": "start_position not navigable"}
    s = agent.get_state()
    s.position = start_snap
    agent.set_state(s)

    gt_snap = sim.pathfinder.snap_point(gt_pos)
    if np.isnan(gt_snap).any():
        gt_snap = gt_pos

    path_shortest = habitat_sim.ShortestPath()
    path_shortest.requested_start = start_snap
    path_shortest.requested_end = gt_snap
    geodesic_distance = 0.0
    if sim.pathfinder.find_path(path_shortest):
        geodesic_distance = float(path_shortest.geodesic_distance)
    else:
        geodesic_distance = float(np.linalg.norm(gt_snap - start_snap))

    with shared_state.lock:
        shared_state.target_list = [instruction]
        shared_state.target_index = 0
        shared_state.current_instruction = instruction
        shared_state.latest_status = "Searching..."
        shared_state.latest_goal_uv = None
        shared_state.latest_goal_depth = None
        shared_state.latest_goal_camera_snapshot = None

    thr = threading.Thread(target=cloud_worker, args=(shared_state,), daemon=True)
    thr.start()

    path_points = []
    current_goal_3d = None
    goal_3d_recorded_at = None
    goal_3d_recorded_dist = None
    goal_blacklist = []
    stuck_count = 0
    step_count = 0
    smoothed_offset = None
    turn_commit_remaining = 0
    last_turn_direction = None
    searching_frames = 0
    in_fbe_mode = False
    smooth_turn_remaining = 0
    current_target_idx = 0
    target_list = [instruction]

    # 根据版本选择成功深度阈值
    SUCCESS_DEPTH_THRESHOLD = SUCCESS_DEPTH_THRESHOLD_V3 if version == "v3" else SUCCESS_DEPTH_THRESHOLD_V1_V2
    
    # v3 专用
    spawn_scan_done = True if version != "v3" else False
    spawn_scan_view_index = 0
    spawn_scan_confidences = []
    spawn_scan_phase = "capture"
    spawn_scan_turn_remaining = 0
    spawn_turn_to_best_remaining = 0
    spawn_scan_best_direction = 0
    target_fully_verified = False
    
    # 位置卡住检测（v2/v3 通用）：即使状态是 Inferred，如果位置长时间不变也触发 FBE
    stuck_position_check_interval = 50   # 每 50 帧检查一次（更频繁）
    stuck_position_threshold = 0.25      # 位置移动小于 0.25m 视为卡住
    stuck_position_frames = 0            # 连续卡住帧数
    stuck_position_last_pos = None       # 上次检查的位置
    STUCK_FBE_TRIGGER_FRAMES = 150       # 连续卡住 150 帧触发 FBE（约 3 秒）
    fbe_trigger_count = 0                # FBE 触发次数统计（仅用于日志）
    fbe_start_step = 0                   # FBE 开始的步数（仅用于日志）

    trajectory = [np.array(agent.get_state().position)]
    actual_path_length = 0.0
    success = False  # SR: 最终是否成功（首次达到阈值即停止）
    oracle_success = False  # OSR: 过程中是否曾经到达过阈值内
    min_dist_to_gt = float('inf')  # 追踪最小距离
    final_pos = None
    episode_start_time = time.time()  # 记录 episode 开始时间

    curr_pos = np.array(agent.get_state().position)  # 初始化，避免 break 时引用错误
    
    # 进展检测：如果长时间没有接近目标，提前结束
    NO_PROGRESS_STEPS = 400  # 400 步无进展则结束
    last_progress_step = 0   # 上次有进展的步数
    last_progress_dist = float('inf')  # 上次记录的最小距离
    
    def check_timeout():
        """检查是否超时，返回 True 表示应该退出"""
        return time.time() - episode_start_time > EPISODE_TIMEOUT
    
    try:
        for step in range(EVAL_MAX_STEPS):
            # 检查 episode 超时（每步开始检查）
            if check_timeout():
                print(f"⏰ episode 内部超时 ({time.time() - episode_start_time:.1f}s)")
                final_pos = curr_pos.copy()
                break
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][:, :, :3][..., ::-1]
            depth = obs["depth_sensor"]
            curr_state = agent.get_state()
            curr_pos = np.array(curr_state.position)
            if len(trajectory) > 0:
                actual_path_length += np.linalg.norm(curr_pos - trajectory[-1])
            trajectory.append(curr_pos.copy())

            with shared_state.lock:
                shared_state.current_rgb = rgb
                shared_state.current_depth = depth
                shared_state.current_agent_state = curr_state
                shared_state.new_image_ready = True
                uv = shared_state.latest_goal_uv
                status = shared_state.latest_status
                snap_d = shared_state.latest_goal_depth
                snap_c = shared_state.latest_goal_camera_snapshot
                latest_instruction_used = shared_state.latest_instruction_used
                # 检测云端严重错误
                if shared_state.cloud_fatal_error:
                    raise ConnectionError("云端服务连续失败，可能已 OOM")

            # 距离计算：追踪最小距离（用于 OSR）
            dist_to_gt = np.linalg.norm(curr_pos - gt_snap)
            if dist_to_gt < min_dist_to_gt:
                min_dist_to_gt = dist_to_gt
                last_progress_step = step_count
                last_progress_dist = dist_to_gt
            # OSR: 只要曾经到达过阈值内
            if dist_to_gt < EVAL_SUCCESS_DISTANCE:
                oracle_success = True
            
            # 无进展检测：长时间没有接近目标则提前结束（避免无意义循环）
            if step_count - last_progress_step > NO_PROGRESS_STEPS and step_count > 100:
                print(f"  [无进展] {NO_PROGRESS_STEPS} 步无改善，提前结束 (min_dist={min_dist_to_gt:.2f}m)")
                final_pos = curr_pos.copy()
                break

            # ---------- v3 出生地环视（仅 version=="v3" 且未完成时）--------------
            if version == "v3" and not spawn_scan_done:
                if spawn_scan_phase == "capture" and len(spawn_scan_confidences) == spawn_scan_view_index:
                    conf = request_scan_confidence(rgb[..., ::-1], instruction)
                    spawn_scan_confidences.append(conf)
                    if spawn_scan_view_index < SPAWN_SCAN_VIEWS - 1:
                        spawn_scan_turn_remaining = TURNS_PER_90
                        spawn_scan_phase = "turning"
                    else:
                        best = int(np.argmax(spawn_scan_confidences))
                        spawn_scan_best_direction = best
                        turn_90_right = (best - (SPAWN_SCAN_VIEWS - 1)) % SPAWN_SCAN_VIEWS
                        spawn_turn_to_best_remaining = turn_90_right * TURNS_PER_90
                        spawn_scan_phase = "turning_to_best"
                if spawn_turn_to_best_remaining > 0:
                    agent.act("turn_right")
                    spawn_turn_to_best_remaining -= 1
                    if spawn_turn_to_best_remaining == 0:
                        spawn_scan_done = True
                elif spawn_scan_phase == "turning" and spawn_scan_turn_remaining > 0:
                    agent.act("turn_right")
                    spawn_scan_turn_remaining -= 1
                    if spawn_scan_turn_remaining == 0:
                        spawn_scan_view_index += 1
                        spawn_scan_phase = "capture"
                step_count += 1
                time.sleep(0.02)
                continue

            # ---------- v1/v2/v3 通用逻辑 ----------
            center = depth[IMG_HEIGHT//2-30:IMG_HEIGHT//2+30, IMG_WIDTH//2-30:IMG_WIDTH//2+30]
            valid = center[(center > 0.1) & (center < 5.0)]
            front_d = float(np.median(valid)) if len(valid) > 0 else 2.0
            if front_d < STUCK_DEPTH_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= STUCK_FRAMES:
                if version == "v3" and target_fully_verified:
                    smooth_turn_remaining = STUCK_TURN_WHEN_TARGET_LOCKED
                    stuck_count = 0
                else:
                    path_points = []
                    current_goal_3d = None
                    if version == "v3":
                        target_fully_verified = False
                    stuck_count = 0
                    with shared_state.lock:
                        shared_state.latest_goal_uv = None
                        shared_state.latest_goal_depth = None
                        shared_state.latest_goal_camera_snapshot = None
                    # v1 没有 FBE，只简单转向；v2/v3 有 FBE
                    if version in ("v2", "v3"):
                        explore_waypoint = get_explore_waypoint(sim, curr_pos)
                        if explore_waypoint is not None:
                            path = habitat_sim.ShortestPath()
                            path.requested_start = sim.pathfinder.snap_point(curr_pos)
                            path.requested_end = explore_waypoint
                            if sim.pathfinder.find_path(path) and len(path.points) > 1:
                                path_points = list(path.points)
                                in_fbe_mode = True
                                fbe_start_step = step_count
                                fbe_trigger_count += 1
                        else:
                            for _ in range(4):
                                agent.act("turn_right")
                                time.sleep(0.08)
                    else:
                        # v1: 简单转向
                        for _ in range(4):
                            agent.act("turn_right")
                            time.sleep(0.08)

            has_path = path_points and len(path_points) > 1
            use_goal = uv and snap_d is not None and snap_c is not None

            if in_fbe_mode and status == "Target Locked" and use_goal:
                print(f"  [FBE] 锁定目标，退出FBE, step={step_count}, 在FBE中 {step_count - fbe_start_step} 步")
                in_fbe_mode = False
                path_points = []
                stuck_position_frames = 0  # 重置卡住计数
                searching_frames = 0  # 重置 searching 计数
                stuck_position_last_pos = None

            if in_fbe_mode and path_points:
                dist_to_wp = np.linalg.norm(curr_pos - path_points[-1])
                if dist_to_wp < 0.3:
                    in_fbe_mode = False
                    path_points = []
                    smooth_turn_remaining = FBE_SMOOTH_TURN_FRAMES if version == "v3" else 8
                    stuck_position_frames = 0  # 重置卡住计数
                    searching_frames = 0  # 重置 searching 计数
                    stuck_position_last_pos = None  # 重置位置检测基准
                    print(f"  [FBE] 到达探索点，开始环视, step={step_count}")
                    if version != "v3":
                        for _ in range(8):
                            agent.act("turn_right")
                            time.sleep(0.05)
            
            current_target_name = target_list[current_target_idx] if current_target_idx < len(target_list) else None
            if (status == "Target Locked" and uv is not None and snap_d is not None
                    and current_target_name is not None and latest_instruction_used == current_target_name):
                # v3 用当前帧深度判断，v1/v2 用 snapshot 深度（与原版一致）
                check_depth = depth if version == "v3" else snap_d
                d_at_target = get_depth_at_uv(uv[0], uv[1], check_depth)
                if d_at_target is not None and d_at_target < SUCCESS_DEPTH_THRESHOLD:
                    # v3: 需要 verify 验证；v2 也需要（USE_VERIFY_BEFORE_SUCCESS）
                    verified = target_fully_verified  # v3 已通过验证一次则不再请求
                    if not verified and USE_VERIFY_BEFORE_SUCCESS and version in ("v2", "v3"):
                        if check_timeout():  # 在耗时操作前检查超时
                            final_pos = curr_pos.copy()
                            break
                        try:
                            _, buf = cv2.imencode('.jpg', rgb)
                            r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                              data={'instruction': current_target_name, 'verify': '1'}, timeout=CLOUD_REQUEST_TIMEOUT)
                            verified = r.json().get('status') == 'success'
                            if not verified:
                                # 验证未通过，清除状态
                                with shared_state.lock:
                                    shared_state.latest_goal_uv = None
                                    shared_state.latest_goal_depth = None
                                    shared_state.latest_goal_camera_snapshot = None
                                    shared_state.latest_status = "Searching..."
                        except Exception:
                            verified = False
                            with shared_state.lock:
                                shared_state.latest_status = "Searching..."
                    else:
                        verified = True  # v1 不需要 verify
                    
                    if verified:
                        current_target_idx += 1
                        if current_target_idx >= len(target_list):
                            path_points = []
                            current_goal_3d = None
                        with shared_state.lock:
                            shared_state.target_index = current_target_idx
                            if current_target_idx < len(target_list):
                                shared_state.current_instruction = target_list[current_target_idx]
                            shared_state.latest_goal_uv = None
                            shared_state.latest_goal_depth = None
                            shared_state.latest_goal_camera_snapshot = None
                            shared_state.latest_status = "Searching..."
                        path_points = []
                        current_goal_3d = None
                        if version == "v3":
                            target_fully_verified = False
                        time.sleep(0.5)

            if current_goal_3d is not None and goal_3d_recorded_at is not None:
                elapsed = time.time() - goal_3d_recorded_at
                if elapsed >= MEMORY_DISTANCE_CHECK_INTERVAL:
                    curr_dist = np.linalg.norm(curr_pos - current_goal_3d)
                    if goal_3d_recorded_dist is not None:
                        improved = goal_3d_recorded_dist - curr_dist
                        if improved < MEMORY_DISTANCE_IMPROVE_THRESHOLD:
                            goal_blacklist.append(np.array(current_goal_3d, dtype=np.float32))
                            path_points = []
                            current_goal_3d = None
                            goal_3d_recorded_at = None
                            with shared_state.lock:
                                shared_state.latest_goal_uv = None
                                shared_state.latest_goal_depth = None
                                shared_state.latest_goal_camera_snapshot = None
                        else:
                            goal_3d_recorded_at = time.time()
                            goal_3d_recorded_dist = curr_dist
                    else:
                        goal_3d_recorded_at = time.time()
                        goal_3d_recorded_dist = curr_dist

            if not in_fbe_mode and status == "Target Locked" and not has_path and use_goal:
                goal_3d, gdepth = get_3d_point(uv[0], uv[1], snap_d, curr_state, sim, camera_snapshot=snap_c)
                if goal_3d is not None and gdepth is not None and not is_goal_in_blacklist(goal_3d, goal_blacklist):
                    replan = current_goal_3d is None or np.linalg.norm(np.array(goal_3d) - np.array(current_goal_3d)) > REPLAN_DISTANCE_THRESHOLD
                    if replan:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(curr_pos)
                        path.requested_end = np.array(goal_3d, dtype=np.float32)
                        if sim.pathfinder.find_path(path) and len(path.points) > 0:
                            current_goal_3d = goal_3d
                            path_points = list(path.points)
                            goal_3d_recorded_at = time.time()
                            goal_3d_recorded_dist = np.linalg.norm(curr_pos - np.array(goal_3d))
                            if version == "v3" and USE_VERIFY_BEFORE_SUCCESS:
                                if check_timeout():  # 在耗时操作前检查超时
                                    final_pos = curr_pos.copy()
                                    break
                                try:
                                    _, buf = cv2.imencode('.jpg', rgb)
                                    rv = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                                      data={'instruction': instruction, 'verify': '1'}, timeout=CLOUD_REQUEST_TIMEOUT)
                                    target_fully_verified = rv.json().get('status') == 'success'
                                except Exception:
                                    target_fully_verified = False
            elif status in ("Inferred", "Searching"):
                if not in_fbe_mode and (version != "v3" or not target_fully_verified):
                    path_points = []
                    current_goal_3d = None
                if status == "Searching":
                    smoothed_offset = None
                    turn_commit_remaining = 0
                    searching_frames += 1
                else:
                    searching_frames = 0

            # FBE 触发条件：Searching 状态 + 不在 FBE 模式
            # 与原始 v3 代码一致：target_fully_verified 或有 current_goal_3d 时不触发
            if status == "Searching" and not in_fbe_mode and searching_frames >= FBE_SEARCHING_FRAMES:
                should_trigger = True
                if version == "v3" and target_fully_verified:
                    should_trigger = False  # 完全确认后不触发 FBE
                if version == "v3" and current_goal_3d is not None:
                    should_trigger = False  # 有导航目标也不触发
                
                if should_trigger:
                    explore_waypoint = get_explore_waypoint(sim, curr_pos)
                    if explore_waypoint is not None:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(curr_pos)
                        path.requested_end = explore_waypoint
                        if sim.pathfinder.find_path(path) and len(path.points) > 1:
                            path_points = list(path.points)
                            in_fbe_mode = True
                            fbe_start_step = step_count
                            searching_frames = 0
                            fbe_trigger_count += 1

            front_clear = front_d >= 0.85
            if smooth_turn_remaining > 0:
                agent.act("turn_right")
                smooth_turn_remaining -= 1
            elif path_points:
                next_pt = path_points[0] if len(path_points) == 1 else path_points[1]
                move_vec = next_pt - curr_pos
                d = np.linalg.norm(move_vec)
                if d < 0.1:
                    path_points.pop(0)
                else:
                    step_pos = curr_pos + (move_vec / d) * MOVE_STEP_SIZE
                    s = agent.get_state()
                    s.position = step_pos
                    agent.set_state(s)
            else:
                if status == "Inferred" and turn_commit_remaining > 0:
                    turn_commit_remaining -= 1
                if step_count % TURN_INTERVAL != 0:
                    pass
                elif status == "Inferred" and uv is not None:
                    cx = IMG_WIDTH / 2.0
                    raw_offset = uv[0] - cx
                    if smoothed_offset is None:
                        smoothed_offset = float(raw_offset)
                    else:
                        smoothed_offset = (1 - INFER_SMOOTH_ALPHA) * smoothed_offset + INFER_SMOOTH_ALPHA * raw_offset
                    offset = smoothed_offset
                    if turn_commit_remaining > 0:
                        agent.act("turn_right" if last_turn_direction == "right" else "turn_left")
                    elif offset > INFER_DEAD_ZONE:
                        agent.act("turn_right")
                        turn_commit_remaining = INFER_TURN_COMMIT_FRAMES
                        last_turn_direction = "right"
                    elif offset < -INFER_DEAD_ZONE:
                        agent.act("turn_left")
                        turn_commit_remaining = INFER_TURN_COMMIT_FRAMES
                        last_turn_direction = "left"
                    elif front_clear:
                        agent.act("move_forward")
                        turn_commit_remaining = 0
                    else:
                        agent.act("turn_right")
                        turn_commit_remaining = INFER_TURN_COMMIT_FRAMES // 2
                        last_turn_direction = "right"
                else:
                    agent.act("turn_right")

            # 位置卡住检测
            if step_count % stuck_position_check_interval == 0:
                if stuck_position_last_pos is not None:
                    dist_moved = np.linalg.norm(curr_pos - stuck_position_last_pos)
                    if dist_moved < stuck_position_threshold:
                        stuck_position_frames += stuck_position_check_interval
                    else:
                        stuck_position_frames = 0
                stuck_position_last_pos = curr_pos.copy()
            
            # v1: 位置卡住直接视为失败，结束 episode
            if version == "v1" and stuck_position_frames >= STUCK_FBE_TRIGGER_FRAMES:
                final_pos = curr_pos.copy()
                break  # 结束 episode
            
            # v2/v3: 位置卡住触发 FBE（与原始代码一致）
            if version in ("v2", "v3") and spawn_scan_done and not in_fbe_mode:
                should_check = True if version == "v2" else not target_fully_verified
                if should_check and stuck_position_frames >= STUCK_FBE_TRIGGER_FRAMES:
                    explore_waypoint = get_explore_waypoint(sim, curr_pos)
                    if explore_waypoint is not None:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(curr_pos)
                        path.requested_end = explore_waypoint
                        if sim.pathfinder.find_path(path) and len(path.points) > 1:
                            path_points = list(path.points)
                            in_fbe_mode = True
                            fbe_start_step = step_count
                            stuck_position_frames = 0
                            searching_frames = 0
                            stuck_position_last_pos = None
                            fbe_trigger_count += 1

            step_count += 1
            time.sleep(0.02)
        else:
            final_pos = np.array(agent.get_state().position)
    finally:
        sim.close()

    if final_pos is None:
        final_pos = trajectory[-1] if trajectory else start_snap
    
    # 最终位置到目标的距离
    nav_error = float(np.linalg.norm(final_pos - gt_snap))
    
    # SR (Success Rate): 最终停止位置距离目标 < 阈值才算成功
    success = nav_error < EVAL_SUCCESS_DISTANCE
    
    # OSR (Oracle Success Rate): 过程中曾经到达过阈值内即算成功
    if min_dist_to_gt < EVAL_SUCCESS_DISTANCE:
        oracle_success = True
    
    # SPL: 只有 SR 成功才计算
    if geodesic_distance <= 0:
        spl = 0.0 if not success else 1.0
    else:
        spl = (1.0 if success else 0.0) * (geodesic_distance / max(actual_path_length, geodesic_distance))
    
    return {
        "success": bool(success),
        "oracle_success": bool(oracle_success),
        "SPL": float(spl),
        "NE": float(nav_error),
        "geodesic": float(geodesic_distance),
        "min_dist": float(min_dist_to_gt),
        "trajectory": trajectory,
        "start_pos": start_snap,
        "gt_pos": gt_snap,
        "scene_id": episode["scene_id"],
        "instruction": instruction,
    }

# ============== 断点续跑支持 ==============
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def get_checkpoint_path(version, run_idx):
    """返回某版本某轮的 checkpoint 文件路径"""
    return os.path.join(CHECKPOINT_DIR, f"ckpt_{version}_run{run_idx}.json")

def load_checkpoint(version, run_idx):
    """加载 checkpoint，返回 (已完成的 episode_id 集合, results 列表)"""
    path = get_checkpoint_path(version, run_idx)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            done_ids = set(data.get("done_ids", []))
            results = data.get("results", [])
            return done_ids, results
        except Exception:
            pass
    return set(), []

def save_checkpoint(version, run_idx, done_ids, results):
    """保存 checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = get_checkpoint_path(version, run_idx)
    # 确保所有数据都是 JSON 可序列化的
    serializable_ids = [int(x) if hasattr(x, 'item') else x for x in done_ids]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"done_ids": serializable_ids, "results": results}, f, ensure_ascii=False)

def clear_checkpoint(version, run_idx):
    """删除 checkpoint（一轮跑完后清理）"""
    path = get_checkpoint_path(version, run_idx)
    if os.path.exists(path):
        os.remove(path)

def save_topdown_trajectory(result, version, run_idx, ep_id):
    """保存成功案例的俯视轨迹图"""
    if not SAVE_SUCCESS_TOPDOWN:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(SUCCESS_TOPDOWN_DIR, exist_ok=True)
        
        trajectory = result.get("trajectory", [])
        start_pos = result.get("start_pos")
        gt_pos = result.get("gt_pos")
        instruction = result.get("instruction", "")
        
        if len(trajectory) < 2:
            return
        
        # 提取 x, z 坐标
        xs = [p[0] for p in trajectory]
        zs = [p[2] for p in trajectory]
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # 绘制轨迹
        ax.plot(xs, zs, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
        
        # 起点
        if start_pos is not None:
            ax.scatter([start_pos[0]], [start_pos[2]], c='green', s=100, marker='s', label='Start', zorder=5)
        
        # 终点（轨迹最后一点）
        ax.scatter([xs[-1]], [zs[-1]], c='blue', s=80, marker='o', label='End', zorder=5)
        
        # GT 目标
        if gt_pos is not None:
            ax.scatter([gt_pos[0]], [gt_pos[2]], c='red', s=120, marker='*', label='GT Goal', zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{version} | EP {ep_id} | {instruction[:30]}...' if len(instruction) > 30 else f'{version} | EP {ep_id} | {instruction}')
        ax.legend(loc='best')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        filename = f"{version}_run{run_idx}_ep{ep_id}.png"
        filepath = os.path.join(SUCCESS_TOPDOWN_DIR, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  [保存轨迹图失败: {e}]", end="")

def check_cloud_health():
    """检查云端服务是否健康"""
    try:
        r = requests.get(CLOUD_URL.replace("/plan", "/health"), timeout=5)
        return r.status_code == 200
    except:
        return False

def wait_for_cloud_recovery(max_wait=300, check_interval=15):
    """等待云端服务恢复，最多等待 max_wait 秒"""
    print(f"\n⚠️ 云端服务不可用，等待恢复（最多 {max_wait}s）...")
    start = time.time()
    while time.time() - start < max_wait:
        # 尝试简单的连接测试
        try:
            r = requests.post(CLOUD_URL, timeout=10, 
                            files={'image': ('test.jpg', b'\xff\xd8\xff\xe0', 'image/jpeg')},
                            data={'instruction': 'test'})
            if r.status_code in (200, 400, 500):  # 任何响应都说明服务在运行
                print(f"✅ 云端服务已恢复")
                time.sleep(5)  # 额外等待确保稳定
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            print(f"  检测异常: {e}")
        
        elapsed = int(time.time() - start)
        print(f"  等待中... ({elapsed}s/{max_wait}s)")
        time.sleep(check_interval)
    
    print(f"❌ 云端服务超时未恢复")
    return False

def run_eval(episodes, version, shared_state, run_idx=0, max_retries=10):
    """跑一轮评估，支持断点续跑和云端 OOM 自动恢复，返回 results 列表。"""
    done_ids, results = load_checkpoint(version, run_idx)
    if done_ids:
        print(f"  [断点续跑] 已完成 {len(done_ids)}/{len(episodes)} 个 episode，继续...")

    retry_count = 0
    i = 0
    while i < len(episodes):
        ep = episodes[i]
        ep_id = ep.get("episode_id", i)
        if ep_id in done_ids:
            i += 1
            continue  # 已完成，跳过
        
        print(f"  [{i+1}/{len(episodes)}] episode_id={ep_id} scene={ep['scene_id']} ... ", end="", flush=True)
        
        try:
            # 设置 SIGALRM 超时（Linux only）
            signal.signal(signal.SIGALRM, episode_timeout_handler)
            signal.alarm(EPISODE_TIMEOUT + 30)  # 比内部超时多 30 秒的硬性保障
            try:
                out = run_episode(ep, shared_state, version)
            finally:
                signal.alarm(0)  # 取消超时
        except EpisodeTimeoutError:
            print(f"⏰ 信号超时 ({EPISODE_TIMEOUT + 30}s)")
            out = {
                "success": False,
                "oracle_success": False,
                "SPL": 0.0,
                "NE": float('inf'),
                "geodesic": 0.0,
                "min_dist": float('inf'),
                "error": "signal_timeout"
            }
        except requests.exceptions.ConnectionError as e:
            # 云端连接失败，可能是 OOM 导致服务崩溃
            print(f"\n❌ 云端连接失败: {e}")
            save_checkpoint(version, run_idx, done_ids, results)
            if retry_count >= max_retries:
                print(f"❌ 重试次数已达上限 ({max_retries})，退出")
                raise
            retry_count += 1
            print(f"  [重试 {retry_count}/{max_retries}]")
            if wait_for_cloud_recovery():
                # 重置 shared_state，包括云端错误标志
                with shared_state.lock:
                    shared_state.latest_goal_uv = None
                    shared_state.latest_status = "Searching..."
                    shared_state.cloud_fatal_error = False
                    shared_state.cloud_error_count = 0
                continue  # 重试当前 episode
            else:
                raise RuntimeError("云端服务恢复超时")
        except (ConnectionError, Exception) as e:
            error_str = str(e).lower()
            # 检测是否是云端相关错误
            if 'connection' in error_str or 'timeout' in error_str or 'refused' in error_str or '云端' in str(e) or 'oom' in error_str:
                print(f"\n❌ 云端错误: {e}")
                save_checkpoint(version, run_idx, done_ids, results)
                if retry_count >= max_retries:
                    print(f"❌ 重试次数已达上限 ({max_retries})，退出")
                    raise
                retry_count += 1
                print(f"  [重试 {retry_count}/{max_retries}]")
                if wait_for_cloud_recovery():
                    with shared_state.lock:
                        shared_state.latest_goal_uv = None
                        shared_state.latest_status = "Searching..."
                        shared_state.cloud_fatal_error = False
                        shared_state.cloud_error_count = 0
                    continue
                else:
                    raise RuntimeError("云端服务恢复超时")
            else:
                print(f"异常: {e}")
                save_checkpoint(version, run_idx, done_ids, results)
                raise
        
        # 处理错误/超时情况
        if "error" in out:
            error_msg = out['error']
            if error_msg == "timeout":
                # 超时也记录为失败结果
                results.append({
                    "episode_id": ep_id,
                    "success": False,
                    "oracle_success": False,
                    "SPL": 0.0,
                    "NE": float('inf'),
                    "geodesic": out.get("geodesic", 0.0),
                    "min_dist": float('inf'),
                })
                print(f"SR=0 OSR=0 (超时)")
            else:
                print(f"跳过: {error_msg}")
            done_ids.add(ep_id)
            save_checkpoint(version, run_idx, done_ids, results)
            i += 1
            continue
        
        success = out["success"]
        oracle_success = out["oracle_success"]
        spl = out["SPL"]
        ne = out["NE"]
        geo = out["geodesic"]
        min_dist = out["min_dist"]
        
        # 计算轨迹长度
        trajectory = out.get("trajectory", [])
        if len(trajectory) >= 2:
            traj_len = sum(np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i])) 
                          for i in range(len(trajectory)-1))
        else:
            traj_len = 0.0
        
        results.append({
            "episode_id": ep_id,
            "scene_id": ep.get("scene_id", ""),
            "instruction": ep.get("instruction", ""),
            "success": bool(success),
            "oracle_success": bool(oracle_success),
            "SPL": float(spl),
            "NE": float(ne),
            "geodesic": float(geo),
            "min_dist": float(min_dist),
            "trajectory_length": float(traj_len),
            "steps": len(trajectory),
            "start_pos": [float(x) for x in out.get("start_pos", [0,0,0])],
            "gt_pos": [float(x) for x in out.get("gt_pos", [0,0,0])],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        done_ids.add(ep_id)
        save_checkpoint(version, run_idx, done_ids, results)
        
        # 保存成功案例的俯视轨迹图
        if success and SAVE_SUCCESS_TOPDOWN:
            save_topdown_trajectory(out, version, run_idx, ep_id)
        
        print(f"SR={int(success)} OSR={int(oracle_success)} SPL={spl:.3f} NE={ne:.2f}m min_d={min_dist:.2f}m")
        
        # 成功完成一个 episode，重置重试计数
        retry_count = 0
        i += 1

    # 本轮全部完成，清理 checkpoint
    clear_checkpoint(version, run_idx)
    return results

def summarize_results(results, version_label):
    if not results:
        return None
    n = len(results)
    sr = sum(r["success"] for r in results) / n
    osr = sum(r.get("oracle_success", r["success"]) for r in results) / n  # OSR
    spl_mean = sum(r["SPL"] for r in results) / n
    ne_mean = sum(r["NE"] for r in results) / n
    
    # 收集成功和 OSR 成功的 episode IDs
    success_eps = [r["episode_id"] for r in results if r["success"]]
    osr_success_eps = [r["episode_id"] for r in results if r.get("oracle_success", r["success"])]
    
    return {
        "version": version_label,
        "n": n,
        "SR": sr,
        "OSR": osr,
        "SPL": spl_mean,
        "NE": ne_mean,
        "success_count": sum(r["success"] for r in results),
        "oracle_success_count": sum(r.get("oracle_success", r["success"]) for r in results),
        "success_episodes": success_eps,
        "oracle_success_episodes": osr_success_eps,
    }


def save_detailed_results(results, version, run_idx, summary):
    """保存详细的评估结果到 JSON 文件，便于后续分析"""
    if not os.path.exists(DETAILED_RESULTS_DIR):
        os.makedirs(DETAILED_RESULTS_DIR)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 计算额外统计
    success_eps = [r for r in results if r["success"]]
    failed_eps = [r for r in results if not r["success"]]
    
    avg_ne_success = np.mean([r["NE"] for r in success_eps]) if success_eps else 0.0
    avg_ne_failed = np.mean([r["NE"] for r in failed_eps]) if failed_eps else 0.0
    avg_steps_success = np.mean([r["steps"] for r in success_eps]) if success_eps else 0.0
    avg_steps_failed = np.mean([r["steps"] for r in failed_eps]) if failed_eps else 0.0
    
    # 保存详细结果
    detailed = {
        "version": version,
        "run_idx": run_idx,
        "timestamp": timestamp,
        "config": {
            "success_distance": EVAL_SUCCESS_DISTANCE,
            "max_steps": EVAL_MAX_STEPS,
            "episode_timeout": EPISODE_TIMEOUT,
        },
        "summary": {
            "total_episodes": summary["n"],
            "success_count": summary["success_count"],
            "oracle_success_count": summary["oracle_success_count"],
            "SR": summary["SR"],
            "OSR": summary["OSR"],
            "SPL": summary["SPL"],
            "NE": summary["NE"],
            "success_episodes": summary["success_episodes"],
            "oracle_success_episodes": summary["oracle_success_episodes"],
        },
        "analysis": {
            "avg_NE_success": float(avg_ne_success),
            "avg_NE_failed": float(avg_ne_failed),
            "avg_steps_success": float(avg_steps_success),
            "avg_steps_failed": float(avg_steps_failed),
        },
        "episodes": results,
    }
    
    filename = f"{version}_run{run_idx}_{timestamp}.json"
    filepath = os.path.join(DETAILED_RESULTS_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    
    print(f"  详细结果已保存: {filepath}")
    
    # 同时保存一个 "latest" 版本，方便快速查看
    latest_filepath = os.path.join(DETAILED_RESULTS_DIR, f"{version}_latest.json")
    with open(latest_filepath, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    
    return filepath


def generate_comparison_report(detailed_results_dir=DETAILED_RESULTS_DIR):
    """生成各版本对比分析报告（读取 _latest.json）"""
    if not os.path.exists(detailed_results_dir):
        return
    
    versions_data = {}
    for ver in ["v1", "v2", "v3"]:
        latest_path = os.path.join(detailed_results_dir, f"{ver}_latest.json")
        if os.path.exists(latest_path):
            try:
                with open(latest_path, "r", encoding="utf-8") as f:
                    versions_data[ver] = json.load(f)
            except Exception:
                pass
    
    if len(versions_data) < 2:
        return
    
    # 生成对比报告
    report_lines = [
        "",
        "=" * 70,
        "版本对比分析报告",
        "=" * 70,
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 70,
    ]
    
    # 汇总表格
    report_lines.append("\n【汇总指标】")
    report_lines.append(f"{'版本':<6} {'SR':>8} {'OSR':>8} {'SPL':>8} {'NE':>8} {'成功数':>8}")
    report_lines.append("-" * 50)
    for ver, data in sorted(versions_data.items()):
        s = data["summary"]
        report_lines.append(f"{ver:<6} {s['SR']:>8.4f} {s['OSR']:>8.4f} {s['SPL']:>8.4f} {s['NE']:>8.3f} {s['success_count']:>8}")
    
    # 成功 episode 对比
    report_lines.append("\n【成功 Episode 对比】")
    all_success_sets = {}
    for ver, data in versions_data.items():
        all_success_sets[ver] = set(data["summary"]["success_episodes"])
        report_lines.append(f"{ver}: {sorted(data['summary']['success_episodes'])}")
    
    # 计算重叠
    if len(all_success_sets) >= 2:
        vers = list(all_success_sets.keys())
        report_lines.append("\n【成功重叠分析】")
        for i in range(len(vers)):
            for j in range(i+1, len(vers)):
                v1, v2 = vers[i], vers[j]
                overlap = all_success_sets[v1] & all_success_sets[v2]
                only_v1 = all_success_sets[v1] - all_success_sets[v2]
                only_v2 = all_success_sets[v2] - all_success_sets[v1]
                report_lines.append(f"  {v1} ∩ {v2}: {len(overlap)} 个重叠")
                report_lines.append(f"    仅 {v1} 成功: {sorted(only_v1)}")
                report_lines.append(f"    仅 {v2} 成功: {sorted(only_v2)}")
    
    # 三版本共同成功
    if len(all_success_sets) == 3:
        common = all_success_sets["v1"] & all_success_sets["v2"] & all_success_sets["v3"]
        report_lines.append(f"\n  三版本共同成功: {sorted(common)}")
    
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    report_path = os.path.join(detailed_results_dir, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n对比报告已保存: {report_path}")

# ============== 主评估循环（支持多次运行，报 mean±std）==============
def main():
    global EVAL_SUCCESS_DISTANCE
    parser = argparse.ArgumentParser(description="VLN 评估 v1/v2/v3，成功=导航过程中任意时刻与 gt 距离 < 阈值")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "all"], default="all",
                        help="评估版本: v1, v2, v3, 或 all (全部)")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="同一配置重复运行次数，报 mean±std，默认 1")
    parser.add_argument("--success_dist", type=float, default=None,
                        help="成功判定距离(m)，默认 2.0")
    args = parser.parse_args()
    if args.success_dist is not None:
        EVAL_SUCCESS_DISTANCE = args.success_dist

    if not os.path.exists(EVAL_EPISODES_JSON):
        print(f"未找到测试集: {EVAL_EPISODES_JSON}")
        sys.exit(1)
    with open(EVAL_EPISODES_JSON, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    if not episodes:
        print("eval_episodes.json 为空")
        sys.exit(1)

    n_runs = max(1, args.runs)
    print("=" * 60)
    print("VLN 自动化评估 (无头) - 论文对齐版")
    print("=" * 60)
    print(f"测试集: {EVAL_EPISODES_JSON}, 共 {len(episodes)} 个 episode")
    print(f"成功判定: 导航过程中任意时刻与 gt_position 距离 < {EVAL_SUCCESS_DISTANCE}m 即成功")
    print(f"最大步数: {EVAL_MAX_STEPS} 步")
    print(f"运行版本: {args.version}  |  重复次数: {n_runs}")
    if SAVE_SUCCESS_TOPDOWN:
        print(f"成功案例轨迹图保存至: {SUCCESS_TOPDOWN_DIR}")
    print("=" * 60)

    # 按版本收集多轮指标
    runs_SR = {"v1": [], "v2": [], "v3": []}
    runs_OSR = {"v1": [], "v2": [], "v3": []}
    runs_SPL = {"v1": [], "v2": [], "v3": []}
    runs_NE = {"v1": [], "v2": [], "v3": []}

    versions_to_run = []
    if args.version == "all":
        versions_to_run = ["v1", "v2", "v3"]
    else:
        versions_to_run = [args.version]

    for run in range(n_runs):
        print(f"\n========== 第 {run+1}/{n_runs} 轮 ==========")
        shared_state = SharedState()

        for ver in versions_to_run:
            print(f"\n--- 评估 {ver} ---")
            results = run_eval(episodes, ver, shared_state, run_idx=run)
            if results:
                s = summarize_results(results, ver)
                runs_SR[ver].append(s["SR"])
                runs_OSR[ver].append(s["OSR"])
                runs_SPL[ver].append(s["SPL"])
                runs_NE[ver].append(s["NE"])
                print(f"  {ver} 本轮: SR={s['SR']:.4f}  OSR={s['OSR']:.4f}  SPL={s['SPL']:.4f}  NE={s['NE']:.4f}m")
                print(f"  成功 episodes: {s['success_episodes']}")
                print(f"  OSR 成功 episodes: {s['oracle_success_episodes']}")
                # 保存详细结果到 JSON 文件
                save_detailed_results(results, ver, run, s)

    # 汇总 mean ± std
    def mean_std(arr):
        if not arr:
            return 0.0, 0.0
        a = np.array(arr)
        return float(np.mean(a)), float(np.std(a)) if len(a) > 1 else 0.0

    report_lines = [
        "",
        "=" * 70,
        "评估报表 (论文对齐版)",
        "=" * 70,
        f"成功判定: 导航过程中任意时刻与 gt 距离 < {EVAL_SUCCESS_DISTANCE}m 即成功",
        f"最大步数: {EVAL_MAX_STEPS}",
        f"Episodes: {len(episodes)}  |  重复轮数: {n_runs}",
        "-" * 70,
        "指标说明:",
        "  SR  = Success Rate (首次达到阈值即停止)",
        "  OSR = Oracle Success Rate (过程中曾到达过阈值)",
        "  SPL = Success weighted by Path Length",
        "  NE  = Navigation Error (最终位置到目标的距离)",
        "-" * 70,
    ]
    for ver in ["v1", "v2", "v3"]:
        if not runs_SR[ver]:
            continue
        m_sr, s_sr = mean_std(runs_SR[ver])
        m_osr, s_osr = mean_std(runs_OSR[ver])
        m_spl, s_spl = mean_std(runs_SPL[ver])
        m_ne, s_ne = mean_std(runs_NE[ver])
        if n_runs > 1:
            report_lines.append(f"  [{ver}] SR={m_sr:.4f}±{s_sr:.4f}  OSR={m_osr:.4f}±{s_osr:.4f}  SPL={m_spl:.4f}±{s_spl:.4f}  NE={m_ne:.3f}±{s_ne:.3f}m")
        else:
            report_lines.append(f"  [{ver}] SR={m_sr:.4f}  OSR={m_osr:.4f}  SPL={m_spl:.4f}  NE={m_ne:.3f}m")
    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(os.path.dirname(__file__), "eval_report.txt")
    # 追加模式：保留之前的结果，在下面添加新结果
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + report_text)
    print(f"\n报表已追加到: {report_path}")
    
    # 生成版本对比分析报告
    if len(versions_to_run) > 1:
        generate_comparison_report()

if __name__ == "__main__":
    main()
