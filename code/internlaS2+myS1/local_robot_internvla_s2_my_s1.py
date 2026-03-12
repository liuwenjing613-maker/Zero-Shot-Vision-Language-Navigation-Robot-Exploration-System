# ==============================================================================
# 文件名: local_robot_internvla_s2_my_s1.py
# 运行位置: 本地 Ubuntu（Habitat-sim）
# 功能: InternVLA S2 + 我的 S1 —— 视觉伺服闭环
#       S2 未看到目标输出左右箭头 → S1 执行旋转 → 新图 → 再推理；
#       S2 看到目标输出像素点 → S1 沿路径/直行靠近；S2 输出 STOP → 成功
# 云端: cloud_internvla_s2.py（同目录）
# ==============================================================================

import matplotlib
matplotlib.use("TkAgg")

import habitat_sim
import numpy as np
import cv2
import requests
import os
import sys
import magnum as mn
import matplotlib.pyplot as plt
import time
import threading
import math
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topdown_visualizer import TopdownVisualizer

# ============== 1. 配置 ==============
CLOUD_URL = os.environ.get("INTERNVLA_S2_CLOUD", "http://127.0.0.1:5000")
CLOUD_PLAN = CLOUD_URL.rstrip("/") + "/plan"
IMG_WIDTH, IMG_HEIGHT, HFOV = 640, 480, 110

SCENES_BASE = "/home/abc/ZeroShot_VLN/assets/scenes"
SCENE_LIST = [
    "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "MatterPort3D/mp3d/8194nk5LbLH/8194nk5LbLH.glb",
    "MatterPort3D/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb",
    "MatterPort3D/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb",
    "MatterPort3D/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
]
SCENE_INDEX = 0

INSTRUCTION = "find the bed in the bedroom"

# 及时反映云端输出：缩短请求间隔与主循环周期，转向指令下一两帧即执行
CLOUD_SEND_INTERVAL = 0.35     # 更频繁请求云端，获取新 action
MAIN_LOOP_SLEEP = 0.05         # 主循环 ~20Hz，收到 action 后尽快执行
MOVE_STEP_SIZE = 0.016         # 保持低速步长
WALK_FORWARD_EVERY_N = 2       # 无目标时每 N 帧前进一步
TURN_WHEN_DIRECTION_N = 2      # turn_left/turn_right 每 2 帧执行一次，及时反映云端
TURN_WHEN_SEARCHING_N = 6      # searching 时每 6 帧转一次，避免乱转
REPLAN_DISTANCE_THRESHOLD = 1.5
STUCK_DEPTH_THRESHOLD = 0.6
STUCK_FRAMES = 4
DEPTH_MIN, DEPTH_MAX = 0.3, 5.0
SUCCESS_DEPTH_THRESHOLD = 0.5

FBE_EXPLORE_RADIUS = 2.5
FBE_MIN_DISTANCE = 1.2
FBE_SEARCHING_FRAMES = 400
FBE_BLIND_TURN_FRAMES = 100
FBE_SAME_FLOOR_MAX_DY = 0.5
FBE_SAME_FLOOR_TRIES = 25
FBE_SMOOTH_TURN_FRAMES = 12
STUCK_SMOOTH_TURN_FRAMES = 4

MEMORY_DISTANCE_CHECK_INTERVAL = 10.0
MEMORY_DISTANCE_IMPROVE_THRESHOLD = 0.15
MEMORY_BLACKLIST_TOLERANCE = 0.5

USE_FIXED_START = True
FIXED_START_POSITION = [11.32, 3.53, 1.59]
GT_TARGET_POSITION = [6.45, 3.53, 1.99]
USE_VERIFY_BEFORE_SUCCESS = False

# ============== 2. 共享状态（适配 InternVLA S2 的 action 输出）==============
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_rgb = None
        self.current_depth = None
        self.current_agent_state = None
        self.new_image_ready = False
        # S2 输出: turn_left | turn_right | waypoint | stop | searching
        self.latest_action = "searching"
        self.latest_goal_uv = None
        self.latest_goal_depth = None
        self.latest_goal_camera_snapshot = None
        self.latest_status = "Searching..."
        self.current_instruction = INSTRUCTION
        self.target_list = []
        self.target_index = 0
        self.latest_instruction_used = None
        self.latest_reason = None
        self.task_stop_received = False  # 当前目标收到 STOP 时置 True

shared_state = SharedState()


def parse_instruction_sequence(instruction):
    instruction = instruction.strip()
    if not instruction:
        return ["red backpack"]
    targets = re.findall(r"(?:find|go\s+to)\s+(?:a\s+|an\s+|the\s+)?(\w+)", instruction, re.IGNORECASE)
    return [t.strip() for t in targets] if targets else [instruction]


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


def get_depth_at_uv(u, v, depth_img):
    u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
    v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
    patch = depth_img[max(0, v_idx - 2) : v_idx + 3, max(0, u_idx - 2) : u_idx + 3]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None
    return float(np.min(valid))


def get_agent_forward_yaw(agent_state):
    cam = agent_state.sensor_states["color_sensor"]
    q = cam.rotation
    mn_q = mn.Quaternion(mn.Vector3(q.imag), q.real)
    mat = mn_q.to_matrix()
    col2 = mat[2]
    fwd_x, fwd_z = -col2[0], -col2[2]
    return math.atan2(fwd_x, fwd_z)


def forward_step_position(agent_state, sim, step_size):
    yaw = get_agent_forward_yaw(agent_state)
    pos = np.array(agent_state.position)
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float32)
    next_pos = pos + fwd * step_size
    if sim.pathfinder.is_loaded:
        next_pos = sim.pathfinder.snap_point(next_pos)
    return next_pos


def get_3d_point(u, v, depth_img, agent_state, sim, camera_snapshot=None):
    u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
    v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
    patch = depth_img[max(0, v_idx - 2) : v_idx + 3, max(0, u_idx - 2) : u_idx + 3]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None, None, None
    z_depth = float(np.min(valid))
    if z_depth < DEPTH_MIN or z_depth > DEPTH_MAX:
        return None, None, None
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
            return snapped, z_depth, raw_3d
    return raw_3d, z_depth, raw_3d


def get_explore_waypoint(sim, curr_pos):
    if not sim.pathfinder.is_loaded:
        return None
    try:
        curr_pos = np.array(curr_pos, dtype=np.float32)
        start_snap = sim.pathfinder.snap_point(curr_pos)
        island_idx = sim.pathfinder.get_island(start_snap)
        curr_y = float(curr_pos[1])
        for _ in range(FBE_SAME_FLOOR_TRIES):
            pt = sim.pathfinder.get_random_navigable_point_near(
                start_snap, FBE_EXPLORE_RADIUS, island_index=island_idx
            )
            if pt is not None and not np.isnan(pt).any():
                if abs(float(pt[1]) - curr_y) > FBE_SAME_FLOOR_MAX_DY:
                    continue
                dist = np.linalg.norm(np.array(pt) - curr_pos)
                if dist >= FBE_MIN_DISTANCE:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = start_snap
                    path.requested_end = np.array(pt, dtype=np.float32)
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        return np.array(pt, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ FBE 探索点获取失败: {e}")
    return None


def is_goal_in_blacklist(goal_3d, blacklist):
    for bl in blacklist:
        if np.linalg.norm(np.array(goal_3d) - np.array(bl)) < MEMORY_BLACKLIST_TOLERANCE:
            return True
    return False


# ============== 3. 云端 worker（解析 S2 的 action）==============
def cloud_worker():
    last_send = 0.0
    while True:
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
            _, buf = cv2.imencode(".jpg", img_to_send)
            try:
                data = {"instruction": instruction}
                if shared_state.target_list:
                    data["target_list"] = json.dumps(shared_state.target_list)
                    data["target_index"] = str(shared_state.target_index)
                r = requests.post(CLOUD_PLAN, files={"image": ("img.jpg", buf.tobytes(), "image/jpeg")}, data=data, timeout=10)
                j = r.json()
                with shared_state.lock:
                    shared_state.latest_instruction_used = instruction
                    if j.get("reason"):
                        shared_state.latest_reason = j["reason"]
                        print(f"💭 S2: {j['reason']}")
                    action = j.get("action", "searching")
                    shared_state.latest_action = action
                    if action == "stop":
                        shared_state.task_stop_received = True
                        shared_state.latest_status = "STOP"
                        shared_state.latest_goal_uv = None
                    elif action == "turn_left":
                        shared_state.latest_status = "Turn left"
                        shared_state.latest_goal_uv = None
                    elif action == "turn_right":
                        shared_state.latest_status = "Turn right"
                        shared_state.latest_goal_uv = None
                    elif action == "waypoint" and "u" in j and "v" in j:
                        u, v = int(j["u"]), int(j["v"])
                        u = max(0, min(u, IMG_WIDTH - 1))
                        v = max(0, min(v, IMG_HEIGHT - 1))
                        shared_state.latest_goal_uv = (u, v)
                        shared_state.latest_goal_depth = depth_snap
                        shared_state.latest_goal_camera_snapshot = cam_snap
                        shared_state.latest_status = "Waypoint"
                    else:
                        shared_state.latest_status = "Searching..."
                        shared_state.latest_goal_uv = None
            except Exception as e:
                print(f"云端请求失败: {e}")
        time.sleep(0.3)


def main():
    global INSTRUCTION
    target_list = parse_instruction_sequence(INSTRUCTION)
    with shared_state.lock:
        shared_state.target_list = target_list
        shared_state.target_index = 0
        shared_state.current_instruction = target_list[0]

    scene_path = None
    for i in range(len(SCENE_LIST)):
        p = os.path.join(SCENES_BASE, SCENE_LIST[(SCENE_INDEX + i) % len(SCENE_LIST)])
        if os.path.exists(p):
            scene_path = p
            break
    if scene_path is None:
        print(f"❌ 无可用场景: {SCENES_BASE}")
        return
    print(f"📍 场景: {scene_path}")
    print(f"🎯 目标: {target_list}")
    print(f"☁️  云端: {CLOUD_PLAN} (InternVLA S2)")

    sim = habitat_sim.Simulator(make_cfg(scene_path))
    nav = habitat_sim.NavMeshSettings()
    nav.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, nav)
    agent = sim.initialize_agent(0)

    if USE_FIXED_START and FIXED_START_POSITION is not None:
        start_pos = np.array(FIXED_START_POSITION, dtype=np.float32)
        if sim.pathfinder.is_loaded:
            start_pos = sim.pathfinder.snap_point(start_pos)
            if np.isnan(start_pos).any():
                start_pos = sim.pathfinder.get_random_navigable_point()
        print(f"📍 起点: {list(start_pos)}")
    else:
        start_pos = sim.pathfinder.get_random_navigable_point()
    s = agent.get_state()
    s.position = start_pos
    agent.set_state(s)

    threading.Thread(target=cloud_worker, daemon=True).start()
    print("✅ 云端 worker 已启动 (InternVLA S2: 箭头/像素/STOP)")

    cv2.namedWindow("Robot Eye", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Eye", 640, 480)
    print("📍 正在生成场景俯视图...")
    topdown_viz = TopdownVisualizer(sim, meters_per_pixel=0.05)
    fig, ax = topdown_viz.init_matplotlib_figure(figsize=(10, 10))
    plt.ion()
    plt.show(block=False)

    path_points = []
    current_goal_3d = None
    current_goal_3d_raw = None
    goal_blacklist = []
    goal_3d_recorded_at = None
    goal_3d_recorded_dist = None
    step_count = 0
    trajectory = [[start_pos[0], start_pos[2]]]
    searching_frames = 0
    blind_turn_frames = 0
    smooth_turn_remaining = 0
    in_fbe_mode = False
    current_target_idx = 0
    stuck_position_frames = 0
    stuck_position_last_pos = None
    stuck_position_threshold = 0.3
    stuck_position_check_interval = 100
    STUCK_FBE_TRIGGER_FRAMES = 300

    print("⏳ 3 秒后开始导航...")
    for _ in range(60):
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3][..., ::-1]
        viz = rgb.copy()
        cv2.putText(viz, "Starting in 3s...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(viz, f"Goal: {target_list[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Robot Eye", viz)
        if cv2.waitKey(50) == ord("q"):
            sim.close()
            cv2.destroyAllWindows()
            return
        time.sleep(0.05)

    try:
        while True:
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][:, :, :3][..., ::-1]
            depth = obs["depth_sensor"]
            curr_state = agent.get_state()
            curr_pos = np.array(curr_state.position)

            with shared_state.lock:
                shared_state.current_rgb = rgb
                shared_state.current_depth = depth
                shared_state.current_agent_state = curr_state
                shared_state.new_image_ready = True
                action = shared_state.latest_action
                uv = shared_state.latest_goal_uv
                status_display = shared_state.latest_status
                snap_d = shared_state.latest_goal_depth
                snap_c = shared_state.latest_goal_camera_snapshot
                latest_instruction_used = shared_state.latest_instruction_used
                task_stop = shared_state.task_stop_received

            trajectory.append([curr_pos[0], curr_pos[2]])

            front_d = 2.0
            center = depth[IMG_HEIGHT // 2 - 30 : IMG_HEIGHT // 2 + 30, IMG_WIDTH // 2 - 30 : IMG_WIDTH // 2 + 30]
            valid = center[(center > 0.1) & (center < 5.0)]
            if len(valid) > 0:
                front_d = float(np.median(valid))
            front_clear = front_d >= 0.85

            # ----- 成功判定：S2 输出 STOP -----
            current_target_name = target_list[current_target_idx] if current_target_idx < len(target_list) else None
            if task_stop and current_target_name and latest_instruction_used == current_target_name:
                with shared_state.lock:
                    shared_state.task_stop_received = False
                print("\n" + "★" * 20 + f"\n🎯 目标 {current_target_name} 达成 (S2 STOP)!\n" + "★" * 20)
                goal_blacklist = []
                goal_3d_recorded_at = None
                current_target_idx += 1
                if current_target_idx >= len(target_list):
                    print("🎉 全部目标完成!")
                    plt.savefig("vln_success.png")
                    time.sleep(2)
                    break
                with shared_state.lock:
                    shared_state.target_index = current_target_idx
                    shared_state.current_instruction = target_list[current_target_idx]
                    shared_state.latest_goal_uv = None
                    shared_state.latest_action = "searching"
                    shared_state.latest_status = "Searching..."
                path_points = []
                current_goal_3d = None
                current_goal_3d_raw = None
                time.sleep(0.5)

            # ----- 深度足够近时也允许记一次 STOP（可选）-----
            if action == "waypoint" and uv is not None and current_target_name and latest_instruction_used == current_target_name:
                d_at = get_depth_at_uv(uv[0], uv[1], depth)
                if d_at is not None and d_at < SUCCESS_DEPTH_THRESHOLD:
                    with shared_state.lock:
                        shared_state.task_stop_received = True

            # ----- 撞墙 / FBE -----
            stuck_count = 0
            if front_d < STUCK_DEPTH_THRESHOLD:
                stuck_count = 1
            if stuck_count >= STUCK_FRAMES:
                path_points = []
                current_goal_3d = None
                current_goal_3d_raw = None
                explore_waypoint = get_explore_waypoint(sim, curr_pos)
                if explore_waypoint is not None:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = sim.pathfinder.snap_point(curr_pos)
                    path.requested_end = explore_waypoint
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        path_points = list(path.points)
                        in_fbe_mode = True
                        blind_turn_frames = 0
                else:
                    smooth_turn_remaining = STUCK_SMOOTH_TURN_FRAMES
                    blind_turn_frames += 1

            if in_fbe_mode and path_points:
                dist_to_wp = np.linalg.norm(curr_pos - path_points[-1])
                if dist_to_wp < 0.3:
                    in_fbe_mode = False
                    path_points = []
                    smooth_turn_remaining = FBE_SMOOTH_TURN_FRAMES
                    stuck_position_frames = 0

            use_goal = uv and snap_d is not None and snap_c is not None
            if action == "waypoint" and not in_fbe_mode and use_goal and (not path_points or len(path_points) < 2):
                goal_3d, gdepth, goal_3d_raw = get_3d_point(uv[0], uv[1], snap_d, curr_state, sim, camera_snapshot=snap_c)
                if goal_3d is None and snap_d is not None:
                    goal_3d, gdepth, goal_3d_raw = get_3d_point(uv[0], uv[1], depth, curr_state, sim, camera_snapshot=None)
                if goal_3d is not None and not is_goal_in_blacklist(goal_3d, goal_blacklist):
                    replan = current_goal_3d is None or np.linalg.norm(goal_3d - current_goal_3d) > REPLAN_DISTANCE_THRESHOLD
                    if replan:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(curr_pos)
                        path.requested_end = goal_3d
                        if sim.pathfinder.find_path(path) and len(path.points) > 0:
                            current_goal_3d = goal_3d
                            current_goal_3d_raw = goal_3d_raw if goal_3d_raw is not None else goal_3d
                            path_points = list(path.points)
                            goal_3d_recorded_at = time.time()
                            goal_3d_recorded_dist = np.linalg.norm(curr_pos - goal_3d)

            if action in ("turn_left", "turn_right", "searching"):
                if not in_fbe_mode:
                    path_points = []
                    current_goal_3d = None
                    current_goal_3d_raw = None
                if action == "searching":
                    searching_frames += 1
                else:
                    searching_frames = 0

            if action == "searching" and not in_fbe_mode and searching_frames >= FBE_SEARCHING_FRAMES and current_goal_3d is None:
                explore_waypoint = get_explore_waypoint(sim, curr_pos)
                if explore_waypoint is not None:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = sim.pathfinder.snap_point(curr_pos)
                    path.requested_end = explore_waypoint
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        path_points = list(path.points)
                        in_fbe_mode = True
                        searching_frames = 0

            if step_count % stuck_position_check_interval == 0:
                if stuck_position_last_pos is not None:
                    dist_moved = np.linalg.norm(curr_pos - stuck_position_last_pos)
                    if dist_moved < stuck_position_threshold:
                        stuck_position_frames += stuck_position_check_interval
                    else:
                        stuck_position_frames = 0
                stuck_position_last_pos = curr_pos.copy()
            if stuck_position_frames >= STUCK_FBE_TRIGGER_FRAMES and not in_fbe_mode and current_goal_3d is None:
                explore_waypoint = get_explore_waypoint(sim, curr_pos)
                if explore_waypoint is not None:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = sim.pathfinder.snap_point(curr_pos)
                    path.requested_end = explore_waypoint
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        path_points = list(path.points)
                        in_fbe_mode = True
                        stuck_position_frames = 0
                        searching_frames = 0

            if current_goal_3d is not None and goal_3d_recorded_at is not None:
                elapsed = time.time() - goal_3d_recorded_at
                if elapsed >= MEMORY_DISTANCE_CHECK_INTERVAL:
                    curr_dist = np.linalg.norm(curr_pos - current_goal_3d)
                    if goal_3d_recorded_dist is not None:
                        if goal_3d_recorded_dist - curr_dist < MEMORY_DISTANCE_IMPROVE_THRESHOLD:
                            goal_blacklist.append(np.array(current_goal_3d, dtype=np.float32))
                            path_points = []
                            current_goal_3d = None
                            current_goal_3d_raw = None
                            goal_3d_recorded_at = None
                        else:
                            goal_3d_recorded_at = time.time()
                            goal_3d_recorded_dist = curr_dist

            # ========== 执行控制：有 A* 路径则沿路径走，否则一直低速行走（慢速前进+慢速转向）==========
            if smooth_turn_remaining > 0:
                agent.act("turn_right")
                smooth_turn_remaining -= 1
            elif path_points:
                # 仅当模型输出目标像素后：A* 已在上方设好 path_points，这里沿路径低速前进
                next_pt = path_points[0] if len(path_points) == 1 else path_points[1]
                move_vec = next_pt - curr_pos
                d = np.linalg.norm(move_vec)
                if d < 0.1:
                    path_points.pop(0)
                else:
                    step = curr_pos + (move_vec / d) * MOVE_STEP_SIZE
                    s = agent.get_state()
                    s.position = step
                    agent.set_state(s)
            else:
                # 无目标像素：低速行走，并对云端 turn_left/turn_right 及时反映
                if step_count % WALK_FORWARD_EVERY_N == 0 and front_clear:
                    step_pos = forward_step_position(curr_state, sim, MOVE_STEP_SIZE)
                    s = agent.get_state()
                    s.position = step_pos
                    agent.set_state(s)
                elif action == "turn_left" and step_count % TURN_WHEN_DIRECTION_N == 0:
                    agent.act("turn_left")
                elif action == "turn_right" and step_count % TURN_WHEN_DIRECTION_N == 0:
                    agent.act("turn_right")
                elif action == "searching" and step_count % TURN_WHEN_SEARCHING_N == 0:
                    agent.act("turn_right")

            current_yaw = get_agent_forward_yaw(curr_state)
            topdown_viz.update_matplotlib(
                trajectory=trajectory,
                path_points=path_points,
                current_pos=curr_pos,
                goal_pos=current_goal_3d_raw if current_goal_3d_raw is not None else current_goal_3d,
                start_pos=start_pos,
                current_yaw=current_yaw,
                title=status_display,
                instruction=target_list[current_target_idx] if current_target_idx < len(target_list) else INSTRUCTION,
                status=status_display,
                fbe_point=None,
                gt_pos=GT_TARGET_POSITION,
            )

            viz = rgb.copy()
            if uv:
                cv2.circle(viz, uv, 12, (0, 255, 0), 2)
            cv2.putText(viz, status_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(viz, f"Goal: {target_list[current_target_idx] if current_target_idx < len(target_list) else INSTRUCTION}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Robot Eye", viz)
            if cv2.waitKey(20) == ord("q"):
                break
            step_count += 1
            time.sleep(MAIN_LOOP_SLEEP)
    finally:
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
