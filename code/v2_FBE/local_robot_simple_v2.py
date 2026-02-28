# ==============================================================================
# 文件名: local_robot_simple_v2.py
# 运行位置: 本地 Ubuntu 机器 (已安装 habitat-sim)
# 功能: VLN 导航 v2 - 在 v1 基础上加入 FBE、短期记忆、多目标序列
# 改进: 1.FBE前沿探索 2.短期记忆防死锁 3.多目标连续导航
# ==============================================================================

import matplotlib
matplotlib.use('TkAgg')

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

# 添加 code 目录到 path，导入俯视图可视化器
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topdown_visualizer import TopdownVisualizer

# ============== 1. 配置 ==============
CLOUD_URL = "http://127.0.0.1:5000/plan"
IMG_WIDTH, IMG_HEIGHT, HFOV = 640, 480, 110

SCENES_BASE = "/home/abc/ZeroShot_VLN/assets/scenes"
SCENE_LIST = [
    "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "MatterPort3D/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    "MatterPort3D/mp3d/8194nk5LbLH/8194nk5LbLH.glb",
    "MatterPort3D/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb",
    "MatterPort3D/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb",
    "MatterPort3D/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb",
    "MatterPort3D/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
    "MatterPort3D/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
    "MatterPort3D/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
    "MatterPort3D/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb",
]
SCENE_INDEX = 0

# 支持单目标或多目标序列，如 "First find a chair, then find a plant, finally go to the door"
#INSTRUCTION = "First find a chair, then find a plant, finally go to the door"
#INSTRUCTION = "the first plant you see"
INSTRUCTION = "fireplace"

# 核心参数
CLOUD_SEND_INTERVAL = 0.6  # 推理间隔（秒），越小越实时但云端负载越高
MOVE_STEP_SIZE = 0.06
REPLAN_DISTANCE_THRESHOLD = 1.5
STUCK_DEPTH_THRESHOLD = 0.6
STUCK_FRAMES = 4
DEPTH_MIN, DEPTH_MAX = 0.3, 5.0
TURN_INTERVAL = 12
INFER_SMOOTH_ALPHA = 0.25
INFER_TURN_COMMIT_FRAMES = 15
INFER_DEAD_ZONE = 100
SUCCESS_DEPTH_THRESHOLD = 0.6
TURN_ANGLE_THRESHOLD = 0.15

# v2 新增：FBE 前沿探索（提高触发门槛，避免过于频繁）
FBE_EXPLORE_RADIUS = 2.5       # 探索点距离当前位置 2-3 米
FBE_MIN_DISTANCE = 1.2         # 探索点至少距当前位置 1.2m
FBE_SEARCHING_FRAMES = 350     # Searching 连续 N 帧后触发 FBE（约 5s，原 80 约 1.6s 太易触发）
FBE_BLIND_TURN_FRAMES = 60     # 撞墙后盲目转向 N 帧后触发 FBE

# v2 新增：短期记忆防死锁 (Think, Remember, Navigate)
MEMORY_DISTANCE_CHECK_INTERVAL = 5.0   # 5 秒内检查距离是否缩短
MEMORY_DISTANCE_IMPROVE_THRESHOLD = 0.15  # 至少缩短 0.15m 才算有进展
MEMORY_BLACKLIST_TOLERANCE = 0.5      # 黑名单内距离此范围内的目标视为已黑名单

# 出生地：为 True 时使用下方坐标作为起点（会 snap 到 navmesh），否则随机
#USE_FIXED_START = False
USE_FIXED_START = True
FIXED_START_POSITION =  [
      0.47,
      0.13,
      4.71
    ] # [x, y, z]，按场景修改

# 成功前二次验证：为 True 时在“深度够近”后再请求云端 verify，避免红桌子当红衣服等误判
USE_VERIFY_BEFORE_SUCCESS = True

# FBE 视野连贯：到达探索点/撞墙后的环视改为每帧转一步，不突变
FBE_SMOOTH_TURN_FRAMES = 12   # 到达探索点后分多少帧转完（每帧 1 次 turn_right）
STUCK_SMOOTH_TURN_FRAMES = 4  # 撞墙无探索点时分多少帧转完

# ============== 2. 共享状态 ==============
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
        self.current_instruction = INSTRUCTION  # 当前搜索目标（多目标时动态更新）
        self.target_list = []  # 多目标列表
        self.target_index = 0
        self.latest_instruction_used = None  # 最近一次云端请求用的指令，用于多目标成功判定
        self.first_request_done = {}  # 记录每个目标是否已发送过首次请求（含推理原因）
        self.latest_reason = None  # 最近一次推理原因

shared_state = SharedState()

# ============== 3. 多目标序列解析 ==============
def parse_instruction_sequence(instruction):
    """解析 'First find X, then Y, finally Z' -> ['X','Y','Z']"""
    instruction = instruction.strip()
    if not instruction:
        return ["red backpack"]
    targets = re.findall(
        r'(?:find|go\s+to)\s+(?:a\s+|an\s+|the\s+)?(\w+)',
        instruction, re.IGNORECASE
    )
    return [t.strip() for t in targets] if targets else [instruction]

# ============== 4. Habitat 配置 ==============
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

# ============== 5. 工具函数 ==============
def get_depth_at_uv(u, v, depth_img):
    u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
    v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
    patch = depth_img[max(0, v_idx-2):v_idx+3, max(0, u_idx-2):u_idx+3]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))

def get_agent_forward_yaw(agent_state):
    cam = agent_state.sensor_states["color_sensor"]
    q = cam.rotation
    mn_q = mn.Quaternion(mn.Vector3(q.imag), q.real)
    mat = mn_q.to_matrix()
    col2 = mat[2]
    fwd_x, fwd_z = -col2[0], -col2[2]
    return math.atan2(fwd_x, fwd_z)

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

# ============== 6. FBE 前沿探索：获取 2-3 米外的随机可达点 ==============
def get_explore_waypoint(sim, curr_pos):
    """使用 pathfinder 获取距离当前位置 2-3m 的随机可达点作为探索点 (主动探索)"""
    if not sim.pathfinder.is_loaded:
        return None
    try:
        start_snap = sim.pathfinder.snap_point(np.array(curr_pos, dtype=np.float32))
        island_idx = sim.pathfinder.get_island(start_snap)
        for _ in range(10):
            # Habitat: get_random_navigable_point_near(center, radius) 或 (center, radius, island_index=)
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
    except Exception as e:
        print(f"⚠️ FBE 探索点获取失败: {e}")
    return None

# ============== 7. 目标黑名单检查 ==============
def is_goal_in_blacklist(goal_3d, blacklist):
    for bl in blacklist:
        if np.linalg.norm(np.array(goal_3d) - np.array(bl)) < MEMORY_BLACKLIST_TOLERANCE:
            return True
    return False

# ============== 8. 云端 worker ==============
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
            _, buf = cv2.imencode('.jpg', img_to_send)
            try:
                data = {'instruction': instruction}
                if shared_state.target_list:
                    data['target_list'] = json.dumps(shared_state.target_list)
                    data['target_index'] = str(shared_state.target_index)
                # 首次请求该目标时要求输出推理原因
                is_first = instruction not in shared_state.first_request_done
                if is_first:
                    data['first_request'] = '1'
                r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                  data=data, timeout=8)
                j = r.json()
                with shared_state.lock:
                    shared_state.latest_instruction_used = instruction
                    # 记录推理原因（首次请求有）
                    if 'reason' in j and j['reason']:
                        shared_state.latest_reason = j['reason']
                        print(f"💭 推理原因: {j['reason']}")
                    # 标记该目标已完成首次请求
                    if is_first:
                        shared_state.first_request_done[instruction] = True
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
                print(f"云端请求失败: {e}")
        time.sleep(0.3)

# ============== 9. 主函数 ==============
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
        print(f"❌ 无可用场景，请检查 {SCENES_BASE}")
        return
    print(f"📍 场景: {scene_path}")
    print(f"🎯 目标序列: {target_list}")

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
                print("⚠️ 自定义出生地不在 navmesh 上，改用随机点")
                start_pos = sim.pathfinder.get_random_navigable_point()
        print(f"📍 起点: 自定义 {FIXED_START_POSITION} → {list(start_pos)}")
    else:
        start_pos = sim.pathfinder.get_random_navigable_point()
        print(f"📍 起点: {start_pos}")
    s = agent.get_state()
    s.position = start_pos
    agent.set_state(s)

    threading.Thread(target=cloud_worker, daemon=True).start()
    print("✅ 云端 worker 已启动 (v2: FBE + 短期记忆 + 多目标)")

    cv2.namedWindow("Robot Eye", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Eye", 640, 480)
    
    # 初始化场景俯视图可视化器（替代原有二维坐标地图）
    print("📍 正在生成场景俯视图...")
    topdown_viz = TopdownVisualizer(sim, meters_per_pixel=0.05)
    fig, ax = topdown_viz.init_matplotlib_figure(figsize=(10, 10))
    plt.ion()
    plt.show(block=False)
    print("✅ 场景俯视图已就绪")

    path_points = []
    current_goal_3d = None
    goal_set_at = None
    stuck_count = 0
    step_count = 0
    smoothed_offset = None
    turn_commit_remaining = 0
    last_turn_direction = None
    trajectory = [[start_pos[0], start_pos[2]]]

    # v2: 短期记忆防死锁
    goal_3d_recorded_at = None
    goal_3d_recorded_dist = None
    goal_blacklist = []

    # v2: FBE 前沿探索（平滑转向，避免视野突变）
    searching_frames = 0
    blind_turn_frames = 0
    explore_waypoint = None
    in_fbe_mode = False
    
    # 位置卡住检测：即使状态是 Inferred，如果位置长时间不变也触发 FBE
    stuck_position_check_interval = 100  # 每 100 帧检查一次
    stuck_position_threshold = 0.3       # 位置移动小于 0.3m 视为卡住
    stuck_position_frames = 0            # 连续卡住帧数
    stuck_position_last_pos = None       # 上次检查的位置
    STUCK_FBE_TRIGGER_FRAMES = 300       # 连续卡住 300 帧触发 FBE（约 6 秒）
    smooth_turn_remaining = 0  # >0 时每帧只转一步，不跟路径、不突变

    current_target_idx = 0

    print("🚀 开始导航 v2 (FBE+短期记忆+多目标, 按 q 退出)...")
    
    # 先显示初始画面，等待 6 秒后开始导航
    print("⏳ 窗口已就绪，导航将在 6 秒后开始...")
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3][..., ::-1]
    curr_state = agent.get_state()
    curr_pos = np.array(curr_state.position)
    # 显示初始 Robot Eye
    viz = rgb.copy()
    cv2.putText(viz, "Starting in 6s...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(viz, f"Goal: {target_list[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Robot Eye", viz)
    # 显示初始俯视图
    topdown_viz.update_matplotlib(
        trajectory=[[start_pos[0], start_pos[2]]],
        current_pos=curr_pos,
        start_pos=start_pos,
        current_yaw=get_agent_forward_yaw(curr_state),
        title="Starting...",
        instruction=target_list[0],
        status="Waiting",
        gt_pos=GT_TARGET_POSITION
    )
    cv2.waitKey(1)
    time.sleep(6)
    print("🚀 导航开始")
    
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
                uv = shared_state.latest_goal_uv
                status = shared_state.latest_status
                snap_d = shared_state.latest_goal_depth
                snap_c = shared_state.latest_goal_camera_snapshot
                latest_instruction_used = shared_state.latest_instruction_used

            trajectory.append([curr_pos[0], curr_pos[2]])

            # 优先：FBE 过程中一旦目标锁定，立刻停止 FBE 行走并清空探索路径，本帧后续会规划到目标
            use_goal = uv and snap_d is not None and snap_c is not None
            if in_fbe_mode and status == "Target Locked" and use_goal:
                in_fbe_mode = False
                path_points = []
                stuck_position_frames = 0  # 重置卡住计数
                print("🔍 FBE 中断: 已锁定目标，转向目标")

            # 撞墙检测
            center = depth[IMG_HEIGHT//2-30:IMG_HEIGHT//2+30, IMG_WIDTH//2-30:IMG_WIDTH//2+30]
            valid = center[(center > 0.1) & (center < 5.0)]
            front_d = float(np.median(valid)) if len(valid) > 0 else 2.0
            if front_d < STUCK_DEPTH_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0

            # v2: 撞墙时用 FBE 替代盲目旋转
            if stuck_count >= STUCK_FRAMES:
                path_points = []
                current_goal_3d = None
                stuck_count = 0
                with shared_state.lock:
                    shared_state.latest_goal_uv = None
                    shared_state.latest_goal_depth = None
                    shared_state.latest_goal_camera_snapshot = None

                explore_waypoint = get_explore_waypoint(sim, curr_pos)
                if explore_waypoint is not None:
                    print("🔍 FBE: 前往探索点 (替代盲目旋转)")
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

            has_path = path_points and len(path_points) > 1

            # FBE 模式：到达探索点后退出，用多帧平滑环视（不突变）
            if in_fbe_mode and path_points:
                dist_to_waypoint = np.linalg.norm(curr_pos - path_points[-1])
                if dist_to_waypoint < 0.3:
                    in_fbe_mode = False
                    path_points = []
                    smooth_turn_remaining = FBE_SMOOTH_TURN_FRAMES
                    stuck_position_frames = 0  # 重置卡住计数
                    print("🔍 FBE: 已到达探索点，平滑环视")

            # 双模态成功判定（多目标：仅当本次 Locked 对应当前目标且深度足够才计为成功）
            current_target_name = target_list[current_target_idx] if current_target_idx < len(target_list) else None
            if (status == "Target Locked" and uv is not None and snap_d is not None
                    and current_target_name is not None and latest_instruction_used == current_target_name):
                d_at_target = get_depth_at_uv(uv[0], uv[1], snap_d)
                if d_at_target is not None and d_at_target < SUCCESS_DEPTH_THRESHOLD:
                    verified = True
                    if USE_VERIFY_BEFORE_SUCCESS:
                        try:
                            _, buf = cv2.imencode('.jpg', rgb)
                            r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                              data={'instruction': current_target_name, 'verify': '1'}, timeout=8)
                            verified = r.json().get('status') == 'success'
                            if not verified:
                                print("⚠️ 成功前验证未通过（非精确目标），不计成功")
                                with shared_state.lock:
                                    shared_state.latest_goal_uv = None
                                    shared_state.latest_goal_depth = None
                                    shared_state.latest_goal_camera_snapshot = None
                                    shared_state.latest_status = "Searching..."
                        except Exception as e:
                            print(f"⚠️ 成功前验证请求失败: {e}，按未通过处理")
                            verified = False
                            with shared_state.lock:
                                shared_state.latest_status = "Searching..."
                    if verified:
                        print("\n" + "★" * 20 + f"\n🎯 目标 {current_target_name} 达成!\n" + "★" * 20)
                        goal_blacklist = []
                        goal_3d_recorded_at = None
                        current_target_idx += 1
                        if current_target_idx >= len(target_list):
                            print("🎉 多目标序列全部完成!")
                            plt.savefig("vln_success.png")
                            time.sleep(2)
                            break
                        with shared_state.lock:
                            shared_state.target_index = current_target_idx
                            shared_state.current_instruction = target_list[current_target_idx]
                            shared_state.latest_goal_uv = None
                            shared_state.latest_goal_depth = None
                            shared_state.latest_goal_camera_snapshot = None
                            shared_state.latest_status = "Searching..."
                        path_points = []
                        current_goal_3d = None
                        time.sleep(0.5)

            # 短期记忆防死锁：5 秒内距离未缩短则放弃并黑名单
            if current_goal_3d is not None and goal_3d_recorded_at is not None:
                elapsed = time.time() - goal_3d_recorded_at
                if elapsed >= MEMORY_DISTANCE_CHECK_INTERVAL:
                    curr_dist = np.linalg.norm(curr_pos - current_goal_3d)
                    if goal_3d_recorded_dist is not None:
                        improved = goal_3d_recorded_dist - curr_dist
                        if improved < MEMORY_DISTANCE_IMPROVE_THRESHOLD:
                            print("⚠️ 短期记忆: 5秒内距离未缩短，放弃当前目标并加入黑名单")
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

            # 规划逻辑（FBE 中断后 has_path 已为 False，此处立刻对目标做 A*）
            if not in_fbe_mode and status == "Target Locked" and not has_path and use_goal:
                goal_3d, gdepth = get_3d_point(uv[0], uv[1], snap_d, curr_state, sim, camera_snapshot=snap_c)
                if goal_3d is None and snap_d is not None:
                    goal_3d, gdepth = get_3d_point(uv[0], uv[1], depth, curr_state, sim, camera_snapshot=None)
                if goal_3d is not None and gdepth is not None:
                    if is_goal_in_blacklist(goal_3d, goal_blacklist):
                        path_points = []
                        current_goal_3d = None
                    else:
                        replan = current_goal_3d is None or np.linalg.norm(goal_3d - current_goal_3d) > REPLAN_DISTANCE_THRESHOLD
                        if replan:
                            path = habitat_sim.ShortestPath()
                            path.requested_start = sim.pathfinder.snap_point(curr_pos)
                            path.requested_end = goal_3d
                            if sim.pathfinder.find_path(path) and len(path.points) > 0:
                                current_goal_3d = goal_3d
                                goal_set_at = curr_pos.copy()
                                path_points = list(path.points)
                                goal_3d_recorded_at = time.time()
                                goal_3d_recorded_dist = np.linalg.norm(curr_pos - goal_3d)
                                print(f"✅ 规划路径 {path.geodesic_distance:.2f}m (Target Locked)")
            elif status in ("Inferred", "Searching"):
                if not in_fbe_mode:
                    path_points = []
                    current_goal_3d = None
                if status == "Searching":
                    smoothed_offset = None
                    turn_commit_remaining = 0
                    searching_frames += 1
                else:
                    searching_frames = 0

            # v2: Searching 持续过久触发 FBE
            if status == "Searching" and not in_fbe_mode and searching_frames >= FBE_SEARCHING_FRAMES:
                explore_waypoint = get_explore_waypoint(sim, curr_pos)
                if explore_waypoint is not None:
                    print("🔍 FBE: Searching 过久，前往探索点")
                    path = habitat_sim.ShortestPath()
                    path.requested_start = sim.pathfinder.snap_point(curr_pos)
                    path.requested_end = explore_waypoint
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        path_points = list(path.points)
                        in_fbe_mode = True
                        searching_frames = 0
            
            # 位置卡住检测：即使 Inferred 状态，位置长时间不变也触发 FBE
            if not in_fbe_mode:
                if step_count % stuck_position_check_interval == 0:
                    if stuck_position_last_pos is not None:
                        dist_moved = np.linalg.norm(curr_pos - stuck_position_last_pos)
                        if dist_moved < stuck_position_threshold:
                            stuck_position_frames += stuck_position_check_interval
                        else:
                            stuck_position_frames = 0
                    stuck_position_last_pos = curr_pos.copy()
                
                # 触发 FBE
                if stuck_position_frames >= STUCK_FBE_TRIGGER_FRAMES:
                    explore_waypoint = get_explore_waypoint(sim, curr_pos)
                    if explore_waypoint is not None:
                        print(f"🔍 FBE: 位置卡住 {stuck_position_frames} 帧，前往探索点")
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(curr_pos)
                        path.requested_end = explore_waypoint
                        if sim.pathfinder.find_path(path) and len(path.points) > 1:
                            path_points = list(path.points)
                            in_fbe_mode = True
                            stuck_position_frames = 0
                            searching_frames = 0

            # 执行控制：目标锁定且已有 A* 规划则直接沿路径向目标移动，其他逻辑不变
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
                    step = curr_pos + (move_vec / d) * MOVE_STEP_SIZE
                    s = agent.get_state()
                    s.position = step
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

            # 可视化
            status_display = f"{status}"
            if len(target_list) > 1:
                status_display += f" [{current_target_idx+1}/{len(target_list)}:{target_list[current_target_idx]}]"
            if in_fbe_mode:
                status_display += " [FBE]"

            # 使用场景俯视图可视化（替代原有二维坐标地图）
            current_instruction_display = target_list[current_target_idx] if current_target_idx < len(target_list) else INSTRUCTION
            current_yaw = get_agent_forward_yaw(curr_state)
            fbe_pt = explore_waypoint if in_fbe_mode else None
            topdown_viz.update_matplotlib(
                trajectory=trajectory,
                path_points=path_points,
                current_pos=curr_pos,
                goal_pos=current_goal_3d,
                start_pos=start_pos,
                current_yaw=current_yaw,
                title=status_display,
                instruction=current_instruction_display,
                status=status,
                fbe_point=fbe_pt,
                gt_pos=GT_TARGET_POSITION  # GT 目标位置（可选）
            )

            viz = rgb.copy()
            if uv:
                cv2.circle(viz, uv, 12, (0, 255, 0), 2)
            cv2.putText(viz, status_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Robot Eye", viz)
            if cv2.waitKey(20) == ord('q'):
                break
            step_count += 1
            time.sleep(0.02)
    finally:
        sim.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
