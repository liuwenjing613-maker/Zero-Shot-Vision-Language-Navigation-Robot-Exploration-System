# ==============================================================================
# 文件名: local_robot_simple.py
# 运行位置: 本地 Ubuntu 机器 (已安装 habitat-sim)
# 功能: VLN 导航 - 状态机 + 双模态融合 (按诊断文档重构)
# ==============================================================================

import matplotlib
matplotlib.use('TkAgg')

import habitat_sim
import numpy as np
import cv2
import requests
import os
import magnum as mn
import matplotlib.pyplot as plt
import time
import threading
import math

# ============== 1. 配置 ==============
CLOUD_URL = "http://127.0.0.1:5000/plan"
IMG_WIDTH, IMG_HEIGHT, HFOV = 640, 480, 110

SCENES_BASE = "/home/abc/ZeroShot_VLN/assets/scenes"
SCENE_LIST = [
    "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "MatterPort3D/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    "MatterPort3D/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
]
SCENE_INDEX = 2

INSTRUCTION = "plant"

# 核心参数
CLOUD_SEND_INTERVAL = 1.2
MOVE_STEP_SIZE = 0.06       # 减小步长，避免冲太快撞墙
REPLAN_DISTANCE_THRESHOLD = 1.5
STUCK_DEPTH_THRESHOLD = 0.6 # 提高阈值，更早检测到前方障碍
STUCK_FRAMES = 4            # 更快触发撞墙处理
DEPTH_MIN, DEPTH_MAX = 0.3, 5.0
TURN_INTERVAL = 12          # 每 N 帧才转一次，降低转速（约 4 次/秒）
INFER_SMOOTH_ALPHA = 0.25   # 推理偏移平滑系数，越小越平滑
INFER_TURN_COMMIT_FRAMES = 15  # 确定转向后至少保持 N 帧，防止左右摇摆
INFER_DEAD_ZONE = 100       # 偏移在此范围内视为对准，前进不转向
# 策略1: 双模态成功判定 - VLM PIXEL + 深度<此值即成功，废弃 Verify
SUCCESS_DEPTH_THRESHOLD = 0.6
# 策略3: 消除 sleep，每帧小步控制
TURN_ANGLE_THRESHOLD = 0.15  # 弧度，约 8.6°，小于此值才前进

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

shared_state = SharedState()

# ============== 3. Habitat 配置 ==============
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

# ============== 4. 工具函数 ==============
def get_depth_at_uv(u, v, depth_img):
    """取 (u,v) 处 5x5 邻域有效深度中值"""
    u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
    v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
    patch = depth_img[max(0, v_idx-2):v_idx+3, max(0, u_idx-2):u_idx+3]
    valid = patch[(patch > 0.1) & (patch < 10.0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))

def get_agent_forward_yaw(agent_state):
    """从 agent 旋转提取 XZ 平面朝向角 (弧度)，Habitat 前向为 -Z"""
    cam = agent_state.sensor_states["color_sensor"]
    q = cam.rotation
    mn_q = mn.Quaternion(mn.Vector3(q.imag), q.real)
    mat = mn_q.to_matrix()
    col2 = mat[2]  # Magnum 列优先，第三列
    fwd_x = -col2[0]
    fwd_z = -col2[2]
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

# ============== 5. 云端 worker ==============
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
        if img_to_send is not None:
            last_send = time.time()
            _, buf = cv2.imencode('.jpg', img_to_send)
            try:
                r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                  data={'instruction': INSTRUCTION}, timeout=8)
                j = r.json()
                with shared_state.lock:
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

# ============== 6. 主函数 ==============
def main():
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
    print(f"🎯 目标: {INSTRUCTION}")

    sim = habitat_sim.Simulator(make_cfg(scene_path))
    nav = habitat_sim.NavMeshSettings()
    nav.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, nav)
    agent = sim.initialize_agent(0)
    start_pos = sim.pathfinder.get_random_navigable_point()
    s = agent.get_state()
    s.position = start_pos
    agent.set_state(s)
    print(f"📍 起点: {start_pos}")

    threading.Thread(target=cloud_worker, daemon=True).start()
    print("✅ 云端 worker 已启动")

    cv2.namedWindow("Robot Eye", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Eye", 640, 480)
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.06)
    fig.canvas.draw()
    plt.show(block=False)

    path_points = []
    current_goal_3d = None
    goal_set_at = None
    stuck_count = 0
    step_count = 0
    smoothed_offset = None      # 推理偏移的指数滑动平均
    turn_commit_remaining = 0   # 转向承诺剩余帧数
    last_turn_direction = None  # "left" or "right"
    trajectory = [[start_pos[0], start_pos[2]]]

    print("🚀 开始导航 (按 q 退出)...")
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

            trajectory.append([curr_pos[0], curr_pos[2]])

            # 撞墙检测（策略3: 消除 sleep，每帧只转一次）
            center = depth[IMG_HEIGHT//2-30:IMG_HEIGHT//2+30, IMG_WIDTH//2-30:IMG_WIDTH//2+30]
            valid = center[(center > 0.1) & (center < 5.0)]
            front_d = float(np.median(valid)) if len(valid) > 0 else 2.0
            if front_d < STUCK_DEPTH_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0
            if stuck_count >= STUCK_FRAMES:
                print("🛑 撞墙，转向")
                path_points = []
                current_goal_3d = None
                stuck_count = 0
                with shared_state.lock:
                    shared_state.latest_goal_uv = None
                    shared_state.latest_goal_depth = None
                    shared_state.latest_goal_camera_snapshot = None
                for _ in range(4):
                    agent.act("turn_right")
                    time.sleep(0.08)  # 撞墙后转向稍慢，避免转晕

            has_path = path_points and len(path_points) > 1
            use_goal = uv and snap_d is not None and snap_c is not None

            # 策略1: 双模态成功判定（废弃 Verify）
            # 仅当 Target Locked 且 snap_d 中 (u,v) 处深度 < 0.8m 时成功（用锁定时的深度）
            if status == "Target Locked" and uv is not None and snap_d is not None:
                d_at_target = get_depth_at_uv(uv[0], uv[1], snap_d)
                if d_at_target is not None and d_at_target < SUCCESS_DEPTH_THRESHOLD:
                    print("\n" + "★" * 20 + "\n🎯 双模态判定成功: VLM 锁定 + 深度 {:.2f}m < {:.2f}m\n".format(
                        d_at_target, SUCCESS_DEPTH_THRESHOLD) + "★" * 20)
                    plt.savefig("vln_success.png")
                    time.sleep(2)
                    break

            # 策略2: 分离 Locked 与 Inferred 控制逻辑
            # 状态 A: Target Locked → A* 规划 + 沿路径走
            # 状态 B: Inferred / Searching → 绝不调用 A*，仅转向
            if status == "Target Locked" and not has_path and use_goal:
                goal_3d, gdepth = get_3d_point(uv[0], uv[1], snap_d, curr_state, sim, camera_snapshot=snap_c)
                if goal_3d is not None and gdepth is not None:
                    replan = current_goal_3d is None or np.linalg.norm(goal_3d - current_goal_3d) > REPLAN_DISTANCE_THRESHOLD
                    if replan:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(curr_pos)
                        path.requested_end = goal_3d
                        if sim.pathfinder.find_path(path) and len(path.points) > 0:
                            current_goal_3d = goal_3d
                            goal_set_at = curr_pos.copy()
                            path_points = list(path.points)
                            print(f"✅ 规划路径 {path.geodesic_distance:.2f}m (Target Locked)")
            elif status in ("Inferred", "Searching"):
                # 状态 B: 不规划，清空路径
                path_points = []
                current_goal_3d = None
                if status == "Searching":
                    smoothed_offset = None
                    turn_commit_remaining = 0

            # 执行控制：有路径则沿路径移动，无路径则转向（Inferred 时防摇摆）
            # 前进前检查前方深度，过近则转向不前进
            front_clear = front_d >= 0.85
            if path_points and front_clear:
                # 有路径且前方畅通：沿路径平移
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
            elif path_points and not front_clear:
                # 有路径但前方有障碍：先转向
                if step_count % TURN_INTERVAL == 0:
                    agent.act("turn_right")
            else:
                # 无路径：Inferred 时平滑+承诺转向防摇摆，Searching 时单向转
                if status == "Inferred" and turn_commit_remaining > 0:
                    turn_commit_remaining -= 1  # 每帧递减，约 0.6s 后重评估
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
                        if last_turn_direction == "right":
                            agent.act("turn_right")
                        else:
                            agent.act("turn_left")
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

            # 可视化：动态范围包含轨迹、路径、目标
            ax.clear()
            all_x = [curr_pos[0]]
            all_z = [curr_pos[2]]
            if len(trajectory) > 1:
                t = np.array(trajectory)
                all_x.extend(t[:, 0])
                all_z.extend(t[:, 1])
            if path_points:
                p = np.array(path_points)
                all_x.extend(p[:, 0])
                all_z.extend(p[:, 2])
            if current_goal_3d is not None:
                all_x.append(current_goal_3d[0])
                all_z.append(current_goal_3d[2])
            margin = 2.0
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            z_min, z_max = min(all_z) - margin, max(all_z) + margin
            if x_max - x_min < 1.0:
                x_min, x_max = x_min - 5, x_max + 5
            if z_max - z_min < 1.0:
                z_min, z_max = z_min - 5, z_max + 5
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)
            ax.set_aspect('equal')
            if len(trajectory) > 1:
                t = np.array(trajectory)
                ax.plot(t[:, 0], t[:, 1], 'r-', lw=1.5, alpha=0.7, label='Trajectory')
            if path_points:
                p = np.array(path_points)
                ax.plot(p[:, 0], p[:, 2], 'b-', lw=2, label='Plan')
            ax.plot(curr_pos[0], curr_pos[2], 'go', markersize=10, label='Current')
            if current_goal_3d is not None:
                ax.plot(current_goal_3d[0], current_goal_3d[2], 'rx', markersize=12, label='Goal')
            ax.set_title(f"Status: {status}", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.canvas.draw()
            fig.canvas.flush_events()

            viz = rgb.copy()
            if uv:
                cv2.circle(viz, uv, 12, (0, 255, 0), 2)
            cv2.putText(viz, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
