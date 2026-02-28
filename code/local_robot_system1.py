'''
# ==============================================================================
# 文件名: local_robot_system1.py
# 运行位置: 本地 Ubuntu 机器 (已安装 habitat-sim)
# 功能: 仿真环境、异步发送请求、坐标转换、A*导航 + 实时可视化窗口
# ==============================================================================

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
import cv2
import requests
import os
from PIL import Image
import magnum as mn
import matplotlib.pyplot as plt
import time
import threading  # 用于异步 System 2
import random

# ============== 1. 配置 ==============
CLOUD_URL = "http://127.0.0.1:5000/plan"  # 修改为你云端实际 IP:端口
SCENE_PATH = "/home/abc/ZeroShot_VLN/assets/scenes/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb"  # 修改为你的 .glb 路径
INSTRUCTION = "chair"

IMG_WIDTH = 640
IMG_HEIGHT = 480
HFOV = 110

UPDATE_INTERVAL = 2.0  # 每 2s 更新一次目标（向云端发送请求）

# 全局变量（共享 System 2 和 System 1）
latest_target_pixel = None
latest_target_3d = None
latest_status = "探索中"

# ============== 2. Habitat 环境初始化 ==============
def make_simple_cfg(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    rgb_sensor_spec.position = [0.0, 0.5, 0.0]
    rgb_sensor_spec.hfov = HFOV

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    depth_sensor_spec.position = [0.0, 0.5, 0.0]
    depth_sensor_spec.hfov = HFOV

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# ============== 3. 2D 像素 → 3D 世界坐标 ==============
def get_3d_point(u, v, depth_img, agent_state, sim):
    z_depth = depth_img[v, u]
    if z_depth <= 0 or z_depth > 10.0 or np.isnan(z_depth):
        print(f"⚠️ 无效深度: {z_depth} at pixel ({u},{v})")
        return None

    f_x = (IMG_WIDTH / 2.0) / np.tan(np.deg2rad(HFOV) / 2.0)
    f_y = (IMG_HEIGHT / 2.0) / np.tan(np.deg2rad(HFOV) / 2.0)
    c_x = IMG_WIDTH / 2.0
    c_y = IMG_HEIGHT / 2.0

    x_c = (u - c_x) * z_depth / f_x
    y_c = -(v - c_y) * z_depth / f_y
    z_c = -z_depth

    sensor_pose = agent_state.sensor_states["color_sensor"]
    pos = mn.Vector3(sensor_pose.position)
    q_raw = sensor_pose.rotation

    w = q_raw.real
    xyz = q_raw.imag
    mn_q = mn.Quaternion(xyz, w)

    mat = mn.Matrix4.from_(mn_q.to_matrix(), pos)
    
    local_point_v3 = mn.Vector3(x_c, y_c, z_c)
    world_point = mat.transform_point(local_point_v3)
    raw_3d = np.array([world_point.x, world_point.y, world_point.z], dtype=np.float32)

    print(f"原始 3D 点: {raw_3d}")

    if sim.pathfinder.is_loaded:
        snapped = sim.pathfinder.snap_point(raw_3d)
        if np.isnan(snapped).any():
            print("snap_point 失败，返回原始")
            return raw_3d
        print(f"snap_point 成功: {snapped}")
        return snapped
    
    return raw_3d

# ============== 4. System 2 线程：异步更新目标 ==============
def system2_thread(sim, agent):
    global latest_target_pixel, latest_target_3d, latest_status
    
    while True:
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        depth = observations["depth_sensor"]
        
        rgb_img = rgb[:, :, :3][..., ::-1]
        temp_img_path = "temp_local_view.jpg"
        cv2.imwrite(temp_img_path, rgb_img)

        print(f"📡 [System 2] 呼叫云端... 指令: {INSTRUCTION}")
        try:
            with open(temp_img_path, 'rb') as f:
                response = requests.post(
                    CLOUD_URL, 
                    files={'image': f},
                    data={'instruction': INSTRUCTION}
                )
            res_json = response.json()
        except Exception as e:
            print(f"❌ [System 2] 连接云端失败: {e}")
            time.sleep(UPDATE_INTERVAL)
            continue

        if res_json['status'] == 'success':
            u, v = res_json['u'], res_json['v']
            print(f"✅ [System 2] 云端更新目标! 像素: ({u}, {v})")
            latest_target_pixel = (u, v)
            agent_state = agent.get_state()
            new_3d = get_3d_point(u, v, depth, agent_state, sim)
            if new_3d is not None:
                latest_target_3d = new_3d
                latest_status = "目标更新"
            else:
                latest_status = "3D 计算失败"
        else:
            latest_status = "未发现目标"
            print(f"🤔 [System 2] 当前未发现目标")

        time.sleep(UPDATE_INTERVAL)  # 每 2s 更新一次

# ============== 5. 主函数 ==============
def main():
    if not os.path.exists(SCENE_PATH):
        print(f"❌ 错误：场景文件不存在 {SCENE_PATH}")
        return

    cfg = make_simple_cfg(SCENE_PATH)
    sim = habitat_sim.Simulator(cfg)

    print("🛠️ [本地] 正在动态计算导航网格 (NavMesh)...")
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    if not sim.pathfinder.is_loaded:
        print("❌ NavMesh 加载失败！")
        return
    print("✅ NavMesh 动态生成成功！")

    agent = sim.initialize_agent(0)
    random_point = sim.pathfinder.get_random_navigable_point()
    agent_state = agent.get_state()
    agent_state.position = random_point
    agent.set_state(agent_state)
    print(f"📍 机器人初始化位置: {random_point}")

    # =========================================================
    # 实时可视化初始化
    # =========================================================
    cv2.namedWindow("实时视角 (RGB + 目标)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("实时视角 (RGB + 目标)", 800, 600)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("实时导航鸟瞰 (XZ 平面)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)
    line_path, = ax.plot([], [], 'b-', linewidth=2, label='规划路径')
    line_history, = ax.plot([], [], 'r--', linewidth=1, label='历史轨迹')
    point_curr, = ax.plot([], [], 'go', markersize=10, label='当前位置')
    point_goal, = ax.plot([], [], 'ro', markersize=10, label='目标')
    text_status = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    ax.legend()
    plt.show(block=False)
    
    history_points = [np.array(random_point)]

    # =========================================================
    # 启动 System 2 异步线程
    # =========================================================
    threading.Thread(target=system2_thread, args=(sim, agent), daemon=True).start()
    print("✅ System 2 线程启动：每 2s 更新目标")

    # =========================================================
    # System 1 主循环：实时导航
    # =========================================================
    print("\n--- 开始实时导航 (按 q 退出) ---")
    while True:
        # 获取当前观测
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        depth = observations["depth_sensor"]
        
        rgb_img = rgb[:, :, :3][..., ::-1]
        rgb_img_with_goal = rgb_img.copy()

        # 如果有最新目标，画绿圈
        if latest_target_pixel:
            cv2.circle(rgb_img_with_goal, latest_target_pixel, 15, (0, 255, 0), -1)
            cv2.putText(rgb_img_with_goal, "Goal Pixel", (latest_target_pixel[0]+20, latest_target_pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示实时视角
        cv2.imshow("实时视角 (RGB + 目标)", rgb_img_with_goal)
        key = cv2.waitKey(100)  # 每 0.1s 刷新
        if key == ord('q'):
            print("用户按 q 退出")
            break

        # 如果有最新 3D 目标，规划路径
        if latest_target_3d is not None:
            agent_state = agent.get_state()
            path = habitat_sim.ShortestPath()
            start_point = sim.pathfinder.snap_point(agent_state.position)
            path.requested_start = start_point
            path.requested_end = latest_target_3d

            try:
                start_island = sim.pathfinder.get_island(start_point)
                end_island = sim.pathfinder.get_island(latest_target_3d)
                if start_island != end_island:
                    print(f"⚠️ 岛屿不连通！尝试吸附")
                    latest_target_3d = sim.pathfinder.snap_point(latest_target_3d, island_index=start_island)
                    path.requested_end = latest_target_3d
            except Exception as e:
                print(f"⚠️ Island 检查失败: {e}")

            found_path = sim.pathfinder.find_path(path)
            
            if found_path and len(path.points) > 0:
                print(f"✅ 新路径规划成功! 长度: {path.geodesic_distance:.2f} 米")
                
                # 更新鸟瞰图
                points = np.array(path.points)
                line_path.set_data(points[:,0], points[:,2])
                point_goal.set_data([latest_target_3d[0]], [latest_target_3d[2]])
                fig.canvas.draw()
                fig.canvas.flush_events()

                # 执行下一步移动（小步前进）
                next_waypoint = path.points[1]  # 下一个点
                agent_state.position = next_waypoint
                agent.set_state(agent_state)

                history_points.append(next_waypoint)
                history_np = np.array(history_points)
                line_history.set_data(history_np[:,0], history_np[:,2])

                point_curr.set_data([next_waypoint[0]], [next_waypoint[2]])

                remaining_dist = path.geodesic_distance  # 剩余路径长度

                text_status.set_text(
                    f"状态: {latest_status}\n"
                    f"位置: {next_waypoint}\n"
                    f"剩余路径: {remaining_dist:.2f}m"
                )
                fig.canvas.draw()
                fig.canvas.flush_events()

            else:
                print("⚠️ 路径规划失败，使用默认动作")
                agent.act("move_forward")  # 默认前进

        else:
            print("⚠️ 无目标，使用随机探索动作")
            action = random.choice(["turn_left", "turn_right", "move_forward"])
            agent.act(action)

        # 小延迟，控制 FPS
        time.sleep(0.05)  # ~20 FPS

    # 清理
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
    '''
# ==============================================================================
# 文件名: local_robot_system1.py
# 运行位置: 本地 Ubuntu 机器
# 功能: Habitat仿真 + A*导航 + 异步云端推理 + 实时可视化 (修复GUI冲突与线程安全)
# ==============================================================================

import matplotlib
matplotlib.use('TkAgg') # 解决 GUI 冲突的关键

import habitat_sim
import habitat_sim.agent
import numpy as np
import cv2
import requests
import os
import magnum as mn
import matplotlib.pyplot as plt
import time
import threading
import random

# ============== 1. 配置 ==============
CLOUD_URL = "http://127.0.0.1:5000/plan"  # 改为云端实际 IP:端口
IMG_WIDTH, IMG_HEIGHT, HFOV = 640, 480, 110

# ---------- 场景：多场景可选，更复杂环境 ----------
# 场景根目录（与 ZeroShot_VLN 或 InternVLA/scenes 对齐）
SCENES_BASE = "/home/abc/ZeroShot_VLN/assets/scenes"
# 可选场景列表：前 3 个为 Replica 风格（室内/城堡），其余为 Matterport3D 多户型
SCENE_LIST = [
    "van-gogh-room.glb",           # Replica 室内
    "skokloster-castle.glb",       # Replica 城堡，结构复杂
    "apartment_1.glb",             # Replica 公寓
    "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "mp3d/8194nk5LbLH/8194nk5LbLH.glb",
    "mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb",
    "MatterPort3D/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb",
    "MatterPort3D/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb",
    "MatterPort3D/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
    "MatterPort3D/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    "MatterPort3D/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
    "MatterPort3D/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb",
]
# 场景选择: "random" 每次随机一个 | "first" 用第一个 | 整数 用 SCENE_LIST[i]
SCENE_CHOICE = 3

# ---------- 指令：可固定或随机，增加任务难度 ----------
INSTRUCTION = "plant"  # SCENE_CHOICE 非 "random" 时使用
INSTRUCTION_LIST = ["chair", "table", "couch", "plant", "lamp", "bed", "potted plant", "sofa"]
# 指令选择: "random" 每次随机一个 | "fixed" 使用上面 INSTRUCTION
INSTRUCTION_CHOICE = "chair"

# 初始位置模式: "random" 随机可导航点 | "fixed" 使用下方固定坐标
INIT_POSITION_MODE = "random"  # 改为 "fixed" 则使用 FIXED_START_POSITION
FIXED_START_POSITION = [0.0, 0.0, 0.0]  # [x, y, z]，固定模式时使用，会吸附到最近可导航点

# （成功判定已去掉，先测试导航功能）

# 深度与目标：先测前方深度，目标距离 = 测量深度 - 缓冲，避免撞墙
DEPTH_BUFFER = 0.4       # 米，目标放在 (测量深度 - 缓冲) 处
DEPTH_MIN_DIST = 0.4     # 目标至少离相机这么远
DEPTH_MAX_DIST = 4.5     # 目标最远这么远（再远多半是墙）
# 每步移动距离，保守策略：稍慢更稳
MOVE_STEP_SIZE = 0.15    # 米
# 向云端发送图像的间隔（秒），保守策略：较慢更新保证稳定
CLOUD_SEND_INTERVAL = 3.5
# 只有新目标与当前目标距离超过此值才重规划（米），避免来回抖
REPLAN_DISTANCE_THRESHOLD = 1.2
# 撞墙检测：前方深度小于此值且连续多帧则判定撞墙，转向并短暂忽略推理目标
STUCK_DEPTH_THRESHOLD = 0.5
STUCK_FRAMES = 5
STUCK_IGNORE_INFERRED_SEC = 5.0  # 撞墙后此秒数内忽略推理目标
# 已探索区域：格子大小（米），用于轨迹记录与 frontier 探索
VISITED_CELL_SIZE = 0.6
# 无目标时 frontier 探索：优先朝未探索空地前进，而非乱转
EXPLORE_FRONTIER_ENABLED = True
EXPLORE_FRONTIER_INTERVAL = 8   # 每 N 步尝试一次 frontier 规划
STUCK_EXPLORE_COOLDOWN_SEC = 3.0  # 撞墙后此秒数内只转向，不尝试 frontier（避免撞墙-后退循环）
# 推理目标：仅当探索超过此秒数后才使用（保守：优先靠 Target Locked）
INFER_USE_AFTER_SEARCH_SEC = 12.0
# 推理目标若指向过近障碍（深度<此值）则拒绝
INFER_MIN_DEPTH = 0.45

# ---------- 初始 360° 扫描：转一周评估各方向可能性，选最佳再导航 ----------
SCAN_ENABLED = True
SCAN_VIEWS = 4              # 4 个视角覆盖 360°（每 90° 一个）
SCAN_TURNS_PER_90 = 3       # 每 90° 需转几次（Habitat 默认约 30° 每次）
SCAN_TURN_DELAY = 0.25      # 每次转向间隔（秒），避免转太快

# 成功判定：到达后必须通过视觉验证才算真正成功
SUCCESS_DISTANCE = 0.6           # 到目标点距离小于此才进入「到达验证」（稍放宽便于触发）
MIN_OBJECT_DEPTH = 0.5           # 目标定位深度需在此范围内（否则为墙）
MAX_OBJECT_DEPTH = 4.0
MIN_PATH_LENGTH_FOR_SUCCESS = 0.2   # 规划路径至少 0.2m 才允许判成功（放宽）
MIN_TRAVEL_FOR_SUCCESS = 0.2       # 从设目标起至少走了 0.2m 才允许判成功（放宽）

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_rgb = None
        self.latest_goal_uv = None
        self.latest_status = "Searching..."
        self.new_image_ready = False
        self.current_depth = None
        self.current_agent_state = None
        # 与 latest_goal_uv 同帧的深度和相机位姿，保证像素→3D 用同一帧
        self.latest_goal_depth = None
        self.latest_goal_camera_snapshot = None  # (position_xyz, rotation_real, rotation_imag)
        # 撞墙后在此时间前忽略「推理目标」，只接受真实看到的目标
        self.ignore_inferred_until = 0.0
        # 撞墙后在此时间前只转向，不尝试 frontier（避免撞墙-后退循环）
        self.stuck_explore_cooldown_until = 0.0

shared_state = SharedState()

def get_frontier_goal(visited_cells, curr_pos, sim, cell_size=VISITED_CELL_SIZE):
    """
    获取最近的未探索 frontier 点（已探索格子的邻接未访问格子中心）。
    返回 (goal_3d, path) 或 (None, None)。
    """
    frontiers = set()
    for (gx, gz) in visited_cells:
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (gx + dx, gz + dz)
            if n not in visited_cells:
                frontiers.add(n)
    if not frontiers:
        return None, None
    floor_y = float(curr_pos[1])
    best_goal, best_path, best_dist = None, None, 0.0
    for (gx, gz) in frontiers:
        wx = (gx + 0.5) * cell_size
        wz = (gz + 0.5) * cell_size
        goal_3d = np.array([wx, floor_y, wz], dtype=np.float32)
        if sim.pathfinder.is_loaded:
            goal_3d = sim.pathfinder.snap_point(goal_3d)
            if np.isnan(goal_3d).any():
                continue
        path = habitat_sim.ShortestPath()
        path.requested_start = sim.pathfinder.snap_point(curr_pos)
        path.requested_end = goal_3d
        if sim.pathfinder.find_path(path) and path.geodesic_distance > 0.15:
            d = path.geodesic_distance
            if d > best_dist:  # 选最远的 frontier，探索更充分
                best_dist, best_goal, best_path = d, goal_3d, path
    if best_goal is not None:
        return best_goal, best_path
    return None, None

def make_simple_cfg(scene_path):
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

def get_3d_point(u, v, depth_img, agent_state, sim, camera_snapshot=None):
    """
    先测量 (u,v) 方向上的前方深度，目标距离 = 测量深度 - 缓冲，再投影到地面。
    返回 (point_3d, depth_used) 或 (None, None)。
    """
    try:
        u_idx = int(np.clip(u, 0, IMG_WIDTH - 1))
        v_idx = int(np.clip(v, 0, IMG_HEIGHT - 1))
        # 测量该像素邻域内的有效深度（前方障碍距离）
        measured_depth = None
        for half in [0, 2, 4]:
            y_start = max(0, v_idx - half)
            y_end = min(IMG_HEIGHT, v_idx + half + 1)
            x_start = max(0, u_idx - half)
            x_end = min(IMG_WIDTH, u_idx + half + 1)
            depth_patch = depth_img[y_start:y_end, x_start:x_end]
            valid = depth_patch[(depth_patch > 0.1) & (depth_patch < DEPTH_MAX_DIST)]
            if len(valid) == 0:
                continue
            measured_depth = float(np.median(valid))
            if measured_depth <= 0.1:
                continue
            break
        if measured_depth is None:
            return None, None
        z_depth = measured_depth
    except Exception as e:
        print(f"⚠️ 深度提取失败: {e}")
        return None, None

    f_x = f_y = (IMG_WIDTH / 2.0) / np.tan(np.deg2rad(HFOV) / 2.0)
    c_x, c_y = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0
    x_c = (u_idx - c_x) * z_depth / f_x
    y_c = -(v_idx - c_y) * z_depth / f_y
    z_c = -z_depth

    if camera_snapshot is not None:
        pos_xyz, rot_real, rot_imag = camera_snapshot
        pos = mn.Vector3(pos_xyz)
        mn_q = mn.Quaternion(mn.Vector3(rot_imag), rot_real)
    else:
        sensor_pose = agent_state.sensor_states["color_sensor"]
        pos = mn.Vector3(sensor_pose.position)
        q_raw = sensor_pose.rotation
        mn_q = mn.Quaternion(mn.Vector3(q_raw.imag), q_raw.real)
    mat = mn.Matrix4.from_(mn_q.to_matrix(), pos)
    world_point = mat.transform_point(mn.Vector3(x_c, y_c, z_c))
    P = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
    point_at_depth = np.array([world_point.x, world_point.y, world_point.z], dtype=np.float32)
    ray_dir = point_at_depth - P
    ray_norm = np.linalg.norm(ray_dir)
    if ray_norm < 1e-6:
        return None, None
    ray_dir /= ray_norm

    # 目标距离 = 测量深度 - 缓冲（避免贴墙），再限制在合理范围
    effective_dist = z_depth - DEPTH_BUFFER
    effective_dist = np.clip(effective_dist, DEPTH_MIN_DIST, DEPTH_MAX_DIST)

    if sim.pathfinder.is_loaded:
        floor_snapped = sim.pathfinder.snap_point(P)
        floor_y = float(floor_snapped[1])
    else:
        floor_y = float(P[1]) - 0.5

    # 沿射线放目标在 effective_dist 处，再压到地面
    raw_3d = P + effective_dist * ray_dir
    raw_3d[1] = floor_y

    if sim.pathfinder.is_loaded:
        snapped_point = sim.pathfinder.snap_point(raw_3d)
        if np.isnan(snapped_point).any():
            return raw_3d, z_depth
        offset = np.linalg.norm(snapped_point - raw_3d)
        if offset > 0.1:
            print(f"📍 坐标修正：吸附到地面，偏移 {offset:.2f}m")
        return snapped_point, z_depth
    return raw_3d, z_depth
def system2_worker():
    last_send_time = 0.0
    while True:
        img_to_send = None
        depth_snapshot = None
        camera_snapshot = None
        now = time.time()
        # 控制发送频率，避免目标坐标频繁变化导致来回走
        if now - last_send_time < CLOUD_SEND_INTERVAL:
            time.sleep(0.2)
            continue
        with shared_state.lock:
            if shared_state.new_image_ready and shared_state.current_rgb is not None:
                img_to_send = shared_state.current_rgb.copy()
                if shared_state.current_depth is not None:
                    depth_snapshot = shared_state.current_depth.copy()
                if shared_state.current_agent_state is not None:
                    cam = shared_state.current_agent_state.sensor_states["color_sensor"]
                    camera_snapshot = (
                        np.array(cam.position, dtype=np.float32),
                        float(cam.rotation.real),
                        np.array(cam.rotation.imag, dtype=np.float32),
                    )
                shared_state.new_image_ready = False

        if img_to_send is not None:
            last_send_time = time.time()
            _, buf = cv2.imencode('.jpg', img_to_send)
            try:
                response = requests.post(
                    CLOUD_URL,
                    files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                    data={'instruction': INSTRUCTION},
                    timeout=5,
                )
                res_json = response.json()
                with shared_state.lock:
                    if res_json['status'] == 'success':
                        shared_state.latest_goal_uv = (res_json['u'], res_json['v'])
                        shared_state.latest_goal_depth = depth_snapshot
                        shared_state.latest_goal_camera_snapshot = camera_snapshot
                        shared_state.latest_status = "Target Locked"
                    elif res_json.get('message') == 'Inferred' and 'u' in res_json and 'v' in res_json:
                        # 模型推理方向：仅在没有真实目标且未在撞墙冷却期内时使用
                        if shared_state.latest_status != "Target Locked" and time.time() >= shared_state.ignore_inferred_until:
                            shared_state.latest_goal_uv = (res_json['u'], res_json['v'])
                            shared_state.latest_goal_depth = depth_snapshot
                            shared_state.latest_goal_camera_snapshot = camera_snapshot
                            shared_state.latest_status = "Inferred Direction"
                    else:
                        shared_state.latest_status = "Searching..."
            except Exception:
                pass
        time.sleep(0.2)

def main():
    # 每次运行用当前时间重置随机种子
    seed = int(time.time() * 1000) % (2**32)
    np.random.seed(seed)
    random.seed(seed)

    # 选场景
    if SCENE_CHOICE == "random":
        idx = random.randint(0, len(SCENE_LIST) - 1)
    elif SCENE_CHOICE == "first":
        idx = 0
    else:
        idx = int(SCENE_CHOICE) % len(SCENE_LIST)
    scene_rel = SCENE_LIST[idx]
    scene_path = os.path.join(SCENES_BASE, scene_rel)
    if not os.path.exists(scene_path):
        for i in range(len(SCENE_LIST)):
            candidate = os.path.join(SCENES_BASE, SCENE_LIST[i])
            if os.path.exists(candidate):
                scene_path = candidate
                print(f"⚠️ 原选场景不存在，改用: {SCENE_LIST[i]}")
                break
        else:
            print(f"❌ 无可用场景，请检查 SCENES_BASE={SCENES_BASE} 下是否存在 .glb")
            return
    print(f"📍 场景: {scene_path}")

    # 选指令（随机时覆盖全局，供 system2_worker 使用）
    if INSTRUCTION_CHOICE == "random":
        global INSTRUCTION
        INSTRUCTION = random.choice(INSTRUCTION_LIST)
    print(f"🎯 指令: {INSTRUCTION}")

    sim = habitat_sim.Simulator(make_simple_cfg(scene_path))
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, nav_settings)

    # 用时间种子设置 pathfinder 的随机数，保证每次运行随机起点不同
    if hasattr(sim.pathfinder, 'seed'):
        sim.pathfinder.seed(seed)

    agent = sim.initialize_agent(0)
    agent_state = agent.get_state()
    if INIT_POSITION_MODE == "fixed":
        start_pos = sim.pathfinder.snap_point(np.array(FIXED_START_POSITION, dtype=np.float32))
        if np.isnan(start_pos).any():
            print("⚠️ 固定起点不可导航，改为随机起点")
            start_pos = sim.pathfinder.get_random_navigable_point()
        else:
            print(f"📍 固定起点: {start_pos}")
    else:
        start_pos = sim.pathfinder.get_random_navigable_point()
        print(f"📍 随机起点 (seed={seed}): {start_pos}")
    agent_state.position = start_pos
    agent.set_state(agent_state)

    # ---------- 初始 360° 扫描：转一周评估各方向，选最佳再开始导航 ----------
    if SCAN_ENABLED:
        def initial_scan_phase():
            nonlocal exploration_no_path_since
            views = []
            print("🔍 初始扫描：转 360° 评估各方向...")
            for i in range(SCAN_VIEWS):
                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][:, :, :3][..., ::-1]
                depth = obs["depth_sensor"].copy() if obs["depth_sensor"] is not None else None
                curr_state = agent.get_state()
                cam = curr_state.sensor_states["color_sensor"]
                camera_snapshot = (
                    np.array(cam.position, dtype=np.float32),
                    float(cam.rotation.real),
                    np.array(cam.rotation.imag, dtype=np.float32),
                )
                _, buf = cv2.imencode('.jpg', rgb)
                try:
                    r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                      data={'instruction': INSTRUCTION, 'scan': '1'}, timeout=10)
                    j = r.json()
                    u = j.get('u')
                    v = j.get('v')
                    conf = float(j.get('confidence', 0.0))
                    st = "Target Locked" if j.get('status') == 'success' else "Inferred Direction"
                    if u is not None and v is not None:
                        views.append((int(u), int(v), depth, camera_snapshot, st, conf))
                        print(f"  视角 {i+1}/4: ({u},{v}) 可能性={conf:.2f} {st}")
                except Exception as e:
                    print(f"  视角 {i+1}/4: 请求失败 {e}")
                if i < SCAN_VIEWS - 1:
                    for _ in range(SCAN_TURNS_PER_90):
                        agent.act("turn_right")
                        time.sleep(SCAN_TURN_DELAY)
            if views:
                best = max(views, key=lambda x: (1 if x[4] == "Target Locked" else 0, x[5]))
                u, v, depth, cam, st, _ = best
                with shared_state.lock:
                    shared_state.latest_goal_uv = (u, v)
                    shared_state.latest_goal_depth = depth
                    shared_state.latest_goal_camera_snapshot = cam
                    shared_state.latest_status = st
                print(f"✅ 扫描完成，选择方向 ({u},{v}) {st}")
                exploration_no_path_since = time.time() - 20  # 视为已探索足够久，直接使用扫描结果
            else:
                print("⚠️ 扫描无有效结果，将进入探索模式")
        initial_scan_phase()

    threading.Thread(target=system2_worker, daemon=True).start()

    # 初始化可视化
    cv2.namedWindow("Robot Eye", cv2.WINDOW_NORMAL)
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(start_pos[0]-8, start_pos[0]+8)
    ax.set_ylim(start_pos[2]-8, start_pos[2]+8)
    line_path, = ax.plot([], [], 'b-', linewidth=2)
    point_curr, = ax.plot([], [], 'go', markersize=8)
    point_goal, = ax.plot([], [], 'rx', markersize=10)
    
    # 【关键修复】初始绘制以缓存渲染器
    fig.canvas.draw()
    plt.show(block=False)
    background = fig.canvas.copy_from_bbox(ax.bbox)

    current_goal_3d = None
    current_goal_depth_used = None
    path_length_when_goal_set = 0.0
    goal_set_at_position = None
    goal_was_target_locked = False
    goal_is_exploration = False  # True=frontier 探索目标，到达后不验证直接清空
    path_points = []
    step_count = 0
    stuck_count = 0
    visited_cells = set()
    exploration_no_path_since = time.time()  # 无路径探索开始时间

# 【轨迹记录 1】：增加历史轨迹线
    line_path, = ax.plot([], [], 'b-', linewidth=2, label='A* Plan')
    line_traj, = ax.plot([], [], 'r-', linewidth=1, alpha=0.6, label='History') # 红色轨迹线
    point_curr, = ax.plot([], [], 'go', markersize=8)
    point_goal, = ax.plot([], [], 'rx', markersize=10)
    ax.legend() # 显示图例

    # 【数据存储】：
    actual_trajectory = [] # 用于存储历史点


    print("🚀 系统启动成功，开始导航...")
    
    try:
        while True:
            # 1. 获取当前传感器数据
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][:, :, :3][..., ::-1]
            depth = obs["depth_sensor"]

            curr_state = agent.get_state() # 捕获这一秒的位姿

            # 2. 将画面推给云端 (异步)
            with shared_state.lock:
                shared_state.current_rgb = rgb
                shared_state.new_image_ready = True 

                shared_state.current_depth = depth       # 【新增】
                shared_state.current_agent_state = curr_state # 【新增】

                uv_goal = shared_state.latest_goal_uv
                status = shared_state.latest_status
                snap_depth = shared_state.latest_goal_depth
                snap_camera = shared_state.latest_goal_camera_snapshot
                in_stuck_cooldown = time.time() < shared_state.stuck_explore_cooldown_until

            # 【轨迹记录 2】：记录当前位置与已探索格子
            curr_pos = agent.get_state().position
            actual_trajectory.append([curr_pos[0], curr_pos[2]])
            ix, iz = int(round(curr_pos[0] / VISITED_CELL_SIZE)), int(round(curr_pos[2] / VISITED_CELL_SIZE))
            visited_cells.add((ix, iz))

            # 2.5 撞墙检测：前方很近有障碍则判定撞墙，清目标、转向、冷却期内忽略推理
            depth_center = depth[IMG_HEIGHT//2 - 25:IMG_HEIGHT//2 + 25, IMG_WIDTH//2 - 25:IMG_WIDTH//2 + 25]
            valid_d = depth_center[(depth_center > 0.1) & (depth_center < 5.0)]
            front_depth = float(np.median(valid_d)) if len(valid_d) > 0 else 2.0
            if front_depth < STUCK_DEPTH_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0
            if stuck_count >= STUCK_FRAMES:
                print("🛑 检测到撞墙，转向并冷却探索")
                path_points = []
                current_goal_3d = None
                current_goal_depth_used = None
                stuck_count = 0
                for _ in range(4):
                    agent.act("turn_right")
                    time.sleep(0.08)
                with shared_state.lock:
                    shared_state.latest_goal_uv = None
                    shared_state.latest_goal_depth = None
                    shared_state.latest_goal_camera_snapshot = None
                    shared_state.latest_status = "Searching..."
                    shared_state.ignore_inferred_until = time.time() + STUCK_IGNORE_INFERRED_SEC
                    shared_state.stuck_explore_cooldown_until = time.time() + STUCK_EXPLORE_COOLDOWN_SEC

            # 3. 规划逻辑
            # 目标不在视野时：云端返回 NOT_VISIBLE，我们不清空上次目标，继续沿当前 path 走到「上次看到时的 3D 点」即可，无需一直看见目标。
            # 仅当「没有在执行路径」时才用云端新结果更新目标并重规划，避免边走边改导致来回走。
            has_active_path = path_points and len(path_points) > 1
            # 保守策略：仅当探索足够久后才使用推理目标，优先靠 Target Locked
            use_cloud_goal = uv_goal and snap_depth is not None and snap_camera is not None
            if use_cloud_goal and status == "Inferred Direction":
                if time.time() - exploration_no_path_since < INFER_USE_AFTER_SEARCH_SEC:
                    use_cloud_goal = False  # 探索时间不够，暂不用推理，纯转向找目标
            if not has_active_path and use_cloud_goal:
                agent_curr_state = agent.get_state()
                goal_3d, goal_depth_used = get_3d_point(
                    uv_goal[0], uv_goal[1], snap_depth, agent_curr_state, sim,
                    camera_snapshot=snap_camera,
                )
                if goal_3d is not None and goal_depth_used is not None:
                    # 推理目标：若指向过近障碍（墙）则跳过，不清空 shared_state
                    if status == "Inferred Direction" and goal_depth_used < INFER_MIN_DEPTH:
                        print("⚠️ 推理方向指向过近障碍，跳过（继续转向探索）")
                        goal_3d = None
                    if goal_3d is not None and (current_goal_3d is None or np.linalg.norm(goal_3d - current_goal_3d) > REPLAN_DISTANCE_THRESHOLD):
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sim.pathfinder.snap_point(agent_curr_state.position)
                        path.requested_end = goal_3d
                        if sim.pathfinder.find_path(path):
                            exploration_no_path_since = time.time()
                            current_goal_3d = goal_3d
                            current_goal_depth_used = goal_depth_used
                            path_length_when_goal_set = path.geodesic_distance
                            goal_set_at_position = np.array(agent_curr_state.position, dtype=np.float32)
                            goal_was_target_locked = (status == "Target Locked")
                            goal_is_exploration = False
                            path_points = list(path.points)
                            print(f"✅ 规划 A* 路径 (距离: {path.geodesic_distance:.2f}m)")
                        else:
                            print("⚠️ 路径不可达，继续转向探索")
                    

            # 3.5 成功判定：到达后视觉验证（frontier 探索目标到达即清空，不验证）
            if (current_goal_3d is not None and goal_set_at_position is not None):
                dist_to_goal = np.linalg.norm(curr_pos - current_goal_3d)
                if goal_is_exploration and dist_to_goal < SUCCESS_DISTANCE:
                    current_goal_3d = None
                    goal_set_at_position = None
                    path_points = []
                    goal_is_exploration = False
                elif current_goal_depth_used is not None:
                    travel_since = np.linalg.norm(curr_pos - goal_set_at_position)
                    depth_ok = MIN_OBJECT_DEPTH <= current_goal_depth_used <= MAX_OBJECT_DEPTH
                    path_ok = path_length_when_goal_set >= MIN_PATH_LENGTH_FOR_SUCCESS
                    travel_ok = travel_since >= MIN_TRAVEL_FOR_SUCCESS
                    if dist_to_goal < SUCCESS_DISTANCE:
                        if not (depth_ok and path_ok and travel_ok):
                            print(f"⚠️ 到达该点但条件不满足 (dist={dist_to_goal:.2f}m depth_ok={depth_ok} path_ok={path_ok} travel_ok={travel_ok})，继续探索")
                            current_goal_3d = None
                            current_goal_depth_used = None
                            path_length_when_goal_set = 0.0
                            goal_set_at_position = None
                            path_points = []
                            with shared_state.lock:
                                shared_state.latest_goal_uv = None
                                shared_state.latest_goal_depth = None
                                shared_state.latest_goal_camera_snapshot = None
                                shared_state.latest_status = "Searching..."
                        else:
                            print(f"📷 到达目标附近 (dist={dist_to_goal:.2f}m)，正在视觉验证...")
                            _, buf = cv2.imencode('.jpg', rgb)
                            try:
                                r = requests.post(CLOUD_URL, files={'image': ('img.jpg', buf.tobytes(), 'image/jpeg')},
                                                  data={'instruction': INSTRUCTION, 'verify': '1'}, timeout=8)
                                j = r.json()
                                if j.get('status') == 'success':
                                    print("\n" + "★" * 30)
                                    print("🎯 MISSION SUCCESS: 已到达目标附近，视觉验证通过!")
                                    print("★" * 30)
                                    plt.savefig("vln_final_trajectory.png")
                                    time.sleep(2)
                                    break
                            except Exception as e:
                                print(f"验证请求失败: {e}")
                            print("⚠️ 视觉验证未通过，继续探索")
                            current_goal_3d = None
                            current_goal_depth_used = None
                            path_length_when_goal_set = 0.0
                            goal_set_at_position = None
                            path_points = []
                            with shared_state.lock:
                                shared_state.latest_goal_uv = None
                                shared_state.latest_goal_depth = None
                                shared_state.latest_goal_camera_snapshot = None
                                shared_state.latest_status = "Searching..."

            # 4. 执行控制逻辑：有路径就沿路径走（含只剩 1 个点时继续朝目标走），无路径则转向探索
            if path_points:
                exploration_no_path_since = time.time()  # 有路径时重置卡住计时
                # 只剩 1 个点时该点即为目标，继续朝它走直到进入成功判定距离
                next_pt = path_points[0] if len(path_points) == 1 else path_points[1]
                curr_pos = agent.get_state().position
                move_vec = next_pt - curr_pos
                dist = np.linalg.norm(move_vec)
                
                if dist < 0.1:
                    path_points.pop(0)  # 到达当前路点，移除（剩 1 点时 pop 后为空，下一帧会进探索）
                else:
                    # 沿路径每步移动 MOVE_STEP_SIZE 米
                    agent_state = agent.get_state()
                    agent_state.position = curr_pos + (move_vec / dist) * MOVE_STEP_SIZE
                    agent.set_state(agent_state)
            else:
                # --- 探索模式：优先朝未探索空地前进（frontier），无 frontier 时才转向 ---
                had_goal = current_goal_3d is not None
                dist_to_goal = np.linalg.norm(curr_pos - current_goal_3d) if current_goal_3d is not None else 999
                if had_goal and dist_to_goal < SUCCESS_DISTANCE:
                    pass  # 接近目标，不转向，留给成功判定处理
                elif (not in_stuck_cooldown and EXPLORE_FRONTIER_ENABLED 
                      and step_count % EXPLORE_FRONTIER_INTERVAL == 0):
                    fgoal, fpath = get_frontier_goal(visited_cells, curr_pos, sim)
                    if fgoal is not None and fpath is not None:
                        current_goal_3d = fgoal
                        current_goal_depth_used = 2.0  # frontier 无深度含义，占位
                        path_length_when_goal_set = fpath.geodesic_distance
                        goal_set_at_position = np.array(curr_pos, dtype=np.float32)
                        goal_was_target_locked = False
                        goal_is_exploration = True
                        path_points = list(fpath.points)
                        exploration_no_path_since = time.time()
                        print(f"🧭 朝未探索区域前进 (距离: {fpath.geodesic_distance:.2f}m)")
                    else:
                        # 无 frontier 时只转向（进入此分支时不在 stuck 冷却期）
                        if step_count % 15 == 0:
                            agent.act("turn_right")
                            time.sleep(0.12)
                        if step_count % 45 == 0:
                            print("🔄 无 frontier，环视寻找...")
                        current_goal_3d = None
                        current_goal_depth_used = None
                        path_length_when_goal_set = 0.0
                        goal_set_at_position = None
                        goal_is_exploration = False
                        if had_goal:
                            with shared_state.lock:
                                shared_state.latest_goal_uv = None
                                shared_state.latest_goal_depth = None
                                shared_state.latest_goal_camera_snapshot = None
                                shared_state.latest_status = "Searching..."
                else:
                    turn_interval = 8 if in_stuck_cooldown else 15
                    if step_count % turn_interval == 0:
                        agent.act("turn_right")
                        time.sleep(0.12)
                    if step_count % 45 == 0:
                        print("🔄 无路径，环视寻找目标..." + (" (撞墙冷却中)" if in_stuck_cooldown else ""))
                    current_goal_3d = None
                    current_goal_depth_used = None
                    path_length_when_goal_set = 0.0
                    goal_set_at_position = None
                    goal_is_exploration = False
                    if had_goal:
                        with shared_state.lock:
                            shared_state.latest_goal_uv = None
                            shared_state.latest_goal_depth = None
                            shared_state.latest_goal_camera_snapshot = None
                            shared_state.latest_status = "Searching..."

                # 【轨迹记录 3】：可视化更新
            fig.canvas.restore_region(background)
            
           
            # 5. 可视化绘制 (保持 blitting 加速)
            fig.canvas.restore_region(background)
            current_pos = agent.get_state().position
            point_curr.set_data([current_pos[0]], [current_pos[2]])
            ax.draw_artist(point_curr)
             # 绘制历史轨迹 (红色细线)
            if len(actual_trajectory) > 1:
                traj_arr = np.array(actual_trajectory)
                line_traj.set_data(traj_arr[:, 0], traj_arr[:, 1])
                ax.draw_artist(line_traj)
                #----------------------------------------------------


            if path_points:
                path_arr = np.array(path_points)
                line_path.set_data(path_arr[:, 0], path_arr[:, 2])
                ax.draw_artist(line_path)
            
            if current_goal_3d is not None:
                point_goal.set_data([current_goal_3d[0]], [current_goal_3d[2]])
                ax.draw_artist(point_goal)

            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

            # OpenCV 第一视角显示
            viz_img = rgb.copy()
            if uv_goal: cv2.circle(viz_img, uv_goal, 10, (0, 255, 0), -1)
            display_status = "导航中(目标不在视野)" if (has_active_path and status == "Searching...") else status
            cv2.putText(viz_img, f"Status: {display_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Robot Eye", viz_img)

            # 【关键控制】：强制等待一个微小时间，防止 CPU 空转和旋转过快
            if cv2.waitKey(10) == ord('q'): break
            step_count += 1
            time.sleep(0.01) # 约 100 FPS

    finally:
        sim.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()