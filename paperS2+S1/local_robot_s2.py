# ==============================================================================
# 文件名: local_robot_s2.py
# 运行位置: 本地 Ubuntu 机器 (已安装 habitat-sim)
# 功能: InternVLA-N1 System2 + System1 导航 (基于 v3 框架)
# 特点: 
#   - System2 输出离散动作 (0=STOP, 1-4=TURN, 5=FORWARD)
#   - System1 输出连续轨迹 (waypoints)
#   - 保持 v3 的 FBE、短期记忆、俯视图可视化等功能
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
from topdown_visualizer import TopdownVisualizer

# ============== 1. 配置 ==============
CLOUD_URL = "http://127.0.0.1:5000/step"  # S2+S1 服务使用 /step 接口
CLOUD_PLAN_URL = "http://127.0.0.1:5000/plan"  # 兼容 v3 的 /plan 接口
CLOUD_RESET_URL = "http://127.0.0.1:5000/reset"
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

# 导航指令 (支持多目标序列)
INSTRUCTION = "the first flowers you see"

# 核心参数
MOVE_STEP_SIZE = 0.06
STUCK_DEPTH_THRESHOLD = 0.5
STUCK_FRAMES = 4
DEPTH_MIN, DEPTH_MAX = 0.3, 5.0
SUCCESS_DEPTH_THRESHOLD = 0.5

# 离散动作定义 (对应 InternVLA-N1 System2)
ACTION_STOP = 0
ACTION_TURN_LEFT_LARGE = 1   # 90°
ACTION_TURN_LEFT_SMALL = 2   # 15°
ACTION_TURN_RIGHT_SMALL = 3  # 15°
ACTION_TURN_RIGHT_LARGE = 4  # 90°
ACTION_MOVE_FORWARD = 5

# 推理间隔
INFER_INTERVAL = 0.6  # 秒

# FBE 配置 (保持 v3 一致)
FBE_EXPLORE_RADIUS = 2.5
FBE_MIN_DISTANCE = 1.2
FBE_SEARCHING_FRAMES = 350
FBE_BLIND_TURN_FRAMES = 60
FBE_SMOOTH_TURN_FRAMES = 12

# 起点配置
USE_FIXED_START = True
FIXED_START_POSITION = [8.23, 0.13, 3.41]

# GT 目标位置
GT_TARGET_POSITION = [14.36, 0.13, 2.01]

# ============== 2. 共享状态 ==============
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_rgb = None
        self.current_depth = None
        self.current_agent_state = None
        self.new_image_ready = False
        
        # S2 输出
        self.latest_action = None  # 离散动作列表
        self.latest_trajectory = None  # 轨迹点 (System1)
        self.latest_pixel_goal = None  # 像素目标
        self.latest_stop = False  # 是否停止
        self.latest_status = "Searching..."
        
        # 指令相关
        self.current_instruction = INSTRUCTION
        self.target_list = []
        self.target_index = 0

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

# ============== 6. FBE 前沿探索 ==============
def get_explore_waypoint(sim, curr_pos):
    """使用 pathfinder 获取距离当前位置 2-3m 的随机可达点"""
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
    except Exception as e:
        print(f"⚠️ FBE 探索点获取失败: {e}")
    return None

# ============== 7. 云端请求 (同步) ==============
def request_cloud_step(rgb_bgr, instruction, timeout=15):
    """
    向云端发送 RGB 图像和指令，获取 S2+S1 输出
    
    Returns:
        dict: {
            'action': 离散动作列表,
            'trajectory': 轨迹点 (可选),
            'pixel_goal': 像素目标 (可选),
            'stop': 是否停止
        }
    """
    try:
        _, buf = cv2.imencode('.jpg', rgb_bgr)
        r = requests.post(
            CLOUD_URL,
            files={'rgb': ('img.jpg', buf.tobytes(), 'image/jpeg')},
            data={'instruction': instruction},
            timeout=timeout,
        )
        return r.json()
    except Exception as e:
        print(f"⚠️ 云端请求失败: {e}")
        return {'action': [ACTION_MOVE_FORWARD], 'stop': False}

def reset_cloud_agent():
    """重置云端 agent 状态"""
    try:
        requests.post(CLOUD_RESET_URL, timeout=5)
        print("✅ 云端 agent 已重置")
    except Exception as e:
        print(f"⚠️ 重置云端失败: {e}")

# ============== 8. 云端 worker ==============
def cloud_worker():
    last_send = 0.0
    while True:
        now = time.time()
        if now - last_send < INFER_INTERVAL:
            time.sleep(0.1)
            continue
        
        img_to_send = None
        instruction = None
        
        with shared_state.lock:
            if shared_state.new_image_ready and shared_state.current_rgb is not None:
                img_to_send = shared_state.current_rgb.copy()
                shared_state.new_image_ready = False
                instruction = shared_state.current_instruction
        
        if img_to_send is not None:
            last_send = time.time()
            # RGB -> BGR for cv2.imencode
            img_bgr = img_to_send[..., ::-1] if img_to_send.shape[-1] == 3 else img_to_send
            
            result = request_cloud_step(img_bgr, instruction)
            
            with shared_state.lock:
                shared_state.latest_action = result.get('action', [ACTION_MOVE_FORWARD])
                shared_state.latest_trajectory = result.get('trajectory')
                shared_state.latest_pixel_goal = result.get('pixel_goal')
                shared_state.latest_stop = result.get('stop', False)
                
                # 根据动作更新状态
                if shared_state.latest_stop:
                    shared_state.latest_status = "Success"
                elif shared_state.latest_pixel_goal is not None:
                    shared_state.latest_status = "Target Locked"
                elif shared_state.latest_action and ACTION_MOVE_FORWARD in shared_state.latest_action:
                    shared_state.latest_status = "Navigating"
                else:
                    shared_state.latest_status = "Searching..."
                
                print(f"📍 Action: {shared_state.latest_action}, Stop: {shared_state.latest_stop}, Status: {shared_state.latest_status}")
        
        time.sleep(0.1)

# ============== 9. 执行离散动作 ==============
def execute_action(agent, action_id):
    """
    执行单个离散动作
    
    Args:
        agent: Habitat agent
        action_id: 动作 ID
            0 = STOP
            1 = TURN_LEFT_LARGE (90°)
            2 = TURN_LEFT_SMALL (15°)
            3 = TURN_RIGHT_SMALL (15°)
            4 = TURN_RIGHT_LARGE (90°)
            5 = MOVE_FORWARD
    """
    if action_id == ACTION_STOP:
        return True  # 返回 True 表示应该停止
    elif action_id == ACTION_TURN_LEFT_LARGE:
        # 左转 90°: 约 9 次 turn_left (每次 ~10°)
        for _ in range(9):
            agent.act("turn_left")  # turn_left
    elif action_id == ACTION_TURN_LEFT_SMALL:
        # 左转 15°: 约 1-2 次 turn_left
        agent.act("turn_left")
    elif action_id == ACTION_TURN_RIGHT_SMALL:
        # 右转 15°: 约 1-2 次 turn_right
        agent.act("turn_right")  # turn_right
    elif action_id == ACTION_TURN_RIGHT_LARGE:
        # 右转 90°: 约 9 次 turn_right
        for _ in range(9):
            agent.act("turn_right")
    elif action_id == ACTION_MOVE_FORWARD:
        # 前进
        agent.act("move_forward")  # move_forward
    
    return False

# ============== 10. 主函数 ==============
def main():
    global INSTRUCTION
    
    target_list = parse_instruction_sequence(INSTRUCTION)
    with shared_state.lock:
        shared_state.target_list = target_list
        shared_state.target_index = 0
        shared_state.current_instruction = target_list[0]
    
    # 查找场景
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
    
    # 初始化 Habitat
    sim = habitat_sim.Simulator(make_cfg(scene_path))
    nav = habitat_sim.NavMeshSettings()
    nav.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, nav)
    agent = sim.initialize_agent(0)
    
    # 设置起点
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
    
    # 重置云端 agent
    reset_cloud_agent()
    
    # 启动云端 worker
    threading.Thread(target=cloud_worker, daemon=True).start()
    print("✅ 云端 worker 已启动 (InternVLA-N1 System2+System1)")
    
    # 初始化窗口
    cv2.namedWindow("Robot Eye", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Eye", 640, 480)
    
    # 初始化俯视图
    print("📍 正在生成场景俯视图...")
    topdown_viz = TopdownVisualizer(sim, meters_per_pixel=0.05)
    fig, ax = topdown_viz.init_matplotlib_figure(figsize=(10, 10))
    plt.ion()
    plt.show(block=False)
    print("✅ 场景俯视图已就绪")
    
    # 状态变量
    trajectory = [[start_pos[0], start_pos[2]]]
    step_count = 0
    stuck_count = 0
    searching_frames = 0
    action_queue = []  # 待执行的动作队列
    current_target_idx = 0
    
    # FBE 状态
    in_fbe_mode = False
    path_points = []
    smooth_turn_remaining = 0
    
    print(f"🚀 开始导航 (InternVLA-N1 S2+S1, 按 q 退出)")
    
    # 先显示初始画面，等待 6 秒
    print("⏳ 窗口已就绪，导航将在 6 秒后开始...")
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3][..., ::-1]
    curr_state = agent.get_state()
    curr_pos = np.array(curr_state.position)
    
    viz = rgb.copy()
    cv2.putText(viz, "Starting in 6s...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(viz, f"Goal: {target_list[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Robot Eye", viz)
    
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
            
            # 更新共享状态
            with shared_state.lock:
                shared_state.current_rgb = rgb
                shared_state.current_depth = depth
                shared_state.current_agent_state = curr_state
                shared_state.new_image_ready = True
                
                action = shared_state.latest_action
                stop = shared_state.latest_stop
                status = shared_state.latest_status
                pixel_goal = shared_state.latest_pixel_goal
            
            trajectory.append([curr_pos[0], curr_pos[2]])
            step_count += 1
            
            # 检查是否应该停止
            if stop:
                current_target_name = target_list[current_target_idx] if current_target_idx < len(target_list) else None
                print("\n" + "★" * 20 + f"\n🎯 目标 {current_target_name} 达成!\n" + "★" * 20)
                
                current_target_idx += 1
                if current_target_idx >= len(target_list):
                    print("🎉 多目标序列全部完成!")
                    plt.savefig("vln_success.png")
                    time.sleep(2)
                    break
                
                # 更新到下一个目标
                with shared_state.lock:
                    shared_state.target_index = current_target_idx
                    shared_state.current_instruction = target_list[current_target_idx]
                    shared_state.latest_stop = False
                    shared_state.latest_action = None
                
                reset_cloud_agent()
                action_queue = []
                time.sleep(0.5)
                continue
            
            # 撞墙检测
            center = depth[IMG_HEIGHT//2-30:IMG_HEIGHT//2+30, IMG_WIDTH//2-30:IMG_WIDTH//2+30]
            valid = center[(center > 0.1) & (center < 5.0)]
            front_d = float(np.median(valid)) if len(valid) > 0 else 2.0
            
            if front_d < STUCK_DEPTH_THRESHOLD:
                stuck_count += 1
            else:
                stuck_count = 0
            
            # 撞墙处理：触发 FBE
            if stuck_count >= STUCK_FRAMES and not in_fbe_mode:
                action_queue = []  # 清空动作队列
                explore_waypoint = get_explore_waypoint(sim, curr_pos)
                if explore_waypoint is not None:
                    print("🔍 FBE: 前往探索点")
                    path = habitat_sim.ShortestPath()
                    path.requested_start = sim.pathfinder.snap_point(curr_pos)
                    path.requested_end = explore_waypoint
                    if sim.pathfinder.find_path(path) and len(path.points) > 1:
                        path_points = list(path.points)
                        in_fbe_mode = True
                        stuck_count = 0
                else:
                    smooth_turn_remaining = 4
                    stuck_count = 0
            
            # 平滑转向 (撞墙无探索点时)
            if smooth_turn_remaining > 0:
                agent.act("turn_right")  # turn_right
                smooth_turn_remaining -= 1
            # FBE 模式：跟踪路径
            elif in_fbe_mode and path_points:
                next_pt = path_points[0]
                dist_to_pt = np.linalg.norm(curr_pos - next_pt)
                
                if dist_to_pt < 0.3:
                    path_points.pop(0)
                    if not path_points:
                        in_fbe_mode = False
                        smooth_turn_remaining = FBE_SMOOTH_TURN_FRAMES
                        print("🔍 FBE: 到达探索点，环视中")
                else:
                    # 计算朝向
                    dx = next_pt[0] - curr_pos[0]
                    dz = next_pt[2] - curr_pos[2]
                    target_yaw = math.atan2(dx, dz)
                    current_yaw = get_agent_forward_yaw(curr_state)
                    yaw_diff = target_yaw - current_yaw
                    while yaw_diff > math.pi:
                        yaw_diff -= 2 * math.pi
                    while yaw_diff < -math.pi:
                        yaw_diff += 2 * math.pi
                    
                    if abs(yaw_diff) > 0.15:
                        if yaw_diff > 0:
                            agent.act("turn_right")  # turn_right
                        else:
                            agent.act("turn_left")  # turn_left
                    else:
                        agent.act("move_forward")  # move_forward
            # 正常导航：执行 S2 输出的动作
            elif action and len(action) > 0:
                # 如果动作队列为空，添加新动作
                if not action_queue:
                    action_queue = list(action)
                
                if action_queue:
                    next_action = action_queue.pop(0)
                    should_stop = execute_action(agent, next_action)
                    if should_stop:
                        with shared_state.lock:
                            shared_state.latest_stop = True
            else:
                # 默认前进
                agent.act("move_forward")
            
            # Searching 帧计数
            if status == "Searching...":
                searching_frames += 1
            else:
                searching_frames = 0
            
            # Searching 过久触发 FBE
            if searching_frames >= FBE_SEARCHING_FRAMES and not in_fbe_mode:
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
            
            # 可视化
            viz = rgb.copy()
            current_target_name = target_list[current_target_idx] if current_target_idx < len(target_list) else "Done"
            cv2.putText(viz, f"Goal: {current_target_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == "Target Locked" else (255, 255, 0), 2)
            cv2.putText(viz, f"Step: {step_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(viz, f"Action: {action}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 绘制像素目标
            if pixel_goal is not None:
                u, v = int(pixel_goal[1]), int(pixel_goal[0])
                cv2.circle(viz, (u, v), 10, (0, 0, 255), 2)
                cv2.line(viz, (u-15, v), (u+15, v), (0, 0, 255), 2)
                cv2.line(viz, (u, v-15), (u, v+15), (0, 0, 255), 2)
            
            if in_fbe_mode:
                cv2.putText(viz, "FBE MODE", (IMG_WIDTH - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow("Robot Eye", viz)
            
            # 更新俯视图
            topdown_viz.update_matplotlib(
                trajectory=trajectory,
                current_pos=curr_pos,
                start_pos=start_pos,
                current_yaw=get_agent_forward_yaw(curr_state),
                title=f"InternVLA-N1 S2+S1 Navigation",
                instruction=current_target_name,
                status=status,
                gt_pos=GT_TARGET_POSITION
            )
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("👋 用户退出")
                break
            
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    finally:
        cv2.destroyAllWindows()
        plt.close('all')
        sim.close()

if __name__ == "__main__":
    main()
