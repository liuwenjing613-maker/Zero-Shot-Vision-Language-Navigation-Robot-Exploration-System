# ==============================================================================
# 文件名: explore_manual.py
# 功能: 键盘手动探索场景，打印可用于 eval_episodes.json 的坐标（严谨处理）
# 用途: 阶段一采集 start_position / gt_position
# 坐标: 经 pathfinder.snap_point 落在 navmesh 上，与 evaluate_system 一致
# ==============================================================================

import os
import json
import habitat_sim
import numpy as np
import cv2

# ============== 配置（与 v2 一致，保证场景与评估一致）==============
SCENES_BASE = "/home/abc/ZeroShot_VLN/assets/scenes"
EVAL_EPISODES_JSON = "/home/abc/InternVLA/val/eval_episodes.json"  # 按 O 时自动追加到此文件
SCENE_LIST = [
    "MatterPort3D/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "MatterPort3D/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    "MatterPort3D/mp3d/8194nk5LbLH/8194nk5LbLH.glb",
    "MatterPort3D/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb",
    "MatterPort3D/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb",
    "MatterPort3D/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb",
    "MatterPort3D/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
    "MatterPort3D/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
    "MatterPort3D/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb",
    "MatterPort3D/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
    "MatterPort3D/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
]
SCENE_INDEX = 7  # 默认第一个场景，可改

# ---------- 出生地：二选一 ----------
# 为 True 时使用下方 FIXED_START_POSITION 作为出生地（会 snap 到 navmesh）
#
USE_FIXED_START = True
# 自定义出生地 [x, y, z]，需在可行走面上或附近（会自动 snap）
FIXED_START_POSITION = [1.26, 0.13, 3.78]  # 示例，可按场景修改

IMG_WIDTH, IMG_HEIGHT, HFOV = 640, 480, 110
# 坐标小数位数（与评估脚本 snap 后精度一致，便于复现）
COORD_DECIMALS = 2

# ============== 从场景路径解析 scene_id（供 JSON 使用）==============
def get_scene_id_from_path(scene_path):
    """从完整路径提取 scene_id，如 2azQ1b91cZZ、Z6MFQCViBuw"""
    base = os.path.basename(os.path.dirname(scene_path))
    if base and base != "scenes":
        return base
    name = os.path.splitext(os.path.basename(scene_path))[0]
    return name if name else "unknown"

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

# ============== 严谨坐标处理：取 agent 位置并 snap 到 navmesh ==============
def get_position_for_json(agent, sim):
    """
    返回可直接用于 eval_episodes.json 的 [x, y, z]。
    使用 pathfinder.snap_point 保证在可行走面上，与 evaluate_system.py 一致。
    """
    raw = np.array(agent.get_state().position, dtype=np.float64)
    if not sim.pathfinder.is_loaded:
        return [round(float(raw[0]), COORD_DECIMALS),
                round(float(raw[1]), COORD_DECIMALS),
                round(float(raw[2]), COORD_DECIMALS)]
    snapped = sim.pathfinder.snap_point(raw)
    if np.isnan(snapped).any():
        return [round(float(raw[0]), COORD_DECIMALS),
                round(float(raw[1]), COORD_DECIMALS),
                round(float(raw[2]), COORD_DECIMALS)]
    return [round(float(snapped[0]), COORD_DECIMALS),
            round(float(snapped[1]), COORD_DECIMALS),
            round(float(snapped[2]), COORD_DECIMALS)]

def format_position_line(label, pos):
    """生成 JSON 一行，便于复制到 eval_episodes.json"""
    return f'  "{label}": [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]'

def load_episodes():
    """加载 eval_episodes.json，若不存在或为空则返回 []"""
    if not os.path.exists(EVAL_EPISODES_JSON):
        return []
    try:
        with open(EVAL_EPISODES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []

def next_episode_id(episodes):
    """返回下一个可用的 episode_id：比现有最大 id 大 1，若无则 0"""
    if not episodes:
        return 0
    ids = [e.get("episode_id", i) for i, e in enumerate(episodes)]
    return max(ids) + 1

def append_episode_to_json(block):
    """将一条 episode 追加到 eval_episodes.json，编号自动为下一顺位"""
    episodes = load_episodes()
    new_id = next_episode_id(episodes)
    block["episode_id"] = new_id
    episodes.append(block)
    try:
        with open(EVAL_EPISODES_JSON, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2, ensure_ascii=False)
        return new_id, None
    except IOError as e:
        return None, str(e)

def main():
    scene_path = None
    for i in range(len(SCENE_LIST)):
        p = os.path.join(SCENES_BASE, SCENE_LIST[(SCENE_INDEX + i) % len(SCENE_LIST)])
        if os.path.exists(p):
            scene_path = p
            break
    if scene_path is None:
        print(f"未找到场景，请检查 SCENES_BASE: {SCENES_BASE}")
        return

    scene_id = get_scene_id_from_path(scene_path)
    print(f"场景: {scene_path}")
    print(f"scene_id (用于 JSON): {scene_id}")

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
        print(f"📍 出生地: 自定义 {FIXED_START_POSITION} → snap 后 {list(start_pos)}")
    else:
        start_pos = sim.pathfinder.get_random_navigable_point()
        print(f"📍 出生地: 随机")
    s = agent.get_state()
    s.position = start_pos
    agent.set_state(s)

    win_name = "Manual Explore (Phase 1)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, IMG_WIDTH, IMG_HEIGHT)

    help_text = [
        "W: forward  A: turn left  D: turn right",
        "S: 记录 start_position（先退后几米再按）",
        "G: 记录 gt_position（在目标旁按）",
        "O: 写入一条 episode 到 eval_episodes.json（编号自动递增）",
        "Q: quit",
    ]
    current_instruction = "chair"  # 当前目标名，按 O 时用；可改脚本顶部
    last_start_position = None
    last_gt_position = None

    print("\n" + "=" * 60)
    print("按键说明:")
    for line in help_text:
        print("  " + line)
    print(f"  写入文件: {EVAL_EPISODES_JSON}")
    print("  建议: 先开到目标旁按 G，再退后几米按 S，最后按 O 自动写入")
    print("=" * 60 + "\n")

    while True:
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3][..., ::-1]
        pos = get_position_for_json(agent, sim)

        # 显示
        disp = rgb.copy()
        cv2.putText(disp, f"pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, "S:start  G:gt  O:episode  Q:quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(50)
        if key < 0:
            continue
        key = chr(key).lower() if 0 <= key < 256 else ""

        if key == "q":
            break
        if key == "w":
            agent.act("move_forward")
            continue
        if key == "a":
            agent.act("turn_left")
            continue
        if key == "d":
            agent.act("turn_right")
            continue

        if key == "s":
            pos = get_position_for_json(agent, sim)
            last_start_position = pos
            line = format_position_line("start_position", pos)
            print("\n--- start_position (已记录，可再按 O 输出整条) ---")
            print(line)
            print("---\n")
            continue

        if key == "g":
            pos = get_position_for_json(agent, sim)
            last_gt_position = pos
            line = format_position_line("gt_position", pos)
            print("\n--- gt_position (已记录，可再按 O 输出整条) ---")
            print(line)
            print("---\n")
            continue

        if key == "o":
            start_pos = last_start_position
            gt_pos = last_gt_position
            if start_pos is None:
                start_pos = get_position_for_json(agent, sim)
                print("(未按过 S，start_position 用当前位置)")
            if gt_pos is None:
                gt_pos = get_position_for_json(agent, sim)
                print("(未按过 G，gt_position 用当前位置)")
            prompt_instruction = input("输入 instruction 后回车（直接回车则用当前默认）: ").strip()
            instruction = prompt_instruction if prompt_instruction else current_instruction
            block = {
                "scene_id": scene_id,
                "instruction": instruction,
                "start_position": start_pos,
                "gt_position": gt_pos,
            }
            new_id, err = append_episode_to_json(block)
            if err:
                print(f"\n--- 写入失败: {err} ---\n")
            else:
                block["episode_id"] = new_id
                print(f"\n--- 已写入 eval_episodes.json，episode_id = {new_id} ---")
                print(json.dumps(block, indent=2, ensure_ascii=False))
                print("---\n")
            continue

    sim.close()
    cv2.destroyAllWindows()
    print("已退出。")

if __name__ == "__main__":
    main()
