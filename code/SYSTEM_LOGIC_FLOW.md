# VLN 双系统导航 - 完整逻辑流程文档

## 一、系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        本地 (System 1)                             │
│  Habitat 仿真 + A* 路径规划 + 坐标转换 + 实时可视化 + 主控制循环     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (图像 + instruction)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        云端 (System 2)                             │
│  Qwen2.5-VL 视觉语言模型：目标定位 / 推理方向 / 到达验证            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、启动流程

### 2.1 初始化

1. **场景选择**：根据 `SCENE_CHOICE` 选择场景（random / first / 整数索引）
2. **指令选择**：根据 `INSTRUCTION_CHOICE` 选择目标物体（如 chair, table）
3. **Habitat 初始化**：加载 .glb 场景、生成 NavMesh、初始化 agent
4. **起点设置**：`INIT_POSITION_MODE` 为 random 或 fixed

### 2.2 初始 360° 扫描（SCAN_ENABLED=True 时）

```
for i in 1..4:
    1. 获取当前 rgb、depth、camera_snapshot
    2. 发送到云端，data={'instruction': INSTRUCTION, 'scan': '1'}
    3. 云端返回 (u, v)、confidence、status(Target Locked / Inferred)
    4. 存储到 views 列表
    5. 转向 90°（SCAN_TURNS_PER_90 次 turn_right，每次间隔 SCAN_TURN_DELAY）

选择最佳视角：优先 Target Locked，其次按 confidence 最高
将最佳结果写入 shared_state（latest_goal_uv, latest_goal_depth, latest_goal_camera_snapshot, latest_status）
exploration_no_path_since = now - 20  # 视为已探索足够久，直接使用扫描结果
```

### 2.3 启动 system2_worker 线程

- 后台线程，每 `CLOUD_SEND_INTERVAL`（3.5s）向云端发送当前帧
- 更新 shared_state

---

## 三、主循环（每帧 ~100 FPS）

### 3.1 数据获取与共享

```
1. 获取 obs（rgb, depth）
2. 写入 shared_state：current_rgb, current_depth, current_agent_state, new_image_ready=True
3. 读取 shared_state：uv_goal, status, snap_depth, snap_camera, in_stuck_cooldown
4. 记录轨迹：actual_trajectory.append(curr_pos)
5. 更新 visited_cells：按 VISITED_CELL_SIZE(0.6m) 格网记录已访问格子
```

### 3.2 撞墙检测

```
front_depth = 图像中心 50x50 区域的有效深度中位数

if front_depth < STUCK_DEPTH_THRESHOLD(0.5m):
    stuck_count += 1
else:
    stuck_count = 0

if stuck_count >= STUCK_FRAMES(5):
    【撞墙处理】
    - path_points = []
    - current_goal_3d = None
    - 执行 4 次 turn_right，每次间隔 0.08s
    - 清空 shared_state 中的 goal_uv/depth/camera
    - ignore_inferred_until = now + 5s     # 5 秒内忽略推理目标
    - stuck_explore_cooldown_until = now + 3s  # 3 秒内只转向，不尝试 frontier
```

### 3.3 规划逻辑

**条件**：`not has_active_path` 且 `use_cloud_goal` 为真

```
has_active_path = path_points 存在且 len > 1

use_cloud_goal = uv_goal 存在 且 snap_depth 存在 且 snap_camera 存在

若 status == "Inferred Direction" 且 (now - exploration_no_path_since) < INFER_USE_AFTER_SEARCH_SEC(12s):
    use_cloud_goal = False  # 探索时间不够，暂不用推理目标
```

**执行**：

```
1. get_3d_point(uv, snap_depth, snap_camera) → goal_3d, goal_depth_used
   - 若 status=="Inferred" 且 goal_depth_used < INFER_MIN_DEPTH(0.45m)：跳过（指向墙）
   - 若 current_goal_3d 为空 或 新目标与当前目标距离 > REPLAN_DISTANCE_THRESHOLD(1.2m)：
2. A* 路径规划：find_path(goal_3d)
   - 成功：设置 path_points, current_goal_3d, goal_was_target_locked, goal_is_exploration=False
   - 失败：打印 "路径不可达"
```

### 3.4 成功判定

**条件**：`current_goal_3d` 存在 且 `goal_set_at_position` 存在

```
dist_to_goal = ||curr_pos - current_goal_3d||

【情况 A】goal_is_exploration == True（frontier 探索目标）
    if dist_to_goal < SUCCESS_DISTANCE(0.6m):
        清空目标、path_points，不验证

【情况 B】goal_is_exploration == False（云端目标）
    depth_ok = 0.5 <= goal_depth_used <= 4.0
    path_ok = path_length_when_goal_set >= 0.2m
    travel_ok = 从设目标起至少走了 0.2m

    if dist_to_goal < SUCCESS_DISTANCE:
        if not (depth_ok and path_ok and travel_ok):
            清空目标，打印条件不满足
        else:
            发送当前帧到云端 verify=1
            if 云端返回 success：
                → 任务成功，退出循环
            else：
                清空目标，继续探索
```

### 3.5 执行控制

**有路径时**（path_points 非空）：

```
exploration_no_path_since = now

next_pt = path_points[0]（若 len==1）或 path_points[1]（若 len>1）
dist = ||next_pt - curr_pos||

if dist < 0.1:
    path_points.pop(0)  # 到达路点，移除
else:
    沿 next_pt 方向移动 MOVE_STEP_SIZE(0.15m)
```

**无路径时**（探索模式）：

```
had_goal = current_goal_3d 存在
dist_to_goal = 到目标距离（若无目标则为 999）

【分支 1】had_goal 且 dist_to_goal < SUCCESS_DISTANCE
    pass  # 不转向，留给成功判定处理

【分支 2】不在 stuck 冷却期 且 EXPLORE_FRONTIER_ENABLED 且 step_count % 8 == 0
    fgoal, fpath = get_frontier_goal(visited_cells, curr_pos, sim)
    if 找到 frontier：
        设置 path_points, current_goal_3d, goal_is_exploration=True
        打印 "朝未探索区域前进"
    else：
        每 15 步 turn_right
        清空目标

【分支 3】其他
    turn_interval = 8（若 in_stuck_cooldown）else 15
    每 turn_interval 步 turn_right
    清空目标
```

### 3.6 Frontier 探索

```
get_frontier_goal():
    frontiers = 已访问格子的四邻接未访问格子
    for each frontier:
        goal_3d = 格子中心，snap 到地面
        if find_path(goal_3d) 且 路径长度 > 0.15m：
            选择路径最长的 frontier（探索更充分）
    return (goal_3d, path) 或 (None, None)
```

### 3.7 可视化与循环

```
- 更新 matplotlib 轨迹图
- OpenCV 显示第一视角 + status
- step_count += 1
- time.sleep(0.01)
```

---

## 四、云端 (System 2) 逻辑

### 4.1 API：POST /plan

**参数**：
- `image`：当前帧图像
- `instruction`：目标物体名（如 chair）
- `verify`：'1' 或 '0'：是否到达验证模式
- `scan`：'1' 或 '0'：是否扫描模式

### 4.2 模式一：验证模式（verify=1）

```
verify_logic(image, target_name):
    prompt: "Is the {target} clearly visible in this image? Answer only YES or NO."
    return "YES" in output
```

### 4.3 模式二：扫描模式（scan=1）

```
prompt 要求：PIXEL/INFER + CONFIDENCE: 0.0-1.0
返回：{u, v, status, confidence}
```

### 4.4 模式三：普通模式

```
inference_logic:
    目标可见 → PIXEL: (u, v) → status: success
    目标不可见 → INFER: (u, v) → status: fail, message: Inferred
    无法解析 → 返回 SEARCHING
```

---

## 五、system2_worker 线程

```
while True:
    if now - last_send_time < CLOUD_SEND_INTERVAL(3.5s):
        sleep(0.2); continue

    if new_image_ready:
        复制 current_rgb, current_depth, current_agent_state
        new_image_ready = False
        POST 到云端（无 scan、无 verify）
        根据返回更新 shared_state：
            - success → Target Locked, 写入 uv/depth/camera
            - Inferred → 若未在 ignore_inferred_until 内 且 非 Target Locked，写入 Inferred Direction
            - 其他 → 仅 status = Searching...

    sleep(0.2)
```

---

## 六、关键状态变量

| 变量 | 含义 |
|------|------|
| path_points | A* 路径点列表，空则无路径 |
| current_goal_3d | 当前目标 3D 点 |
| goal_was_target_locked | 目标是否来自「真实看到」 |
| goal_is_exploration | 是否 frontier 探索目标 |
| goal_set_at_position | 设目标时的机器人位置 |
| exploration_no_path_since | 上次有路径的时间 |
| stuck_count | 连续检测到前方深度 < 0.5m 的帧数 |
| visited_cells | 已访问格子集合 |
| shared_state | 与云端共享：uv/depth/camera/status/ignore_inferred_until/stuck_explore_cooldown_until |

---

## 七、配置参数速查

| 参数 | 默认值 | 含义 |
|------|--------|------|
| CLOUD_SEND_INTERVAL | 3.5s | 向云端发送间隔 |
| MOVE_STEP_SIZE | 0.15m | 每步移动距离 |
| REPLAN_DISTANCE_THRESHOLD | 1.2m | 重规划距离阈值 |
| STUCK_DEPTH_THRESHOLD | 0.5m | 撞墙深度阈值 |
| STUCK_FRAMES | 5 | 撞墙判定连续帧数 |
| STUCK_IGNORE_INFERRED_SEC | 5s | 撞墙后忽略推理目标时长 |
| STUCK_EXPLORE_COOLDOWN_SEC | 3s | 撞墙后只转向、不 frontier 时长 |
| VISITED_CELL_SIZE | 0.6m | 已探索格子大小 |
| INFER_USE_AFTER_SEARCH_SEC | 12s | 探索多久后才用推理目标 |
| INFER_MIN_DEPTH | 0.45m | 推理目标最小深度阈值 |
| SUCCESS_DISTANCE | 0.6m | 成功判定距离 |
| EXPLORE_FRONTIER_INTERVAL | 8 | 每 N 步尝试 frontier |
| SCAN_VIEWS | 4 | 初始扫描视角数 |
| SCAN_TURNS_PER_90 | 3 | 每 90° 转向次数 |

---

## 八、完整流程简图

```
启动
  → 初始化 Habitat + 场景 + 指令
  → [可选] 初始 360° 扫描：4 视角 → 选最佳 → 写入 shared_state
  → 启动 system2_worker 线程

主循环（每帧）:
  1. 获取 rgb/depth，推 shared_state
  2. 撞墙检测 → 若撞墙：清路径、转 4 次、设冷却
  3. 规划：若无路径 且有云端目标 → 用 uv+snap 算 3D → A* → 设 path
  4. 成功判定：若接近目标 → frontier 直接清空 / 云端目标做视觉验证
  5. 执行：
     - 有路径 → 沿路径移动
     - 无路径 → 尝试 frontier 或 转向
  6. 可视化
```
