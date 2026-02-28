# VLN Simple 版 - 完整详细流程

> 供分析合理性使用

---

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  本地 (local_robot_system1.py)                                        │
│  - Habitat 仿真环境                                                   │
│  - 主循环 ~50 FPS (每帧 0.02s)                                        │
│  - cloud_worker 后台线程                                               │
└────────────────────────────┬──────────────────────────────────────────┘
                             │ HTTP POST /plan
                             │ (image + instruction [+ verify=1])
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  云端 (cloud_brain_server.py)                                         │
│  - Qwen2.5-VL-7B-Instruct                                            │
│  - 两种模式: 目标定位 (inference) / 到达验证 (verify)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、启动流程

### 2.1 初始化

| 步骤 | 动作 |
|------|------|
| 1 | 从 SCENE_LIST 选第一个存在的 .glb 场景 |
| 2 | 加载 Habitat、生成 NavMesh |
| 3 | agent 置于随机可导航点 `start_pos` |
| 4 | 启动 `cloud_worker` 后台线程 |
| 5 | 初始化可视化窗口 (OpenCV + Matplotlib) |
| 6 | 初始化: `path_points=[]`, `current_goal_3d=None`, `stuck_count=0`, `step_count=0` |

### 2.2 无初始扫描

- 不进行 360° 扫描
- 直接进入主循环，等待云端首次返回目标

---

## 三、双线程数据流

### 3.1 主循环 (主线程)

每帧 (~0.02s) 执行：

1. 获取 `rgb`, `depth`, `curr_state`
2. 写入 `shared_state`: `current_rgb`, `current_depth`, `current_agent_state`, `new_image_ready=True`
3. 读取 `shared_state`: `uv`, `status`, `snap_d`, `snap_c`（来自云端 worker 上次更新）
4. 记录轨迹 `trajectory.append(...)`

### 3.2 cloud_worker (后台线程)

```
while True:
    if 距上次发送 < CLOUD_SEND_INTERVAL (1.2s):
        sleep(0.3); continue
    if new_image_ready:
        复制 current_rgb, current_depth, camera_snapshot
        new_image_ready = False
        POST /plan (无 verify)
        解析返回:
            success + (u,v) → Target Locked, 写入 shared_state
            Inferred + (u,v) → Inferred, 写入 shared_state
            其他 → Searching...
    sleep(0.3)
```

**注意**：主循环与 cloud_worker 共享的是**同一时刻**的 rgb/depth/camera，因为 worker 复制时主循环会置 `new_image_ready=False`，下一帧才再置 True。但 worker 发送的是**上一帧**的画面（有 ~1.2s 延迟）。

---

## 四、主循环每帧详细逻辑

### 4.1 撞墙检测

```
取图像中心 60×60 区域深度中位数 → front_d

if front_d < 0.6m:
    stuck_count += 1
else:
    stuck_count = 0

if stuck_count >= 6:
    清空 path_points, current_goal_3d
    清空 shared_state 中的 goal_uv/depth/camera
    执行 4 次 turn_right，每次间隔 TURN_DELAY (1s)
```

### 4.2 规划逻辑

**触发条件**：`not has_path` 且 `use_goal`

- `has_path` = path_points 存在且 len > 1
- `use_goal` = uv 存在 且 snap_d 存在 且 snap_c 存在

**执行**：

```
goal_3d, gdepth = get_3d_point(uv, snap_d, snap_c)
  - 用 snap_d 在 (u,v) 处取 5×5 邻域深度中值
  - 深度需在 [0.3, 5.0] 米
  - 相机坐标系 → 世界坐标系 (用 snap_c 位姿)
  - snap_point 吸附到地面

if goal_3d 有效:
    replan = 无当前目标 或 新目标与当前目标距离 > 1.5m
    if replan:
        A* find_path(goal_3d)
        if 找到路径:
            path_points = path.points
            current_goal_3d = goal_3d
            goal_set_at = curr_pos
```

### 4.3 成功判定（验证）

**触发条件**：`step_count > 0` 且 `step_count % 40 == 0`

即每 40 步 (~0.8s) 触发一次。

**执行**：

```
POST /plan, verify=1
云端 verify_logic:
    prompt: "This is a first-person view from a robot navigating to find a {target}.
             Has the robot reached the target {target}?
             Is the robot now at/near the {target}? Answer only YES or NO."
    模型输出包含 "YES" → 返回 status=success

if 云端返回 success:
    打印任务成功，break
```

**无其他条件**：不检查距离、不检查移动量、不检查深度，完全依赖模型推理。

### 4.4 执行控制

**有路径时**：

```
next_pt = path_points[0] (若 len==1) 或 path_points[1] (若 len>1)
dist = ||next_pt - curr_pos||

if dist < 0.1:
    path_points.pop(0)
else:
    position = curr_pos + (move_vec / dist) * 0.12  # MOVE_STEP_SIZE
    agent.set_state(position)
```

**无路径时**：

```
agent.act("turn_right")
sleep(TURN_DELAY)  # 1s
```

### 4.5 可视化与步进

- 更新 Matplotlib 鸟瞰图（轨迹、规划、当前位置、目标）
- OpenCV 显示第一视角 + status
- `step_count += 1`
- `sleep(0.02)`

---

## 五、云端 API 详解

### 5.1 POST /plan（目标定位模式，verify=0）

**输入**：image, instruction

**inference_logic**：

- 目标可见 → `PIXEL: (u, v)` → status=success
- 目标不可见 → `INFER: (u, v)` → status=fail, message=Inferred
- 无法解析 → status=fail, message=SEARCHING

**返回**：`{status, u, v, message, confidence}`

### 5.2 POST /plan（验证模式，verify=1）

**输入**：image, instruction, verify=1

**verify_logic**：

- prompt 询问模型：机器人是否已到达目标
- 输出包含 "YES" → status=success
- 否则 → status=fail

---

## 六、关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| CLOUD_SEND_INTERVAL | 1.2s | 云端目标更新间隔 |
| MOVE_STEP_SIZE | 0.12m | 每步移动距离 |
| REPLAN_DISTANCE_THRESHOLD | 1.5m | 重规划距离阈值 |
| STUCK_DEPTH_THRESHOLD | 0.6m | 撞墙深度 |
| STUCK_FRAMES | 6 | 撞墙判定连续帧数 |
| DEPTH_MIN/MAX | 0.3~5.0m | 有效目标深度 |
| VERIFY_INTERVAL_STEPS | 40 | 每 40 步验证一次 |
| TURN_DELAY | 1s | 每次转向间隔 |
| 主循环 sleep | 0.02s | ~50 FPS |

---

## 七、数据流时序示意

```
时间线 (主循环每 0.02s 一帧):

t=0.0   主循环: 写 shared_state, 读 uv/snap (可能为空)
        cloud_worker: 等待 1.2s 间隔

t=0.02  主循环: 继续...
t=0.04  ...
...
t=1.2   cloud_worker: 取 t≈0 时的 rgb/depth, POST 云端
        主循环: 继续运行，uv 仍为旧值或空

t=1.2+ 云端推理完成 (~2-5s 取决于 GPU)
        cloud_worker: 收到 (u,v), 写入 shared_state

t=1.2+ 主循环: 下一帧读到新 uv, 若无路径则规划 A*
```

**潜在问题**：规划用的 `snap_d`/`snap_c` 是 worker 发送请求时的快照，但收到响应时机器人可能已移动，位姿已变。代码用 `camera_snapshot` 保证像素→3D 时用**发送时的位姿**，所以 3D 点是在**当时相机坐标系**下的正确投影，再变换到世界坐标。只要 snap 与 (u,v) 同帧，逻辑正确。

---

## 八、合理性分析要点

### 8.1 可能的问题

1. **验证与规划解耦**：验证每 40 步触发，与是否到达目标点无关，可能在远处就误判成功（若模型过于宽松）。
2. **TURN_DELAY=1s**：无路径时每次转向等 1s，主循环 0.02s，导致一帧内大量时间在 sleep，转向很慢。
3. **Inferred 无过滤**：推理方向可能指向墙或错误方向，仍会规划并执行。
4. **撞墙后清空目标**：撞墙后清空 shared_state 的 goal，需等待下次 worker 返回才有新目标，期间只能转向。
5. **path_points 只剩 1 个点**：`next_pt = path_points[0]`，即目标点本身，会朝目标点移动直到 dist<0.1 才 pop，此时 path 变空，下一帧进入「无路径」转向。

### 8.2 设计优点

1. **成功判定简单**：完全由模型判断，无需手工规则。
2. **无前沿探索**：逻辑简单，无 visited_cells 等状态。
3. **Target Locked 与 Inferred 一视同仁**：都用于规划，提高探索效率。

---

## 九、完整流程简图

```
启动 → 加载场景 → 随机起点 → 启动 cloud_worker

主循环 (每 0.02s):
  ├─ 获取 rgb/depth
  ├─ 写 shared_state, 读 uv/snap/status
  ├─ 撞墙? → 清空+转向
  ├─ 无路径 且有 uv? → 像素→3D → A* → 设 path
  ├─ step_count % 40 == 0? → verify → YES 则成功退出
  ├─ 有 path? → 沿 path 移动 : 转向
  ├─ 可视化
  └─ step_count++, sleep(0.02)

cloud_worker (每 1.2s):
  ├─ 取 shared_state 的 rgb/depth/camera
  ├─ POST /plan (无 verify)
  └─ 解析 (u,v) → 写 shared_state
```
