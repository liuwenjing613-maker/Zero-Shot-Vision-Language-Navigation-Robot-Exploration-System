# local_robot_simple_v2 完整流程说明

> 含 FBE 前沿探索、短期记忆防死锁、多目标序列；流程按主循环执行顺序组织，逻辑闭环。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  主线程 (约 50Hz)                                                 │
│  每帧: 取图 → 撞墙/FBE/成功/短期记忆/规划 → 执行动作 → 可视化       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 读写
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  SharedState (与云端 worker 共享)                                 │
│  current_rgb/depth/agent_state, latest_goal_uv/depth/snapshot,   │
│  latest_status (Target Locked | Inferred | Searching...),        │
│  current_instruction, target_list, target_index,                │
│  latest_instruction_used                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 写
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  云端 worker 线程 (每 CLOUD_SEND_INTERVAL 发一帧到 VLM)            │
│  请求 /plan，写回 uv、status、snapshot 等                         │
└─────────────────────────────────────────────────────────────────┘
```

- **状态来源**：主循环每帧从 `SharedState` 读 `status`、`uv`、`snap_d`、`snap_c`，决定是否规划、是否成功、是否 FBE、如何动作。
- **规划与执行**：仅当 **Target Locked** 且拿到有效 3D 目标时才做 A*；有路径则**直接沿路径向目标移动**，无路径则按 Inferred 转向或 Searching 转向/FBE。

---

## 二、主循环每帧执行顺序（逻辑顺序）

下面按**代码执行顺序**列出每一块的职责与前后依赖，保证“先判停、再判 FBE、再规划、再执行”的闭环。

### 1. 取观测与写共享状态

- 从 sim 取 `rgb`、`depth`、`curr_state`。
- 写回 `SharedState`：`current_rgb/depth/agent_state`、`new_image_ready = True`。
- 本帧读取：`uv, status, snap_d, snap_c, latest_instruction_used`（供后续成功判定与规划使用）。

### 2. 撞墙检测

- 取视野中心区域深度中值 `front_d`。
- `front_d < STUCK_DEPTH_THRESHOLD` 则 `stuck_count += 1`，否则置 0。
- **若 `stuck_count >= STUCK_FRAMES`**：视为撞墙，进入“撞墙处理”（见下节 FBE 触发 ①）。

### 3. FBE 相关（详见第三节）

- **FBE 中断**：若当前处于 FBE 且云端返回 **Target Locked** 且有用目标信息 → 立刻退出 FBE，清空路径，本帧起重新按目标规划。
- **FBE 到达探索点**：若在 FBE 且已到达探索点（距终点 < 0.3m）→ 退出 FBE，清路径，做一小段环视。
- **Searching 过久触发 FBE**：在“规划逻辑”之后、执行之前判断；若 `status == "Searching"` 且 `searching_frames >= FBE_SEARCHING_FRAMES` 且当前不在 FBE → 尝试取探索点并进入 FBE。

### 4. 成功判定（多目标）

- 仅当 **Target Locked** 且 **latest_instruction_used == 当前目标名** 且 (u,v) 处深度 < `SUCCESS_DEPTH_THRESHOLD` 时，认为**当前目标**达成。
- 达成后：当前目标下标 +1，清空路径与目标 3D，清空 SharedState 中本次 Locked 相关状态，避免重复计数；若已是最后一目标则结束。

### 5. 短期记忆防死锁

- 仅当**已有 A* 目标**（`current_goal_3d`）且**已记录过时间与距离**时检查。
- 若距**上次记录**已超过 `MEMORY_DISTANCE_CHECK_INTERVAL`（5s）：
  - 若相对上次记录**距离缩短 < MEMORY_DISTANCE_IMPROVE_THRESHOLD**（0.15m）→ 认为死锁：当前目标进黑名单，清路径与目标，清 SharedState 中目标相关状态。
  - 否则：更新“上次记录时间”和“上次记录距离”，作为下一轮 5s 的基线。

### 6. 规划逻辑

- **进入条件**：非 FBE、`status == "Target Locked"`、当前无路径、且有有效 uv/snap_d/snap_c。
- 由 (u,v) + 深度 + 相机位姿得到目标 3D；若在黑名单则只清路径不规划。
- 若需 replan（无当前目标或目标位移 > `REPLAN_DISTANCE_THRESHOLD`）：对当前位与目标 3D 做 A*，得到 `path_points`，并记录 `goal_3d_recorded_at`、`goal_3d_recorded_dist`（供短期记忆用）。
- **Inferred / Searching**：若非 FBE，清空路径与当前目标 3D；若是 Searching，`searching_frames += 1`（为后面“Searching 过久触发 FBE”提供计数）。

### 7. Searching 过久触发 FBE

- 条件：`status == "Searching"`、当前未在 FBE、`searching_frames >= FBE_SEARCHING_FRAMES`。
- 调用 `get_explore_waypoint` 取探索点；若有路径则写入 `path_points`，置 `in_fbe_mode = True`，`searching_frames = 0`。

### 8. 执行控制

- **有路径**：沿路径逐步移动（取下一路径点，按 `MOVE_STEP_SIZE` 前进或 pop 已到达点），不判 front_clear。
- **无路径**：
  - **Inferred**：按 uv 相对画面中心的偏移做平滑与死区判断，决定左转/右转/前进或保守右转。
  - **Searching**：仅右转（或保持当前转向承诺）。

### 9. 可视化与步进

- 更新轨迹、路径、当前/目标在 2D 图上的显示；更新第一视角窗口与 status 文案（含是否在 FBE）。
- `step_count += 1`，`time.sleep(0.02)`，进入下一帧。

---

## 三、FBE（前沿探索）详细说明

### 3.1 为什么要 FBE

- 当**长时间拿不到目标**（云端一直返回 Searching）或**撞墙**时，若只做原地/盲目旋转，在长走廊等场景容易转一圈仍看不到目标，形成**死锁**。
- FBE 利用 Habitat 的 pathfinder，在当前位置附近取一个** 2–3 米外的随机可行走点**作为“探索点”，**先走到该点再环视**，相当于主动换视角，属于**主动探索（Active Exploration）**，避免盲目旋转。

### 3.2 探索点如何得到（get_explore_waypoint）

- 使用 **pathfinder**：
  1. 将当前位置 `curr_pos` **snap** 到 navmesh，得到 `start_snap`，并取当前 **island_index**（同层可行走区域）。
  2. 在**同一 island** 内，用 `get_random_navigable_point_near(start_snap, FBE_EXPLORE_RADIUS, island_index=...)` 在半径 **2.5m** 内随机采样一点。
  3. 若该点与当前位置距离 **≥ FBE_MIN_DISTANCE（1.2m）**，再对该点与 `start_snap` 做一次 **find_path**；若存在且路径点数 > 1，则将该点作为探索点返回。
  4. 最多尝试 10 次；失败则返回 None（本帧不进入 FBE，撞墙分支会退化为原地转 4 次）。

因此：**FBE 的探索点一定是与当前在同一可行走岛、距离约 1.2–2.5m、且能 A* 到达的点**，保证“走过去”在几何上可行。

### 3.3 FBE 的两种触发条件

| 触发方式 | 条件 | 行为 |
|----------|------|------|
| **① 撞墙** | `stuck_count >= STUCK_FRAMES`（前方过近连续多帧） | 清路径与当前目标、清 SharedState 目标相关；调用 `get_explore_waypoint`。若得到探索点则 A* 到该点，置 `in_fbe_mode = True`；否则原地转 4 次。 |
| **② Searching 过久** | `status == "Searching"` 且 `searching_frames >= FBE_SEARCHING_FRAMES`（约 5s 内全是 Searching） | 调用 `get_explore_waypoint`；若有路径则 `path_points = ...`，`in_fbe_mode = True`，`searching_frames = 0`。 |

- **①** 解决“贴墙/卡角”后的行为，用“去探索点”替代单纯转圈。
- **②** 解决“转了很久仍 Searching”的情况，主动换位置再找目标。
- `FBE_SEARCHING_FRAMES` 设得较大（如 350 帧 ≈ 7s），避免 FBE 过于频繁、打断正常环视。

### 3.4 FBE 模式下的行为

- **路径**：`path_points` 存的是“到探索点”的 A* 路径；执行阶段**有 path 就沿 path 走**，与是否 FBE 无关，因此会**朝探索点移动**。
- **每帧还会做**：
  - **中断**：若云端返回 **Target Locked** 且有用目标信息 → 立刻 `in_fbe_mode = False`，`path_points = []`，下一帧会按“Target Locked + 无路径”走规划逻辑，改为朝**目标**走。
  - **到达探索点**：若到探索点终点距离 < 0.3m → 退出 FBE，清路径，做 8 次右转（短环视），然后继续主循环（可能马上又 Searching，或得到 Inferred/Locked）。

逻辑上：**FBE 要么被“目标锁定”打断，要么走到探索点后主动结束并环视**，不会一直挂着 FBE 不放。

### 3.5 FBE 与规划/成功的关系

- **规划**：条件里有 `not in_fbe_mode`，因此**在 FBE 期间不会因为 Target Locked 而去规划到真实目标**；只有**先中断 FBE**（见上），下一帧才会以 Target Locked 做 A* 到目标。
- **成功**：成功判定与 `in_fbe_mode` 无关，只要 status 是 Target Locked、指令对应、深度够近就计成功。
- **短期记忆**：在 FBE 期间若之前有 `current_goal_3d`，仍会按 5s 检查；若死锁会清目标与路径（路径本就是去探索点的，清掉后本帧执行会变成无路径分支，可能继续转或下一轮再触发 FBE）。

### 3.6 FBE 如何做到上楼梯探索

FBE **没有单独写“上楼梯”的逻辑**，能自然走到楼上，是因为**完全沿用 Habitat 的 navmesh 与 A\***，而 navmesh 里通常**已经包含楼梯**，把多层连成同一套可行走面。

- **Navmesh 与楼梯**  
  - 在 Matterport3D / Habitat 里，场景的 navmesh 一般会把**楼梯、斜坡**都烘焙成可行走面，和地面、楼板连在一起。  
  - 因此“一楼”和“二楼”在 pathfinder 里往往是**同一连通区域**（同一个 island），或通过楼梯所在的三角形连通。

- **探索点采样**  
  - `get_random_navigable_point_near(start_snap, FBE_EXPLORE_RADIUS, island_index=island_idx)` 是在**当前所在 island** 上、在给定半径内随机采一个可行走点。  
  - 若楼梯把上下层连在**同一 island**，那“半径 2.5m 内”的可行走点就**可以落在楼上**（沿楼梯走的 geodesic 距离在 2.5m 内的楼上某点）。  
  - 若某场景里楼上楼下被做成**不同 island**（不连通），则当前实现只会采当前 island，不会采到楼上，也就不会主动上楼梯。

- **路径与执行**  
  - 一旦采到的探索点在楼上，`find_path` 会算出一条从当前位到该点的**最短可行走路径**，这条路径会**自动经过楼梯**（因为楼梯已在 navmesh 上）。  
  - 执行阶段只是**沿 path_points 逐步移动**，没有“平地/楼梯”的区分，所以会自然**沿 A\* 路径走上/下楼梯**。

- **小结**  
  - **能上楼梯**：因为 FBE 的探索点来自 pathfinder、路径也来自 A\*，而 navmesh 包含楼梯且上下层连通，所以探索点可能落在楼上，路径自然包含楼梯段。  
  - **若某场景不上楼梯**：多半是 navmesh 里楼梯未连通、或上下层是不同 island，FBE 不会跨 island 采样，也就不会选到楼上的点。

---

## 四、短期记忆防死锁（简述）

- **目的**：避免对“卡住点”（如桌角后的目标）反复 A*、反复尝试同一不可达目标。
- **做法**：每次对**当前目标 3D** 做 A* 时记录 `goal_3d_recorded_at` 和 `goal_3d_recorded_dist`（当前到目标距离）。每隔 **5s** 再量一次距离；若**缩短量 < 0.15m** 则认为 5s 内几乎没靠近 → 将该目标 3D 加入**黑名单**，清路径与当前目标，并清 SharedState 中目标相关状态。
- **后续**：规划时若算出的目标 3D 落在某黑名单点附近（距离 < MEMORY_BLACKLIST_TOLERANCE），则**不规划**该目标，只清 path，迫使系统依赖 Inferred 或 FBE 换视角再试。

---

## 五、多目标序列（简述）

- **指令解析**：如 "first find a door, then find a plant" → `target_list = ["door", "plant"]`，`current_target_idx` 从 0 开始。
- **云端**：始终用 `target_list[current_target_idx]` 作为当前指令发 /plan；多目标时还会传 `target_list`、`target_index`（若云端支持）。
- **成功**：仅当 **latest_instruction_used == 当前目标名** 且 Target Locked 且深度足够时，当前目标达成；然后 `current_target_idx += 1`，更新 `current_instruction`，并**清空 SharedState 中本次 Locked**，避免同一帧/下一帧误判为下一目标也达成。
- **全部完成**：当 `current_target_idx >= len(target_list)` 时结束主循环。

---

## 六、执行控制小结

| 情况 | 行为 |
|------|------|
| 有 path_points | 沿路径向当前路径目标移动（目标锁定时为真实目标，FBE 时为探索点）；不判 front_clear。 |
| 无路径 + Inferred + 有 uv | 按 uv 偏移平滑 + 死区：偏右则右转，偏左则左转，居中且前方畅则前进，否则保守右转；有转向承诺则保持上一转向。 |
| 无路径 + Searching | 每 TURN_INTERVAL 帧右转一次（或保持承诺），不前进。 |

---

## 七、关键参数速查

| 参数 | 典型值 | 含义 |
|------|--------|------|
| FBE_EXPLORE_RADIUS | 2.5 | 探索点采样半径 (m) |
| FBE_MIN_DISTANCE | 1.2 | 探索点与当前位最小距离 (m) |
| FBE_SEARCHING_FRAMES | 350 | Searching 连续多少帧后触发 FBE（约 7s） |
| STUCK_FRAMES | 4 | 前方多近、连续几帧视为撞墙并触发撞墙处理（含 FBE 尝试） |
| MEMORY_DISTANCE_CHECK_INTERVAL | 5.0 | 短期记忆检查间隔 (s) |
| MEMORY_DISTANCE_IMPROVE_THRESHOLD | 0.15 | 5s 内至少缩短距离 (m)，否则判死锁 |
| SUCCESS_DEPTH_THRESHOLD | 0.6 | 目标 (u,v) 处深度小于此认为到达 (m) |

---

## 八、单帧流程简图（便于对照代码）

```
取图 → 写 SharedState → 读 status/uv/snap_d/snap_c
  ↓
撞墙? ──是──→ 清路径/目标/SharedState 目标 → 取探索点 → 有? 设 path + in_fbe_mode
  ↓ 否
FBE 且 Target Locked? ──是──→ 退出 FBE，清 path
  ↓ 否
FBE 且 到探索点? ──是──→ 退出 FBE，清 path，环视 8 次
  ↓ 否
成功判定（Locked + 指令对应 + 深度够）→ 是则下一目标或结束
  ↓
短期记忆：有目标且满 5s → 距离几乎没缩短则黑名单 + 清目标/路径
  ↓
规划：非 FBE 且 Locked 且无 path 且有效 uv → A* 到目标，记时间/距离
  Inferred/Searching → 清 path/目标（非 FBE）；Searching 则 searching_frames++
  ↓
Searching 过久? ──是──→ 取探索点 → 有 path 则 in_fbe_mode=True, path_points=...
  ↓
执行：有 path 则沿 path 走；否则 Inferred 按 uv 转/前进，Searching 右转
  ↓
可视化 → step_count++, sleep(0.02) → 下一帧
```

以上即 v2 的完整流程，FBE 与规划、成功、执行之间的先后与互斥关系均按实际代码逻辑整理，便于对照阅读和修改。
