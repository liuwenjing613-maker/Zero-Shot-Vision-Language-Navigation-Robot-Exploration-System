# InternVLA S2 + 我的 S1 导航系统

视觉伺服闭环：用 InternVLA System2 作为“大脑”，输出**旋转指令**或**像素目标点**或 **STOP**；本地 S1 策略基本不变，只适配上述三种输出，完成“看哪个模型能力更强”的对比实验。

## 思路（与论文/附录一致）

- **正确做法**不是逼模型在单张图上猜像素，而是建立**视觉伺服闭环**：  
  `输出旋转指令 → 控制机器人旋转 → 获取新图 → 再次推理`。
- **InternVLA S2**（原论文附录 User Prompt）：
  - 未看到目标：输出**左右箭头**（← / →）或 "turn left" / "turn right"。
  - 看到目标：输出**下一 waypoint 的像素坐标**（如 `<point>x,y</point>`）。
  - 任务完成：输出 **STOP**。
- **我的 S1**：路径规划、FBE、撞墙处理、深度靠近判定、多目标序列等逻辑不变，只根据云端返回的 **action** 执行：
  - `turn_left` → 左转；
  - `turn_right` → 右转；
  - `waypoint` + (u,v) → 与现有一致：按 (u,v) 做路径/直行；
  - `stop` → 当前目标达成，进入下一目标或结束。

## 目录结构

```
internlaS2+myS1/
├── README.md
├── cloud_internvla_s2.py    # 云端：InternVLA S2 推理，解析 箭头/坐标/STOP
└── local_robot_internvla_s2_my_s1.py  # 本地：Habitat 机器人，S1 适配 S2 的 action
```

## 云端 (cloud_internvla_s2.py)

- **运行**：在带 GPU 的机器上  
  `python cloud_internvla_s2.py`  
  默认 `0.0.0.0:5000`。
- **模型**：与现有一致，使用 Qwen2.5-VL 架构；通过环境变量 `INTERNVLA_S2_MODEL` 指定路径。
- **Prompt**：原论文附录一字不差  
  `You are an autonomous navigation assistant. Your task is {instruction}. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task. These are your historical observations: {history}.`
- **无 system message**，以便模型在“看不到目标”时自然输出箭头而非被迫猜坐标。
- **解析**：
  - STOP → `action: "stop"`
  - ← / left / turn left → `action: "turn_left"`
  - → / right / turn right → `action: "turn_right"`
  - `<point>x,y</point>` 或 `(u,v)` → `action: "waypoint", u, v`
  - 否则 → `action: "searching"`
- **API**：`POST /plan`，表单 `image` + `instruction`，可选 `target_list`、`target_index`、`history`。  
  返回 JSON：`action`，以及 `u,v`（当 action 为 waypoint）、`reason` 等。

## 本地 (local_robot_internvla_s2_my_s1.py)

- **运行**：在已安装 Habitat-sim 的本机  
  `python local_robot_internvla_s2_my_s1.py`  
  需先启动云端服务；可通过环境变量 `INTERNVLA_S2_CLOUD` 指定云端 base URL（默认 `http://127.0.0.1:5000`）。
- **共享状态**：`latest_action`（turn_left / turn_right / waypoint / stop / searching）、`latest_goal_uv`（waypoint 时）、`task_stop_received`（当前目标收到 STOP）。
- **控制**：
  - `turn_left` / `turn_right`：只执行转向，形成“转 → 新图 → 再推理”的闭环。
  - `waypoint` + (u,v)：与现有一致，路径规划 + 死区转向 + 直行（同 `MOVE_STEP_SIZE` 步长）。
  - `stop`：当前目标达成，多目标时切下一目标或结束。
  - `searching`：无有效目标时每 N 帧转一次，避免卡死；Searching 过久仍触发 FBE。
- **S1 保留**：FBE、撞墙转向、短期记忆防死锁、深度靠近判定、多目标序列、俯视图可视化等均保留。

## 使用流程

1. 云端：  
   `cd /home/abc/InternVLA/code/internlaS2+myS1 && python cloud_internvla_s2.py`
2. 本地：  
   `cd /home/abc/InternVLA/code/internlaS2+myS1 && python local_robot_internvla_s2_my_s1.py`
3. 在本地界面按 `q` 退出。

## 与 v3 / cloud_brain_server_v2 的区别

| 项目       | v3 + cloud_brain_server_v2     | InternVLA S2 + my S1           |
|------------|--------------------------------|---------------------------------|
| 云端输出   | status + (u,v)，无显式“转向”   | action：turn_left / turn_right / waypoint / stop / searching |
| 未看到目标 | Inferred (u,v) 或 Searching   | 显式 turn_left / turn_right，闭环旋转再推理 |
| 任务完成   | 深度近 + verify                | S2 输出 STOP 即记为目标达成    |
| Prompt     | 自研 STATUS/PIXEL/REASON       | 原论文附录 User Prompt（含 history 占位） |

这样可以在同一套 S1 下对比“原 v2 云端”与“InternVLA S2 云端”的导航能力。
