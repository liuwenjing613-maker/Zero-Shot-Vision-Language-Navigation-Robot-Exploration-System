# VLN Simple 版导航 - 逻辑流程

> 按诊断文档重构：状态机 + 双模态融合

## 一、架构

```
本地 (local_robot_simple.py)          云端 (cloud_brain_server.py)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ Habitat + 状态机 + 50FPS     │ ──→  │ Qwen2.5-VL: 目标定位 (无verify)│
│ Locked→A*  Inferred→仅转向  │ ←──  │ PIXEL / INFER / SEARCHING    │
└─────────────────────────────┘      └─────────────────────────────┘
```

## 二、核心策略

| 策略 | 内容 |
|-----|------|
| 1. 双模态成功判定 | Target Locked + 当前帧 (u,v) 处深度 < 0.8m → 成功，废弃 Verify |
| 2. 分离 Locked/Inferred | Locked 才 A*；Inferred/Searching 绝不 A*，仅按 u 偏移转向 |
| 3. 50FPS 丝滑 | 消除 sleep，每帧小步 turn/move |
| 4. 转向+前进 | 先转向对准目标再 move_forward，避免螃蟹式平移 |

## 三、主循环逻辑

```
每帧 (~0.02s):
  1. 获取 rgb/depth，推 shared_state
  2. 撞墙: 前方深度 < 0.6m 连续 6 帧 → 清目标、转 4 次（无 sleep）
  3. 成功: status==Target Locked 且 当前 depth(u,v)<0.8m → 成功
  4. 规划: 仅当 Target Locked 且无路径 → A*；Inferred/Searching 清空 path
  5. 执行: 有 path→转向对准+move_forward；无 path→按 u 偏移 turn_left/right
  6. 可视化
```

## 四、配置参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| SUCCESS_DEPTH_THRESHOLD | 0.8m | 双模态成功：深度小于此即成功 |
| TURN_ANGLE_THRESHOLD | 0.15 rad | 朝向偏差小于此才前进 |
| CLOUD_SEND_INTERVAL | 1.2s | 云端请求间隔 |
| STUCK_DEPTH_THRESHOLD | 0.6m | 撞墙深度 |

## 五、启动方式

```bash
# 云端 (RTX 3090)
python cloud_brain_server.py

# 本地
python local_robot_simple.py
```
