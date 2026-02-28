# VLN v2 导航 - 改进说明

> 在 v1 基础上加入：FBE 前沿探索、短期记忆防死锁、多目标序列导航

## 一、文件对应

| 角色 | v1 (原版) | v2 (改进版) |
|-----|-----------|-------------|
| 云端 | cloud_brain_server.py | cloud_brain_server_v2.py |
| 本地 | local_robot_simple.py | local_robot_simple_v2.py |

## 二、v2 新增功能

### 1. FBE 前沿探索 (Frontier-Based Exploration)

- **问题**：找不到目标时盲目 `turn_right`，长走廊转一圈易死锁
- **方案**：调用 `sim.pathfinder.get_random_navigable_point_near()` 获取距当前位置 2–3m 的随机可达点作为探索点
- **触发**：
  - 撞墙时：替代盲目旋转，先移动到探索点再环视
  - Searching 持续约 1.6s：主动前往探索点换视角

### 2. 短期记忆防死锁 (Short-term Memory)

- **问题**：卡在桌角等障碍旁，反复尝试同一目标
- **方案**：规划 A* 时记录目标 3D 坐标；5 秒内若距离未缩短 ≥0.15m，判定死锁
- **动作**：将当前目标加入黑名单，放弃并重新探索

### 3. 多目标连续导航 (Sequential Navigation)

- **指令格式**：`"First find a chair, then find a plant, finally go to the door"`
- **解析**：自动提取 `["chair", "plant", "door"]`，按序导航
- **单目标**：`"plant"` 仍兼容

## 三、启动方式

```bash
# 云端 (RTX 3090) - 使用 v2
python cloud_brain_server_v2.py

# 本地 - 使用 v2
python local_robot_simple_v2.py
```

## 四、配置参数 (local_robot_simple_v2.py)

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| FBE_EXPLORE_RADIUS | 2.5 | 探索点采样半径 (m) |
| FBE_MIN_DISTANCE | 1.2 | 探索点最小距离 (m) |
| FBE_SEARCHING_FRAMES | 80 | Searching 多少帧后触发 FBE |
| MEMORY_DISTANCE_CHECK_INTERVAL | 5.0 | 死锁检测间隔 (s) |
| MEMORY_DISTANCE_IMPROVE_THRESHOLD | 0.15 | 5s 内至少缩短距离 (m) |

## 五、INSTRUCTION 示例

```python
# 单目标
INSTRUCTION = "plant"

# 多目标序列
INSTRUCTION = "First find a chair, then find a plant, finally go to the door"
```
