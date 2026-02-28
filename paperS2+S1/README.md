# InternVLA-N1 System2 + System1 导航系统

本目录包含基于 InternVLA-N1 论文的 System2 + System1 双系统导航实现。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    云端服务器 (GPU)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           InternVLA-N1-System2 模型                  │   │
│  │  ┌───────────────┐    ┌──────────────────────┐     │   │
│  │  │   System2     │    │     System1          │     │   │
│  │  │  (规划推理)    │───▶│   (动作生成)         │     │   │
│  │  │ 输出离散动作   │    │ 输出连续轨迹 waypoints│     │   │
│  │  └───────────────┘    └──────────────────────┘     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▲                                 │
│                           │ HTTP POST                       │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────┐
│                    本地机器 (Ubuntu)                         │
│  ┌────────────────────────┴────────────────────────────┐   │
│  │              local_robot_s2.py                       │   │
│  │  ┌───────────────┐    ┌──────────────────────┐     │   │
│  │  │ Habitat-sim   │    │   控制逻辑            │     │   │
│  │  │ 3D 仿真环境   │───▶│ 执行离散动作/FBE探索  │     │   │
│  │  └───────────────┘    └──────────────────────┘     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 文件说明

| 文件 | 位置 | 说明 |
|------|------|------|
| `cloud_s2_server_official.py` | 云端 | System2 推理服务 (推荐使用) |
| `cloud_s2_server.py` | 云端 | 简化版推理服务 |
| `local_robot_s2.py` | 本地 | 机器人控制脚本 |

## 离散动作定义

| Action ID | 动作 | 描述 |
|-----------|------|------|
| 0 | STOP | 停止，导航完成 |
| 1 | TURN_LEFT_LARGE | 左转 90° |
| 2 | TURN_LEFT_SMALL | 左转 15° |
| 3 | TURN_RIGHT_SMALL | 右转 15° |
| 4 | TURN_RIGHT_LARGE | 右转 90° |
| 5 | MOVE_FORWARD | 前进一步 |

## 安装依赖

### 云端服务器

```bash
# 1. 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装 transformers 和相关库
pip install transformers accelerate qwen-vl-utils flask pillow opencv-python numpy

# 3. (可选) 安装 InternNav 框架获取完整功能
git clone https://github.com/InternRobotics/InternNav
cd InternNav
pip install -e .
git submodule update --init

# 4. 下载模型
# 方法1: 使用 huggingface-cli
huggingface-cli download InternRobotics/InternVLA-N1-System2 --local-dir /root/autodl-tmp/.autodl/models/InternVLA-N1-System2

# 方法2: 使用 git clone
cd /root/autodl-tmp/.autodl/models
git clone https://huggingface.co/InternRobotics/InternVLA-N1-System2
cd InternVLA-N1-System2
git lfs pull
```

### 本地机器

```bash
# 确保已安装 habitat-sim
pip install habitat-sim
pip install opencv-python matplotlib numpy requests
```

## 使用方法

### 1. 启动云端服务

```bash
# 在云端服务器上
cd /root/autodl-tmp/.autodl/InternVLA/paperS2+S1
python cloud_s2_server_official.py
```

服务启动后会监听 `http://0.0.0.0:5000`

### 2. 配置 SSH 端口转发

```bash
# 在本地机器上
ssh -L 5000:localhost:5000 user@cloud-server
```

### 3. 启动本地导航

```bash
# 在本地机器上
cd /home/abc/InternVLA/paperS2+S1
python local_robot_s2.py
```

## API 接口

### GET /health
健康检查

### POST /reset
重置 agent 状态

### POST /step
执行一步推理

**请求参数:**
- `rgb` (file): RGB 图像
- `depth` (file, 可选): 深度图
- `instruction` (string): 导航指令
- `pose` (json, 可选): 位姿矩阵

**响应:**
```json
{
    "action": [5, 5],
    "stop": false,
    "pixel_goal": [240, 320],
    "trajectory": [[0, 0], [0.1, 0.05], ...],
    "step_count": 10
}
```

### POST /plan
兼容 v3 的推理接口

### POST /verify
验证是否到达目标

## 与 v3 的区别

| 特性 | v3 (你的版本) | Paper S2+S1 |
|------|--------------|-------------|
| 云端模型 | Qwen2.5-VL-7B | InternVLA-N1-System2 |
| 动作空间 | 连续 (像素坐标) | 离散 (6个动作) |
| 决策方式 | 目标点规划 + A* | System2 直接输出动作 |
| 轨迹输出 | 无 | System1 输出 waypoints |
| FBE 探索 | ✓ | ✓ |
| 多目标支持 | ✓ | ✓ |
| 出生地环视 | ✓ | ✗ (由 S2 自动处理) |

## 配置参数

在 `local_robot_s2.py` 中可以修改以下参数:

```python
# 场景设置
SCENE_INDEX = 0  # 场景索引
USE_FIXED_START = True  # 是否使用固定起点
FIXED_START_POSITION = [8.23, 0.13, 3.41]

# 导航指令
INSTRUCTION = "the first flowers you see"

# 目标位置 (用于可视化)
GT_TARGET_POSITION = [14.36, 0.13, 2.01]
```

## 注意事项

1. **显存要求**: InternVLA-N1-System2 约需 16-24GB 显存
2. **网络延迟**: 推理间隔默认 0.6 秒，可根据网络情况调整
3. **模型大小**: 模型约 8GB，需提前下载

## 常见问题

**Q: 模型加载失败？**
A: 检查模型路径是否正确，显存是否足够。可以尝试使用 `bfloat16` 或 `int8` 量化减少显存使用。

**Q: 连接超时？**
A: 确保 SSH 端口转发正确设置，云端服务已启动。

**Q: 导航效果不好？**
A: 调整 `INFER_INTERVAL` 提高推理频率，或检查场景 NavMesh 是否正确加载。
