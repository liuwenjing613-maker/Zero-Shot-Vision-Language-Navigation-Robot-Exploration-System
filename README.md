# InternVLA - Vision-Language Navigation System

基于 InternVLA 的视觉语言导航（VLN）系统实现，包含多个版本的对比实验。

## 项目结构

```
InternVLA/
├── code/                    # 核心代码
│   ├── v1_simple/           # v1 版本：基础导航
│   ├── v2_FBE/              # v2 版本：增加 FBE 探索
│   └── v3_lookaround/       # v3 版本：出生环视 + 多目标
├── paperS2+S1/              # System2 + System1 实现
│   ├── cloud_s2_server.py   # 云端推理服务器
│   ├── local_robot_s2.py    # 本地机器人控制
│   └── README.md            # 详细说明
├── val/                     # 评估相关
│   ├── evaluate_system.py   # 自动化评估脚本
│   ├── eval_episodes.json   # 测试集
│   └── run_evaluation.sh    # 评估运行脚本
└── ppt/                     # 文档和分析
```

## 系统架构

采用云端-本地分离架构：
- **云端**: 运行 VLM 模型进行视觉语言推理
- **本地**: Habitat-sim 仿真环境 + 机器人控制

## 版本对比

| 版本 | 特性 | 说明 |
|------|------|------|
| v1 | 基础导航 | 简单的目标跟踪 |
| v2 | FBE 探索 | 增加前沿探索机制 |
| v3 | 环视+多目标 | 出生环视、多目标追踪、目标验证 |

## 依赖

### 云端（GPU 服务器）
- PyTorch
- Transformers
- Flask

### 本地
- Habitat-sim
- OpenCV
- NumPy

## 场景数据

场景文件（.glb）较大，未包含在仓库中。请从以下来源下载：
- [Matterport3D](https://niessner.github.io/Matterport/)
- [Habitat-Matterport 3D Research Dataset](https://aihabitat.org/datasets/hm3d/)

放置到 `scenes/` 目录下。

## 评估

```bash
cd val
./run_evaluation.sh --version v3
```

评估结果保存在：
- `val/eval_report.txt` - 汇总报告
- `val/detailed_results/` - 详细 JSON 结果

## 许可证

MIT License
