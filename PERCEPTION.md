# VLN 感知模块

本分支根目录的 Python 包提供目标解析、单视角场景感知和候选验证。三个接口共用一个 OpenAI 兼容的本地 Qwen 服务。实验使用的模型 ID 为 `qwen3.8:27b`。

## 安装与调用

在分支根目录运行 `python -m pip install -e .`。将下面的服务地址和图片路径替换为实际值。

```python
from vln_perception import VLMClient

client = VLMClient("http://127.0.0.1:18281/v1", "qwen3.8:27b")
rgb = "scene.png"
goal = client.parse_goal("找桌子")
scene = client.perceive_scene(
    rgb, goal, task_id="find_table", observation_id="obs_001", view_id="front"
)
for item in scene["target_entities"]:
    verdict = client.verify_target(rgb, goal, scene, item["local_id"])
    print(item["local_id"], verdict["status"])
```

| 接口 | 结果与约定 |
|---|---|
| `parse_goal()` | 提取目标类别、属性、空间关系和完成条件，保留原始指令。 |
| `perceive_scene()` | 输出 `target_status`、目标实体、主要环境实体和空间线索。可见框为 0–1000 整数坐标 xyxy，实体 ID 在当前观测内有效。 |
| `verify_target()` | 输入原图、GoalSpec、原感知结果和候选 ID，返回 VERIFIED、REJECTED 或 UNCERTAIN 及双方视觉证据。换视角时需要先在新图上定位同一实例。 |

## 评测与结果

四个评测脚本位于 `evals/`，均支持 `--help`。下面的文本评测可直接在已启动模型服务的环境中运行，输出路径使用新文件名。

```bash
python evals/eval_goal.py --model qwen3.8:27b --base-url http://127.0.0.1:18281/v1 --output goal-repeat.jsonl
```

同图换指令和候选验证脚本复用实验视角。复现时从已有数据中准备 MP3D 场景 `2azQ1b91cZZ` 的同一张餐厅帧，路径为 `results/mp3d-2azQ1b91cZZ-chair-challenges-v1/multiple.png`。验证脚本核对 RGB 的 SHA-256。早期双视角脚本使用 `evals/scene_pair/A.jpg` 和 `B.jpg` 两张 Replica 帧。上面的接口示例可以使用自己的图片。

- [中英文目标解析结果](results/goal-27b-gpu3.jsonl)
- [同一餐厅找椅子、桌子和冰箱的三组完整输出](results/mp3d-three-goals-dining/dining-three-goals.json)
- [对原始两个桌子候选的验证](results/verify-table-step1/verify-table-pair.json)
- [同一餐边柜换视角的验证对照](results/verify-e02-new-view/e02-view-comparison.json)

MP3D 餐厅冰箱负例返回了 NOT_VISIBLE 和空目标列表。真实餐桌获确认。餐边柜在第二个视角露出抽屉和柜门后仍被误确认为桌子。以上记录是这些具体样例的实际结果。
