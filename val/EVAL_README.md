# VLN 评估说明（v2 / v3 指标与对比）

本目录用于在 **eval_episodes.json**（40 条数据）上统一评估 **v2** 与 **v3**，采用**严格成功判定**与领域常用指标。

---

## 一、测试集与成功判定

### 1.1 测试集 `eval_episodes.json`

- 每条 episode 字段：
  - **episode_id**: 序号
  - **scene_id**: 场景 ID（与场景路径中的文件夹名一致）
  - **instruction**: 单目标指令（如 `"fireplace"`、`"plant"`）
  - **start_position**: 机器人起点 `[x, y, z]`
  - **gt_position**: 目标真实 3D 坐标（Ground Truth）`[x, y, z]`

- 评估时：机器人从 `start_position` 出发，根据 `instruction` 导航；**成功与否看过程内任意时刻与 `gt_position` 的距离是否 < 阈值**（默认 1 m），与界面内“目标达成”逻辑解耦。

### 1.2 成功判定（过程内任意时刻 < 阈值即成功）

- **规则**：**取消“仅看终点”**；只要在**导航过程中任意时刻** agent 与 gt_position 的欧氏距离 < 阈值，该 episode 即记为成功，并**在该时刻停止该 episode**（用于 SPL 的路径长度也截止于此）。
- **默认阈值**：**1.0 m**（VLN / embodied 常用）。
- **更严**：`--success_dist 0.5` 要求更近才算成功。

公式：

- `success = (存在某时刻 t：|| pos(t) - gt_position || < 成功阈值)`；首次达到即判成功并停止该条。

---

## 二、多次运行与 mean±std

- 默认 **同一配置跑 5 轮**（`--runs 5`），对 SR、SPL、NE 汇报 **mean ± std**，用于判断 v2/v3 差异是否稳定。
- 示例：`python evaluate_system.py --version both --runs 5`
- 报表中格式：`SR = 0.15 ± 0.03` 等。

## 三、指标说明

| 指标 | 含义 |
|------|------|
| **SR (Success Rate)** | 成功率 = 成功 episode 数 / 总 episode 数。 |
| **SPL (Success weighted by Path Length)** | 成功时 SPL = (最短路径长度 / 实际路径长度)；失败时 0。均值越接近 1 表示路径越短且成功越多。 |
| **NE (Navigation Error)** | 终点到 gt_position 的平均距离（米），越小越好。 |

- **最短路径长度**：起点到 gt 的 geodesic 距离（pathfinder 规划）。
- **实际路径长度**：轨迹点逐段累加的欧氏距离。

---

## 四、测试流程

### 4.1 前置条件

1. **云端已启动**：在带 GPU 的机器上运行  
   `python code/v2_FBE/cloud_brain_server_v2.py`  
   （v2 / v3 评估共用同一云端，本地只跑评估脚本。）

2. **本地环境**：已安装 habitat-sim，能导入并运行 `val/evaluate_system.py`。

3. **测试集**：`val/eval_episodes.json` 已就绪（当前约 40 条），且其中 `scene_id` 对应的场景在 `SCENES_BASE` 下存在。

### 4.2 运行评估（无头，不弹窗）

在项目根或 `val` 目录下执行：

```bash
cd /home/abc/InternVLA/val
python evaluate_system.py
```

- **推荐**：`--version both --runs 5`，先跑 5 轮 v2、再 5 轮 v3，最后打印 SR/SPL/NE 的 mean±std 及 v2/v3 对比。
- **默认**：`--version both`，`--runs` 默认为 5；仅跑单轮可用 `--runs 1`。

仅跑某一版本：

```bash
python evaluate_system.py --version v2
python evaluate_system.py --version v3
```

使用更严成功距离（0.5 m）：

```bash
python evaluate_system.py --version both --success_dist 0.5
```

### 4.3 输出与报表

- **终端**：每个 episode 一行（是否成功、SPL、NE）；多轮跑完后输出各版本 SR/SPL/NE 的 **mean±std** 及 v3−v2 对比。
- **报表文件**：`val/eval_report.txt`，内容与终端汇总一致（含 mean±std），便于留存和对比。

---

## 五、v2 / v3 在评估脚本中的差异

| 项目 | v2 | v3 |
|------|----|----|
| 出生地 | 直接开始导航 | 先 4 视野 scan 置信度，转向最高置信度方向后再开始导航 |
| FBE | 撞墙 / Searching 过久可触发 | 仅当**未**完全确认目标时触发；完全确认后不触发、不随便转向 |
| 完全确认 | 无 | 规划到目标后请求 verify，通过则 `target_fully_verified=True` |
| 成功判定 | 与 v3 相同 | 与 v2 相同：**过程内任意时刻与 gt 距离 < 阈值即成功** |

评估脚本中 v2 / v3 共用同一套 **成功判定**（过程内任意时刻与 gt 距离 < 阈值即成功），与交互版 v3 内部的“目标达成”逻辑独立。

---

## 六、结果对比示例

运行 `python evaluate_system.py --version both --runs 5` 后，报表类似：

```
======================================================================
评估报表 (多轮 mean ± std)
======================================================================
成功判定: 导航过程中任意时刻与 gt 距离 < 1.0m 即成功
Episodes: 41  |  重复轮数: 5
  [v2] SR = 0.xxxx ± 0.xxxx   SPL = 0.xxxx ± 0.xxxx   NE = x.xxx ± x.xxx m
  [v3] SR = 0.xxxx ± 0.xxxx   SPL = 0.xxxx ± 0.xxxx   NE = x.xxx ± x.xxx m

  对比 (v3 − v2) 均值:
    SR:  +x.xxxx
    SPL: +x.xxxx
    NE:  +x.xxxx m (负=更好)
======================================================================
```

- **SR/SPL 提升、NE 下降**：表示 v3 在该测试集上优于 v2。
- 同一套 `eval_episodes.json` 和同一成功阈值下，可直接对比 v2 与 v3 的指标。

---

## 七、与本地交互运行的对应关系

- **仅做指标测试**：云端已开 → 直接运行 `python val/evaluate_system.py`（无头），无需再开本地 v2/v3 窗口。
- **想边看边跑**：可单独运行 `python code/v2_FBE/local_robot_simple_v2.py` 或 `python code/v3_lookaround/local_robot_simple_v3.py`，但评估指标以 `evaluate_system.py` 在 `eval_episodes.json` 上的统计为准（成功=终点与 gt 距离 < 阈值）。

---

## 八、文件一览

| 文件 | 说明 |
|------|------|
| `eval_episodes.json` | 测试集（约 40 条，含 scene_id / instruction / start_position / gt_position） |
| `evaluate_system.py` | 无头评估脚本，支持 `--version v2|v3|both`、`--success_dist` |
| `eval_report.txt` | 运行后生成的报表（SR/SPL/NE 及 v2/v3 对比） |
| `EVAL_README.md` | 本说明（测试流程与结果对比） |
