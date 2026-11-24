# 完整联调

分支 `full` 继承 `hyf` 中已合入的 `tls` 和 `lwj` 实现。入口为 `evals/eval_tls_hyf_episode.py`。每个回合依次运行语言解析、RGB-D 感知、占据与对象记忆、前沿排序、高层决策、导航执行和到达复核。执行器通过 navmesh 移动，在每次平移前用已配置的转向动作将机身与相机对准即将移动的水平位移。

在安装 Habitat-Sim 0.3.3、NumPy、SciPy、Pillow、numpy-quaternion 的环境运行。感知配置见 `evals/live/model.py` 和 `evals/live/README.md`。本评测需要 MP3D 场景 2azQ1b91cZZ 及其同名 navmesh，数据目录内应包含 mp3d.scene_dataset_config.json。输出目录使用新的路径。

```bash
python evals/eval_tls_hyf_episode.py /tmp/vln-full-wood-chair \
  --dataset /path/to/mp3d/scenes --gpu 0 --frames 18 --exploration-budget 7 \
  --start-position 9.8350982666 0.1271100044 3.703299284 \
  --instruction 'Find a wooden chair beside a dining table. Approach that chair and stop once you can verify it beside the table.'
```

查看 `trace.json` 中的 terminal 判断任务结果；各观测目录保存 RGB、传感器、感知和记忆。到达成功需执行器接近成功、到达后在新图像上验证相同全局 ID、水平估计距离在 1 米内并停机。材质和关系来自视觉证据，需要和独立实例、空间标注区分。该后端在导航网格上更新位姿，适用于仿真评测。

## 生成离线回放

将回合目录放在同一个目录下。下面的命令会按原始扫描重建网格和前沿，用 Habitat 后端重放中间位姿，并校验原始观测图、导航终点和行进时相机偏角。打开输出目录里的 index.html 即可逐步检查。使用本地获授权的 MP3D 数据生成回放。

```bash
python evals/full_replay/build.py --runs-root /tmp/vln-runs --dataset /path/to/mp3d/scenes --gpu 0 --output /tmp/vln-export
```

可单独对视觉验证框做 MP3D 实例核查，再将结果加入回放。审计使用同位姿重渲染。相邻桌子的统计基于同视野、垂直包围盒相交或间距不超过 0.15 米的标注桌子，给出水平包围盒间隙与中心距。物体材质和桌子的用途仍需视觉证据。

```bash
python evals/full_replay/audit_instance.py /tmp/vln-runs/wood-chair --dataset /path/to/mp3d/scenes --gpu 0 > /tmp/vln-runs/audit-chair.json
```

给导出命令增加 `--manifest /path/to/episodes.json` 可指定顺序和审计文件。例如：

```json
[{"id":"wood-chair","label":"复杂任务","goal":"寻找餐桌旁的木椅","audit":"audit-chair.json"}]
```

回放从首帧传感器矩阵恢复初始朝向，并按回合配置重放移动转向和到达策略。对于历史回合可在清单指定 planner_at_arrival。

## 跨帧对象记忆是否改变导航动作

在场景 2azQ1b91cZZ 中，先用“寻找餐桌旁的木椅”获得一次通过到达复核的源回合。该次首次搜索在第 4 帧从探索转入对象验证，验证通过后接近 O000009，第 5 帧以新图复核并在 0.44 米处成功停机。另一起点到达后对材质存疑，以 ARRIVAL_UNVERIFIED 结束。

回访测试从成功回合恢复 M1 地图、M2 对象与验证记录、M3 前沿状态。测试指定一个中间任务，要求机器人沿导航网格回到入口附近，再转到视觉模型未检出目标的朝向。两组共享相同位姿、RGB、深度、网格、前沿和任务。完整组保留跨帧对象，当前帧组用同一张图重新建立 M2。两组分别执行规划器选出的首个动作。额外的可达性对照只重新计算决策。

| 输入 | 实际选择 | 执行与证据 |
| --- | --- | --- |
| 跨帧 M2 + 可达性 + 40 个前沿 | APPROACH_TARGET，O000009 | 引用 obs_0004/E02、obs_0005/E01，规划路径 12.26 米，执行成功 |
| 当前帧 M2 + 同一可达性检查与前沿 | EXPLORE_FRONTIER，F0011 | 规划路径 5.34 米，执行成功 |
| 保留跨帧 M2，只撤去该对象的可达标记 | OBSERVE_TARGET，O000009 | 决策对照，未执行 |

回访出发时，场景语义图中椅子 #161 占 0 像素。完整组返回后，第一个视角的 E01 框中心再次命中 #161，但 M2 关联到 O000013。四个回访视角均未把候选关联回任务目标 O000009，因此没有触发对同一对象的新验证或成功停机。场景实例只在评估中核对。该单例验证了历史证据和可达性对导航动作的影响，不能估算总体成功率。M3 为两组提供前沿，M4 在本例的独立作用尚未做消融。

运行回访实验并将报告嵌入离线回放。输出目录需要是新路径，以保存每次独立运行的输入与输出。

```bash
python evals/eval_memory_recall.py --source /tmp/vln-runs/wood-chair --dataset /path/to/mp3d/scenes --gpu 0 --output /tmp/recall-trial
python evals/full_replay/build.py --runs-root /tmp/vln-runs --dataset /path/to/mp3d/scenes --gpu 0 --output /tmp/vln-export --memory-trial /tmp/recall-trial/report.json
```

报告中包含两个分支的实际移动轨迹、新视角、候选的场景实例与系统 ID。给回访脚本传入 `--replay-perception /tmp/recall-trial` 可使用同一批视觉输出重新核查几何与决策，脚本会逐像素校验输入。回放页启用 `?view=memory` 会直接展开对照。
