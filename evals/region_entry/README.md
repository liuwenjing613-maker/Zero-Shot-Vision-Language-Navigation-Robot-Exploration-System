# MP3D 进入新区域实测

服务器上的 M1 记忆模块通过 `RegionEntryDetector.start` 建立起点，`perceive` 接收已有视觉感知的 `visual_cues`、同帧公制深度、相机内外参和当前占据网格，`advance` 接收实际执行动作后的位姿和新观测编号。每个场景使用一个检测器实例。检测器返回区域、门口和拓扑节点编号，并更新传入的 `TopologyMemory`。

在 Habitat 中，机器人先向门前进四步再退回，然后前进十步穿门、转向后前进十步返回。移动步长为 0.25 米。每个位置采集 RGB-D 和 720 束完整水平扫描。测试复用起点 RGB 的真实 Qwen 结果，核对像素和文件哈希，在门前折返时用新增的扫描和实际位姿复核候选。区域编号、人工边界和正交俯视图只用于选点及事后评分。每个方向要求恰好触发一次，触发位姿位于目标区域内且距人工边界至少 0.10 米，回程复用同一门口并恢复原区域。初段靠近并退回要求零触发。

在项目根目录运行以下命令。路径可替换为独立实验目录。场景选择脚本和俯视图脚本使用服务器的 GPU 3。项目感知服务使用本机 Ollama 的 qwen3-vl:8b，保留项目感知提示词、字段和校验器。

```bash
export PYTHONPATH="$PWD/src" HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
V=/data/chenkejun/vln/envs/vln-sim/bin/python
W=/data/chenkejun/vln/region-entry-work
O=/data/chenkejun/vln/region-entry-replay
"$V" -m unittest discover -s tests -v
"$V" evals/region_entry/fixture.py "$W"
"$V" evals/region_entry/perceive.py "$W" 1 6 2 5 15 9 17
"$V" evals/region_entry/walk.py "$W" "$O" 1 6 2 5 15 9 17
"$V" evals/region_entry/overview.py "$W" "$O" 1 6 2 5 15 9 17
"$V" evals/region_entry/report.py "$O" 1 6 2 5 15 9 17
```

已有的 `$W/qwen-*.json` 和 `view-*.png` 可直接用于复现冻结感知输入下的检测结果。每条路线的编号来自 `fixtures.json`；路线 17 是路线 15 的反向观察位置，冻结规则后单独检验。回放提供 RGB、同步网格和纹理俯视图，下载按钮提供当前路线的完整感知候选、每步位姿、区域标注、建点事件和拓扑边。
