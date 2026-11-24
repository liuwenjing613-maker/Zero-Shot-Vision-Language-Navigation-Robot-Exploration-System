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
