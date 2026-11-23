# M1 几何与拓扑记忆

一个场景、一个楼层使用一组 `OccupancyMemory` 和 `TopologyMemory`。同一场景切换任务时可以保留这两个实例。核心包只依赖 NumPy，可用 `python -m pip install -e '.[memory]'` 安装。Habitat 适配器在调用时使用仿真环境自带的 quaternion。

## 帧输入

```python
from vln_memory.geometry_memory import Intrinsics, OccupancyMemory
from vln_memory.habitat_adapter import update_from_habitat
from vln_memory.topology_memory import TopologyMemory

calib = Intrinsics.from_hfov(width, height, hfov_degrees)
memory = OccupancyMemory((rows, cols), (origin_x, origin_z), resolution=0.1)
topology = TopologyMemory()
obs = sim.get_sensor_observations()
state = agent.get_state()
update_from_habitat(memory, obs, state, calib, timestamp, floor_y=floor_world_y)
place = topology.visit("start", state.position[[0, 2]], timestamp, "start", [observation_id])
```

传入同一时刻的原始深度和传感器位姿。深度为米制轴向距离，图像列向右、行向下，Habitat 相机朝局部 −Z。地图平面为世界 (X,Z)，高度为世界 Y；数组索引为 [Z 行, X 列]。地图左上格从 origin 开始，边界按左闭右开处理。地图大小由调用方按场景配置，底盘越界会在写入前报错。每次采集使用严格递增的时间戳；过期帧、重复帧和无效位姿都会被拒绝。一个流使用单写入者。

深度样本默认取 (0.05, 8) 米内的有限值，每 4 像素采样。默认障碍高度带为距地面 0.12–1.65 米，同帧每格最多投一票，命中优先。空闲和占用的 log-odds 增量分别为 −0.45、+0.9，累积限制为 [−4,4]。`grid()` 输出 −1 未知或证据不足、0 空闲、100 占用。全无效深度帧可更新有效底盘位置和帧号，空间证据保持原值。

`visit_count` 统计底盘进入栅格的次数。若执行器提供经验证连续的轨迹，传入 `base_path=[(x0,z0), ..., (xn,zn)]`，端点须对应上一帧和当前帧。这样能统计中间经过的格子。只有离散位姿时只计采样点，原地转向不重复计数。

## 拓扑输入

在起点、抵达并观察 Frontier、进入新区域或有意义的观察位置调用 `visit`。默认半径 0.5 米内复用最近 Place。传感器每帧更新几何，只有这些事件更新 Place。Frontier 归属通过 `link_frontier(frontier_id, place_id)` 建立，传入 None 可解除引用。

```python
arrival = topology.visit("arrival-1", xy, t, "frontier_observed", ["obs-8"])
if place != arrival:
    edge = topology.record_navigation("action-1/success", place, arrival, "succeeded", actual_path_length)
```

边有方向。已经验证的规划可上报 `planned`，执行成功上报 `succeeded`。已有路线连续失败两次或被确认阻断后进入 BLOCKED，后续成功穿越可恢复 TRAVERSED。代价单位为米。每个事件使用场景内唯一 ID，重复提交相同事件不会重复计数。Place 与 Frontier 的关联只表示归属。

## 验收复现

在服务器项目根目录运行。最后一个参数选择空闲 GPU，输出目录必须是新路径。

```bash
V=/data/chenkejun/vln/envs/vln-sim/bin/python
export PYTHONPATH="$PWD/src"
"$V" -m unittest discover -s tests -p 'test_m1_*.py' -v
"$V" evals/eval_m1_habitat.py /data/chenkejun/datasets/mp3d_habitat/scenes /tmp/m1-mp3d-new --gpu 7
```

验收使用 Habitat-Sim 0.3.3 和项目感知实验的 MP3D 扫描 `2azQ1b91cZZ`。脚本直接加载 `mp3d.scene_dataset_config.json`、该扫描的原始 GLB 和同名 navmesh。起点沿用餐厅候选验证记录 `results/verify-e02-new-view/e02-view-comparison.json` 中的 old_view.position_m。脚本从记录的朝向开始，每隔 15° 检查四段 0.25 米直行路径，选出能走通、具有有效中心深度的朝向。朝向和实际行驶距离均写入报告。

机器人执行 4 次前进和 4 次左转。M1 接收深度、内参、位姿和已执行的底盘轨迹。验收器单独从原始 GLB 读取三角面，在 MP3D 配置规定的坐标系下求交，逐帧核对中心深度与命中点附近的占用栅格，并用密采样轨迹核对访问次数。此轨迹的实测结果位于 `results/m1-mp3d-2azQ1b91cZZ/`，其中 report.json 记录输入资产的 SHA-256，occupancy.png 展示地图与轨迹。

当前结果覆盖此固定场景轨迹、合成几何和拓扑事件。接入其他场景时，需要使用对应传感器的深度约定、标定、地面高度和地图范围。完整导航任务成功率与真实机器人效果需要单独评测。
