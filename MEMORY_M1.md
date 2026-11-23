# M1 几何与拓扑记忆

一个场景、一个楼层使用一组 `OccupancyMemory` 和 `TopologyMemory`。同一场景切换任务时可以保留这两个实例。几何和拓扑记忆使用 NumPy 与 SciPy，可用 `python -m pip install -e '.[memory]'` 安装。Habitat 适配器在调用时使用仿真环境自带的 quaternion。

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

## 360° 二维扫描验收

首页 `http://127.0.0.1:18764/` 默认打开 MP3D 餐厅的整圈扫描实验。左侧按圈累计建图，可切换命中点、参考可见表面轮廓与全部射线。右侧在传感器坐标系显示当前整圈距离回波。每圈 720 个方位，间隔 0.5°，量程 0.05–6 米。切换「已知尺寸房间」查看从柱体正面绕到背面时遮挡区的变化。

餐厅的「实际场景俯视图」来自同一 MP3D 扫描的彩色纹理网格。参考相机在起点地面上方 2.05 米正交向下拍摄；浅灰色表示裁剪高度范围内无可见表面。图像每格 8 像素，使用累计网格的范围、原点与 X/Z 方向。两图支持同步选格和按圈显示轨迹；「在实景上叠加占用格」显示当前圈累计的障碍格。家具顶面与离底盘 0.45 米处的水平扫描截面可以自然不同。直接打开 `/?view=360&scan=15` 查看去程终点。

重建参考图可运行 `$V evals/m1_lab/render_topdown.py /data/chenkejun/datasets/mp3d_habitat/scenes "$R/verified/laser360/0.json" results/m1-mp3d-2azQ1b91cZZ --gpu 3`。脚本检查地图角点的投影与向下独立取景的纹理采样。原图与配准参数由 `/mp3d/topdown.jpg` 和 `/mp3d/topdown.json` 提供。

调用 `OccupancyMemory.update_scan` 时提供米制平面距离向量、起始角、角增量、传感器平面位置和朝向、底盘位置及时间戳。内部 map 的 +X 为零角，正角转向 +Z。驱动需根据设备约定转换坐标和无回波值。正无穷表示已确认至最大量程为空；缺失或无效测量不产生空间证据。每圈每格最多一票，障碍命中优先。当前实验假设单圈采集时位姿固定。

解析房间使用 8 × 6 米墙体和实心柱，通过射线与矩形的解析交点生成测量。验收器独立对线段求交，并检查柱内、墙外和首圈柱后未知，换站后原遮挡区可见。MP3D 使用扫描 `2azQ1b91cZZ` 原始场景，在同一底盘位姿渲染四个共光心的水平 90° 深度视角，取水平行并将轴向深度换成径向距离。每个站点的扫描面在底盘上方 0.45 米；站点沿 Habitat navmesh 往返路径选取，使用已知位姿。拓扑边显示规划连通性。MP3D 的参考来自原始 GLB 对扫描平面的切片和首个可见面的几何求交，不送入建图器。

验收器使用每圈 1,440 个参考方向。占用和空闲 IoU 使用精确栅格交并比，同时报告允许 0.15 米偏差的占用精确率与召回率。对空闲区分别报告有效束角区和全角理想可见范围，保留缺测造成的覆盖差异。在参考与测量都有回波的方向统计每圈测距 P95，同时检查全部方向的命中状态一致率。负对照将射线角度故意错置 90° 并断言正确触发验收失败。结果覆盖固定扫描高度、理想位姿和仿真回波；真机噪声、扫描运动畸变和定位误差需要独立数据。

## 逐步查看占据网格与拓扑图

通过首页「深度与拓扑实验」或 `http://127.0.0.1:18764/depth` 打开原深度回放。选择「可控输入实验」检查墙前、墙面、墙后和导航边的状态转换。选择「MP3D 餐厅实测」检查 Habitat 的每帧 RGB、深度、实走轨迹与 Place。点击栅格可检查 log-odds 和进入次数。首帧的「候选框与输入核对」展示历史感知输入的哈希、目标框和观测 ID。历史结果只挂接在对应帧的 Place；其余帧没有执行视觉模型推理。

「服务器重新运行」先执行可控输入，验证通过后再启动 Habitat。服务逐步写出原始状态与验证结果，浏览器跟随最新步骤。也可以暂停播放、拖动进度条或跳到任一步复查。报告中的已知连续轨迹来自实际执行位姿；受控实验中的失败和重试事件来自模拟执行器。

重启服务时，在服务器项目根目录执行下列命令。首行选择本项目的 Habitat Python。首次部署需生成通过验证的默认回放。

```bash
V=/data/chenkejun/vln/envs/vln-sim/bin/python
export PYTHONPATH="$PWD/src" HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
R=/data/chenkejun/vln/m1-web-runs
"$V" evals/m1_lab/run.py "$R/verified" --gpu 7
"$V" evals/m1_lab/run.py "$R/verified" --mode scans --gpu 3
"$V" evals/m1_lab/serve.py "$R" --port 18764 --gpu 3
```

若 Windows 的 SSH 会话断开，使用下面的命令恢复本地通道并保持会话运行。网页服务仅监听服务器回环地址。

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18764:127.0.0.1:18764 -p 64906 chenkejun@frp-van.com
```

## 在线整体测试

手动驾驶入口为 `/live`，可同时查看实时扫描与按帧提交的视觉语义、门口、岔路和拓扑事件。部署、预热和实验导出见[驾驶台说明](evals/live/README.md)。
