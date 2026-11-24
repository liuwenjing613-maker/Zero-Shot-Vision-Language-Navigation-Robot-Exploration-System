# 岔路建点与 MP3D 验收

每个场景和楼层实例化一个 `JunctionDetector(graph, robot_radius, margin)`。扫描融合后调用 `detector.update(grid, (x,z), timestamp, observation_id, area_id=area_id)`，返回已确认到达的 Place ID。候选可以从 `detector.candidates` 读取；节点保存地图坐标系下的分支方向、长度和最小宽度。0° 沿 +X，90° 沿 +Z。导航执行器在实际成功走完路径后使用已有的 `TopologyMemory.record_navigation` 记录有向边。

检测只使用已知空闲格。先按机器人半径和两侧安全余量计算安全区，再做 Zhang–Suen 骨架细化、短枝剪除、相邻分叉像素聚类与路径追踪。分支最低长度为 1 米，方向至少相隔 40°，宽度至少容纳机器人及余量。按可通行环带和径向开口区分独立出口。通过分支中心线拟合和局部净空估计候选中心。机器人距离中心小于 0.5 米、两个独立扫描的位移最多 0.12 米并且持续确认至少三个相符方向后，调用已有的拓扑建点接口。停留期间只触发一次；离开 0.85 米后重新到达可累计访问。同区邻近节点必须能通过已知自由空间直接连接才能复用。一次分支观察只提供可能的通行方向。

实验在 2azQ1b91cZZ 的同一层独立选取 T、Y、十字、直廊、L 弯和双向门口。参考位置及方向来自 Habitat 导航网格上约 1.5 米和 1.9 米两个尺度的可行扇区，检测器的输入为机器人实际移动后每帧 720 束、最大量程 6 米的模拟水平扫描和位姿。每条路线从 1.5 米外走近，停稳，再离开并返回。正例要求两次事件均发生在停稳阶段、分支数量正确、与参考方向一一对应的最大误差不超过 28°、复访 Place ID 相同。负例要求全程零事件。网页上紫色方向线仅用于验收。首轮失败已归档在实验目录的 first 子目录。

在服务器项目根目录复现：

```bash
export PYTHONPATH="$PWD/src" HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
V=/data/chenkejun/vln/envs/vln-sim/bin/python
JW=/data/chenkejun/vln/junction-work
mkdir -p "$JW"
"$V" -m unittest discover -s tests -v
"$V" evals/junction/select_routes.py "$JW/routes.json"
"$V" evals/junction/walk.py "$JW"
"$V" evals/junction/report.py "$JW" results/junction
```
