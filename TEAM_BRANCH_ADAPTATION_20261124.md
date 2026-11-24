# lwj / tls / hyf 适配复核（2026-11-24）

## 本次同步基线

- `lwj`: `93f7549` — M1 几何与拓扑、M2 对象证据、M3 前沿/区域记忆与 M4 事件回放。
- `tls`: `34fdf3d` — 已合并 `lwj`，另提供 `code/exploration` 和 `code/execution`。
- `hyf`: 在自己的分支中合并以上两个最新分支，并新增 Decision 适配层；没有修改
  `main`、远端 `lwj` 或远端 `tls`。

## 对齐结果

| 上游输出 | 实际来源 | hyf 使用方式 | 安全约束 |
| --- | --- | --- | --- |
| `object_memory.objects` / `verification` | `src/vln_memory/online_memory.py` | 转为 V4 `WorldObject` | 不能将分类、bbox 或验证结果直接当作可导航坐标 |
| `map_version`、事件 | `OnlineMemory.apply()` | 填入 `WorldState`，用于拒绝过期 Decision | `robot_pose` 不在该返回值中，必须由总控传入 |
| 排序后的 `FrontierCandidate` | `code/exploration/explorer.py::rank()` | 结构化复制为 V4 的同名契约 | 仅接受可达、未失效 frontier |
| 目标/前沿 ID 执行 | `code/execution/task_executor.py` | `execution_bridge.py` 分发 | 全局 ID 由 Memory 维护，Execution 再解析成 pose |

新增 `code/v4/decision/team_adapters.py` 的两个公共入口：

```python
world_state_from_online_memory(goal, online_update, robot_pose=pose, ...)
planner_context_from_team_outputs(goal, online_update, tls_frontiers, robot_pose=pose, ...)
```

其中 `reachable_object_ids` 是总控完成路径/可达性检查后传入的集合。只有同时满足
`VERIFIED + 3D position + reachable` 的对象才是 `CONFIRMED`，可触发
`APPROACH_TARGET`；缺少可达性证据时 Decision 只能 `OBSERVE_TARGET`。这正好匹配
`lwj` 的证据所有权、`tls` 的执行所有权和 `hyf` 的“只做高层确定性决策”边界。

## 已执行的验证

临时虚拟环境（未修改系统 Python）中安装了项目声明的 `Pillow`、`numpy`、`scipy`：

- `code/v4/decision/tests`: 33 项通过（含 3 项跨分支适配测试、13 项模拟决策矩阵，
  以及 1 项 TLS 实际前沿生成 → hyf 决策 → TLS MockBackend 执行测试）；另启用的
  三模块 Mock 闭环 1 项通过。
- `code/tests`: 63 项通过（TLS Execution、Frontier 检测/聚类/Viewpoint/评分，以及
  FrontierMemory 集成）。
- `tests`: 44 项通过（lwj M1 栅格/拓扑、M2 对象记忆、M3/M4 探索记忆、门户/区域）。

Mock 闭环覆盖：未定位候选 `SCAN`、无目标时 `EXPLORE_FRONTIER`、已确认且可达目标
`APPROACH_TARGET`、`NO_PATH` 后 `RECOVER` 并取消导航。它不调用 Qwen、Habitat 或
真实 MP3D；因此证明的是跨模块数据契约和安全分发，而不是仿真导航指标。

模拟矩阵额外发现并修正了一条 Decision 规则：多属性目标中，观测缺少某个属性
（例如颜色被遮挡的椅子）是未知证据，必须作为候选 `OBSERVE_TARGET`，而不能直接
丢弃后去探索；只有已观测到相冲突的属性（例如蓝色而目标要求红色）才排除候选。

后续扩展场景还修正了两条安全边界：已验证但位置方差过大、路径不再可达或置信度
降到接近阈值的目标，会回到 `OBSERVE_TARGET` 而非被静默遗忘；携带
`VLM_TIMEOUT` 的结果即使状态为 `TIMEOUT`，也优先遵循一次受限视觉重试规则，
不会被通用导航超时恢复逻辑误截获。

## 仍由总控/后续模块负责的接口

1. 从 Habitat/RobotBackend 获取当前 3-D `robot_pose`，显式传给适配器。
2. 对目标 ID 做真实可达性/路径检查，得到 `reachable_object_ids`。
3. `SCAN` / `OBSERVE_TARGET` 后调用感知与 `OnlineMemory.apply()`，再重建上下文。
4. `VERIFY` 调用 `lwj` 验证；`RELOCALIZE` 由定位模块实现。

这些不是 `hyf` 可安全猜测或替代的能力，因此本次没有改动它们。
