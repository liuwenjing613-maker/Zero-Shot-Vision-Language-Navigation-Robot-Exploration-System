# V4 Decision 实现说明

日期：2026-11-23

## 交付范围

本次只完成 V4 的 `common` 协议和 `decision` 模块，不修改 v3、`lwj` 感知实现、
World Memory、Semantic Exploration 或底层执行后端。

新增目录：

```text
code/v4/
├── common/schemas.py
└── decision/
    ├── adapters.py
    ├── planner_context.py
    ├── task_state_machine.py
    ├── high_level_planner.py
    ├── recovery_manager.py
    ├── demo_decision_loop.py
    └── tests/test_decision.py
```

## `lwj` 分支已经实现的内容

截至提交 `4e0aa15`，`lwj` 是一个独立的 `vln_perception` Python 包：

1. `VLMClient.parse_goal()`：把自然语言指令转为目标类别、属性、关系、完成条件。
2. `VLMClient.perceive_scene()`：针对单帧 RGB 输出目标候选、环境对象、视觉线索；
   bbox 是 `0..1000` 的归一化坐标，`local_id` 仅在当前观测内有效。
3. `VLMClient.verify_target()`：在候选框上绘制标记后，返回 `VERIFIED`、`REJECTED` 或
   `UNCERTAIN` 和相应视觉证据。
4. 四个评测脚本及样例结果。

它不保存跨帧世界状态，不做深度投影或路径规划，也不直接生成导航动作。

## 本分支的 Decision 如何与 `lwj` 对齐

```text
lwj.parse_goal()        → goal_spec_from_lwj()       → GoalSpec
lwj.perceive_scene()    → perception_result_from_lwj() → PerceptionResult
PerceptionResult + RGB-D → Geometry / World Memory   → WorldObject
WorldState + 排序前沿     → PlannerContext            → ActionDecision
ActionDecision           → Execution                 → ExecutionResult
```

不能把 `lwj` 的 `local_id` 当成 `WorldObject.object_id`，也不能把 bbox 当作世界坐标。
Geometry 必须从 bbox 的中央有效深度区域取稳健深度统计，再利用相机内外参投影；深度无效时
只创建未定位候选。Decision 对未定位候选只能返回 `SCAN`，不会输出导航命令。

## 决策逻辑

`HighLevelPlanner.step()` 按以下确定性优先级选动作：

1. 执行失败：调用 `RecoveryManager`，本地规则处理，不问 Qwen。
2. 定位或传感器异常：`RELOCALIZE`。
3. 存在类别/属性匹配、可达、位置可信且 `CONFIRMED` 的目标：`APPROACH_TARGET`。
4. 存在匹配但未确认的候选：`OBSERVE_TARGET`；若尚无三维位置，则 `SCAN`。
5. 存在可达的历史目标：`REVISIT_TARGET`。
6. 没有可靠目标时，从 Semantic Exploration 给出的排序列表选择可达且非
   `BLOCKED/RESOLVED/STALE/COOLDOWN` 的第一个：`EXPLORE_FRONTIER`。
7. 前沿为空，或搜索预算耗尽：`STOP`。

每项 `ActionDecision` 都附带 `map_version` 和 `observation_id`。执行前必须调用
`planner.is_fresh()`；地图或观测已经变化时，旧决策不得执行。

## 状态机和恢复

状态机顺序为：

```text
IDLE → PARSE_TASK → INITIALIZE → OBSERVE → UPDATE_MEMORY → DECIDE
DECIDE → APPROACH / EXPLORE / VERIFY / RECOVERY
执行成功 → OBSERVE；执行失败 → RECOVERY；验证失败 → REPLAN → DECIDE；验证成功 → SUCCESS
```

恢复规则：

| 故障 | 决策 | 给 Memory / Execution 的指令 |
|---|---|---|
| `NO_PATH` / `UNREACHABLE` | `RECOVER` | 取消导航、将当前前沿标记 `BLOCKED`、重选 |
| `TIMEOUT` | `RECOVER` | 取消导航、`COOLDOWN`、重新观察 |
| `STUCK` / `COLLISION` | `RECOVER` | 停止、重新观察、重新规划 |
| `TARGET_REJECTED` | `SCAN` | 记录并降级目标，再搜索 |
| `VLM_TIMEOUT` | `SCAN` | 只允许一次 VLM 重试，之后回退本地规则 |
| 定位/传感器异常 | `RELOCALIZE` | 禁止继续导航 |

Decision 自身只返回指令，不直接篡改 Frontier 或 Object Memory；集成器/Memory 根据
`recovery_directives` 完成状态更新。这保证模块边界可测试。

## 验证

在 Windows Python 环境完成：

```text
python -m compileall -q code/v4
PYTHONPATH=code python -m unittest discover -s code/v4/decision/tests -t code -v
```

结果：12/12 通过。覆盖目标优先、未确认/未定位候选、前沿过滤、搜索耗尽、传感器保护、
路径失败恢复、决策新鲜度、`lwj` 适配和状态机闭环。

## 仍待系统集成的接口

- 成员 B 的 World Memory：输出 `WorldState`、全局 `WorldObject` 和 object 状态。
- 成员 C 的 Geometry：将 `lwj` bbox 和 RGB-D 融合到真实世界坐标。
- 成员 C 的 Exploration：输出按 `local_utility` / 可选 Qwen 排序的
  `FrontierCandidate` 列表。
- Execution：根据 `ActionDecision` 的对象/前沿 ID 从 World State 解析目标位置，
  反馈 `ExecutionResult`。

当前未推送的原因：本地工作区没有 GitHub remote，且公开上游仓库只存在 `main`、`lwj`、
`env` 分支，未发现你的 fork 或 `hyf` 分支。需要提供你的 GitHub fork/repository URL，
才能将本次提交准确推到你指定的 `hyf` 分支，而不误写上游或他人分支。
