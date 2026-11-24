# V4 Decision 实现说明

日期：2026-11-23（已于 2026-11-24 对齐最新 `lwj` / `tls`，详见
`TEAM_BRANCH_ADAPTATION_20261124.md`）

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

## 与 tls Execution 分支的适配（2026-11-23 复核）

`tls` 的 `TaskExecutor` 已实现并测试通过的动作是 `APPROACH_TARGET`、
`EXPLORE_FRONTIER`、`STOP`，与本模块的同名动作语义一致：决策只给全局 ID，
Execution 再通过 pose resolver 查到世界坐标。因此 `execution_bridge.py` 会将 V4
的枚举动作规范化为 TLS 的字符串 payload，并将 TLS 的字符串状态转换回 V4
`ExecutionResult`。

不是 TLS 的缺陷、也不应由本模块伪造的能力包括：

- `SCAN` / `OBSERVE_TARGET` / `REVISIT_TARGET`：总控调用 TLS `RobotBackend.scan()`，
  再回到 Perception 和 Memory。
- `RECOVER`：总控先调用 TLS `cancel_navigation()`，随后将本模块给出的恢复指令交给
  Memory。
- `VERIFY`：交给 `lwj` 的 `verify_target()`。
- `RELOCALIZE`：需要未来定位模块；TLS 第一版尚未声明该能力。

本次仅修改 `hyf`，没有触碰 `tls`。TLS 的 27 项单元测试在独立快照中全部通过；
`lwj` 的 Goal / Scene schema 校验通过。当前 Windows 审计 Python 缺少 `Pillow`，
所以未运行 `lwj` 依赖图像标注的 Target Verification；该依赖已在 `lwj` 的
`pyproject.toml` 中声明，并非本模块错误。

实现已推送到原协作仓库的 `hyf` 分支。提交作者使用已关联 `yf-eve` 的 GitHub noreply
地址；本模块没有直接推送或修改 `main`、`lwj`、`tls`。

## 三分支模拟闭环（2026-11-23）

`mock_closed_loop.py` 使用 lwj 格式的确定性感知 JSON、hyf 的实际 Planner 和 TLS 的
`MockBackend + TaskExecutor` 跑通以下事件序列：

```text
未定位红椅候选 → SCAN（TLS 返回 4 帧）
无可靠目标 → EXPLORE_FRONTIER（TLS 导航成功）
几何/记忆确认红椅 → APPROACH_TARGET（TLS 导航成功）
模拟 NO_PATH → RECOVER（TLS 取消导航，Frontier 指令为 BLOCKED）
```

该测试不调用真实 Qwen 或 Habitat，不把模拟结果误当作真实导航评测；它验证的是三分支的
数据契约、动作分发与失败恢复是否能闭环。运行方式见 `code/v4/decision/README.md`。
