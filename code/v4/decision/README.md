# V4 Decision

Decision implements the frozen V4 policy: **rule-constrained FSM + local
utility-ranked exploration + optional upstream semantic tie-breaking**. It does
not call Qwen, update maps, mutate Memory, or invoke a robot backend.

## Public interfaces

- `HighLevelPlanner.step(context) -> ActionDecision`
- `TaskStateMachine` — records the legal stage sequence around the planner.
- `RecoveryManager.plan(context) -> RecoveryPlan | None`
- `build_planner_context(...) -> PlannerContext`
- `goal_spec_from_lwj()` / `perception_result_from_lwj()` — adapters for the
  schemas already implemented on the `lwj` branch.

## Priority order

1. Execution failure → deterministic recovery.
2. Invalid pose or sensor → `RELOCALIZE`.
3. A geometrically localized, reachable, confirmed matching target →
   `APPROACH_TARGET`.
4. A matching uncertain target → `OBSERVE_TARGET` (or `SCAN` if unlocalized).
5. A reachable historical target → `REVISIT_TARGET`.
6. A reachable, non-blocked ranked frontier → `EXPLORE_FRONTIER`.
7. Otherwise, or after budget exhaustion → `STOP`.

Each decision carries `map_version` and `observation_id`; Execution must call
`planner.is_fresh()` before acting on a delayed decision.

## lwj Perception integration

The `lwj` package produces observation-local IDs and normalized `0..1000`
bounding boxes. Its outputs are not global object records and cannot be used as
navigation coordinates. The integration order is:

```text
lwj perceive_scene → PerceptionResult → Geometry (RGB-D projection)
→ WorldMemory association / WorldObject → PlannerContext → Decision
```

Only World Memory may create global `object_id` values. Geometry must convert a
bbox's central region into a robust depth estimate, then project it to world
coordinates. Invalid depth stays an unlocalized candidate, which Decision only
asks to observe again; it never approaches it.

## tls Execution integration

The independently developed `tls` branch exposes `TaskExecutor` and an
ID-to-pose resolver. `execution_bridge.py` converts APPROACH_TARGET,
EXPLORE_FRONTIER, and STOP to its ID-only `ActionDecision` payload, and converts
TLS string statuses back to the V4 `ExecutionResult` contract.

SCAN / OBSERVE_TARGET / REVISIT_TARGET call `RobotBackend.scan()` through the
orchestrator. RECOVER calls `cancel_navigation()` before Memory applies its
recovery directives. VERIFY and RELOCALIZE deliberately remain external: they
belong to `lwj` verification and a future localization module, respectively.

## Run

```bash
PYTHONPATH=code python -m unittest discover -s code/v4/decision/tests -t code -v
PYTHONPATH=code python -m v4.decision.demo_decision_loop
```
