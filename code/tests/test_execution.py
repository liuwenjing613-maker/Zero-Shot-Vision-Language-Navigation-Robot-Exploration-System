"""模块五 Execution 的单元测试。

用 stdlib unittest，零第三方依赖（只需 numpy）。
全部跑在 MockBackend 上，**不需要 Habitat、不需要 GPU、不需要联网**。

跑法：
    cd tls
    python code/tests/test_execution.py
    # 或
    python -m unittest discover -s code/tests -t code

覆盖：
  · 数据结构（ExecutionResult 工厂方法 / to_dict）
  · MockBackend 五个方法的行为
  · TaskExecutor 的分发、坐标解析、失败路径、异常兜底
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.enums import ActionType, EntityKind, ErrorCode, ExecutionStatus  # noqa: E402
from common.schemas import ActionDecision, ExecutionResult  # noqa: E402
from execution.mock_backend import MockBackend  # noqa: E402
from execution.robot_backend import RobotBackend  # noqa: E402
from execution.task_executor import TaskExecutor  # noqa: E402


def make_decision(action_type: str, **kw) -> ActionDecision:
    return ActionDecision(
        task_id="T1", decision_id="D1", action_type=action_type, **kw
    )


# ══════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════


class TestSchemas(unittest.TestCase):
    def test_succeeded_factory(self):
        r = ExecutionResult.succeeded("T1", "D1", final_pose=(1.0, 2.0, 3.0))
        self.assertEqual(r.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(r.task_id, "T1")
        self.assertEqual(r.action_id, "D1")

    def test_failed_factory_carries_error_code(self):
        r = ExecutionResult.failed("T1", "D1", ErrorCode.NO_PATH, "不可达")
        self.assertEqual(r.status, ExecutionStatus.FAILED)
        self.assertEqual(r.error_code, ErrorCode.NO_PATH)
        self.assertEqual(r.error_message, "不可达")

    def test_to_dict_is_json_serializable(self):
        import json

        r = ExecutionResult.succeeded("T1", "D1", final_pose=(1.5, 2.5, 3.5),
                                      path_length=4.2, execution_time=0.3)
        d = r.to_dict()
        json.dumps(d)          # 不抛异常即说明可序列化
        self.assertEqual(d["final_pose"], [1.5, 2.5, 3.5])
        self.assertEqual(d["status"], "SUCCEEDED")

    def test_to_dict_handles_none_pose(self):
        d = ExecutionResult.failed("T1", "D1", ErrorCode.TIMEOUT).to_dict()
        self.assertIsNone(d["final_pose"])


# ══════════════════════════════════════════════════════════════════
# MockBackend
# ══════════════════════════════════════════════════════════════════


class TestMockBackend(unittest.TestCase):
    def test_is_a_robot_backend(self):
        self.assertIsInstance(MockBackend(), RobotBackend)

    def test_observe_returns_complete_packet(self):
        pkt = MockBackend().observe()
        self.assertIsNotNone(pkt.rgb)
        self.assertIsNotNone(pkt.depth)
        self.assertIsNotNone(pkt.robot_pose)
        self.assertEqual(pkt.source, "mock")
        self.assertEqual(len(pkt.position()), 3)

    def test_scan_returns_multiple_frames(self):
        frames = MockBackend().scan()
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(f.robot_pose is not None for f in frames))

    def test_navigate_to_success_updates_position(self):
        b = MockBackend(position=(0.0, 0.0, 0.0))
        r = b.navigate_to((2.0, 0.0, 3.0))
        self.assertEqual(r.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(b.position, (2.0, 0.0, 3.0))

    def test_navigate_to_unreachable(self):
        b = MockBackend(reachable={(0.0, 0.0, 0.0)})
        r = b.navigate_to((9.0, 0.0, 9.0))
        self.assertEqual(r.status, ExecutionStatus.FAILED)
        self.assertEqual(r.error_code, ErrorCode.NO_PATH)

    def test_fault_injection_timeout(self):
        b = MockBackend()
        b.next_failure = "TIMEOUT"
        r = b.navigate_to((1.0, 0.0, 1.0))
        self.assertEqual(r.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(r.error_code, ErrorCode.TIMEOUT)

    def test_fault_injection_stuck(self):
        b = MockBackend()
        b.next_failure = "STUCK"
        r = b.navigate_to((1.0, 0.0, 1.0))
        self.assertEqual(r.status, ExecutionStatus.FAILED)
        self.assertEqual(r.error_code, ErrorCode.STUCK)

    def test_cancel_navigation(self):
        b = MockBackend()
        r = b.cancel_navigation()
        self.assertEqual(r.status, ExecutionStatus.CANCELED)
        self.assertTrue(b.canceled)

    def test_stop_succeeds(self):
        r = MockBackend().stop()
        self.assertEqual(r.status, ExecutionStatus.SUCCEEDED)


# ══════════════════════════════════════════════════════════════════
# TaskExecutor
# ══════════════════════════════════════════════════════════════════


class TestTaskExecutor(unittest.TestCase):
    def test_approach_target_resolves_pose_and_navigates(self):
        b = MockBackend()
        poses = {("target", "O01"): (3.0, 0.0, 4.0)}
        ex = TaskExecutor(b, pose_resolver=lambda k, i: poses.get((k, i)))

        r = ex.execute(make_decision(ActionType.APPROACH_TARGET, target_id="O01"))
        self.assertEqual(r.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(b.position, (3.0, 0.0, 4.0))

    def test_explore_frontier_uses_frontier_namespace(self):
        b = MockBackend()
        calls = []

        def resolver(kind, entity_id):
            calls.append((kind, entity_id))
            return (1.0, 0.0, 1.0)

        ex = TaskExecutor(b, pose_resolver=resolver)
        r = ex.execute(make_decision(ActionType.EXPLORE_FRONTIER, frontier_id="F01"))
        self.assertEqual(r.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(calls, [(EntityKind.FRONTIER, "F01")])

    def test_stop_action(self):
        b = MockBackend()
        ex = TaskExecutor(b)
        r = ex.execute(make_decision(ActionType.STOP))
        self.assertEqual(r.status, ExecutionStatus.SUCCEEDED)
        self.assertIn(("stop",), b.calls)

    def test_unsupported_action(self):
        ex = TaskExecutor(MockBackend())
        r = ex.execute(make_decision("FLY_TO_MOON"))
        self.assertEqual(r.status, ExecutionStatus.FAILED)
        self.assertEqual(r.error_code, ErrorCode.UNSUPPORTED_ACTION)

    def test_missing_entity_id(self):
        ex = TaskExecutor(MockBackend(), pose_resolver=lambda k, i: (0, 0, 0))
        r = ex.execute(make_decision(ActionType.APPROACH_TARGET, target_id=None))
        self.assertEqual(r.error_code, ErrorCode.UNKNOWN_ID)

    def test_resolver_returns_none(self):
        ex = TaskExecutor(MockBackend(), pose_resolver=lambda k, i: None)
        r = ex.execute(make_decision(ActionType.APPROACH_TARGET, target_id="O99"))
        self.assertEqual(r.error_code, ErrorCode.UNKNOWN_ID)

    def test_no_resolver_configured(self):
        ex = TaskExecutor(MockBackend())
        r = ex.execute(make_decision(ActionType.APPROACH_TARGET, target_id="O01"))
        self.assertEqual(r.error_code, ErrorCode.UNKNOWN_ID)

    def test_backend_exception_is_contained(self):
        """后端抛异常时，Executor 返回 INTERNAL_ERROR，不向外抛。"""
        class Boom(MockBackend):
            def navigate_to(self, pose):
                raise RuntimeError("驱动崩了")

        ex = TaskExecutor(Boom(), pose_resolver=lambda k, i: (0.0, 0.0, 0.0))
        r = ex.execute(make_decision(ActionType.APPROACH_TARGET, target_id="O01"))
        self.assertEqual(r.status, ExecutionStatus.FAILED)
        self.assertEqual(r.error_code, ErrorCode.INTERNAL_ERROR)
        self.assertIn("RuntimeError", r.error_message)

    def test_executor_fills_task_and_action_ids(self):
        ex = TaskExecutor(MockBackend(), pose_resolver=lambda k, i: (0.0, 0.0, 0.0))
        r = ex.execute(make_decision(ActionType.APPROACH_TARGET, target_id="O01"))
        self.assertEqual(r.task_id, "T1")
        self.assertEqual(r.action_id, "D1")
        self.assertGreaterEqual(r.execution_time, 0.0)

    def test_navigation_failure_propagates(self):
        b = MockBackend()
        b.next_failure = "NO_PATH"
        ex = TaskExecutor(b, pose_resolver=lambda k, i: (1.0, 0.0, 1.0))
        r = ex.execute(make_decision(ActionType.EXPLORE_FRONTIER, frontier_id="F01"))
        self.assertEqual(r.status, ExecutionStatus.FAILED)
        self.assertEqual(r.error_code, ErrorCode.NO_PATH)


# ══════════════════════════════════════════════════════════════════
# 接口合规
# ══════════════════════════════════════════════════════════════════


class TestInterfaceCompliance(unittest.TestCase):
    def test_robot_backend_is_abstract(self):
        with self.assertRaises(TypeError):
            RobotBackend()

    def test_mock_implements_everything(self):
        self.assertEqual(MockBackend.__abstractmethods__, frozenset())

    def test_habitat_backend_matches_signatures(self):
        """HabitatBackend 与抽象接口的方法签名一致（不实例化，只查签名）。"""
        import inspect

        from execution.habitat_backend import HabitatBackend

        self.assertTrue(issubclass(HabitatBackend, RobotBackend))
        self.assertEqual(HabitatBackend.__abstractmethods__, frozenset())
        for name in ("observe", "scan", "navigate_to",
                     "cancel_navigation", "stop"):
            base = inspect.signature(getattr(RobotBackend, name))
            impl = inspect.signature(getattr(HabitatBackend, name))
            self.assertEqual(list(base.parameters), list(impl.parameters),
                             f"{name} 参数不一致")

    def test_habitat_backend_imports_without_habitat_sim(self):
        """没装 habitat_sim 也能 import —— 靠延迟导入。"""
        import importlib

        mod = importlib.import_module("execution.habitat_backend")
        self.assertTrue(hasattr(mod, "HabitatBackend"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
