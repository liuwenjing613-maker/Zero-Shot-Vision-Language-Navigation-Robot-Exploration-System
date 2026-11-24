"""Real TLS frontier generator -> hyf Decision -> TLS MockBackend test.

The map is synthetic, but all three modules used here are their real merged
implementations. Habitat, Qwen and GPU are deliberately not prerequisites.
"""

from __future__ import annotations

import unittest

import numpy as np

from execution import MockBackend, TaskExecutor
from exploration.explorer import FrontierExplorer
from exploration.gridops import FREE, UNKNOWN
from exploration.viewpoint import ViewpointConfig
from v4.common.schemas import ActionType, GoalSpec
from v4.decision import HighLevelPlanner, planner_context_from_team_outputs
from v4.decision.execution_bridge import (
    TLSDispatchOperation,
    execution_result_from_tls,
    plan_tls_dispatch,
)


class TLSFrontierExecutionIntegrationTests(unittest.TestCase):
    def test_real_frontier_is_selected_and_executed(self):
        # A known free room surrounded by unknown cells produces a real TLS
        # frontier; the small configuration keeps this deterministic and fast.
        values = np.full((60, 60), UNKNOWN, dtype=np.int16)
        values[20:40, 10:50] = FREE
        explorer = FrontierExplorer(
            viewpoint_config=ViewpointConfig(
                sample_radius=1.0, sensor_range=2.0, num_rays=16, max_candidates=60,
            ),
            min_cells=8,
        )
        frontiers = explorer.explore(
            values, resolution=0.1, origin=(0.0, 0.0), robot_cell=(30, 30),
            map_version=12,
        )
        self.assertTrue(frontiers)

        goal = GoalSpec(task_id="sim-frontier", target_category="chair")
        context = planner_context_from_team_outputs(
            goal,
            {"map_version": 12, "object_memory": {"objects": {}, "verification": []}},
            frontiers,
            robot_pose=(3.0, 0.0, 3.0),
        )
        decision = HighLevelPlanner().step(context)
        self.assertEqual(ActionType.EXPLORE_FRONTIER, decision.action_type)
        dispatch = plan_tls_dispatch(decision)
        self.assertEqual(TLSDispatchOperation.TASK_EXECUTOR, dispatch.operation)

        poses = {candidate.frontier_id: candidate.safe_viewpoint for candidate in frontiers}
        backend = MockBackend(reachable=set(poses.values()))
        executor = TaskExecutor(backend, lambda _kind, identifier: poses.get(identifier))
        result = execution_result_from_tls(executor.execute(dispatch.executor_payload))
        self.assertEqual("SUCCEEDED", result.status.value)
        self.assertEqual(poses[decision.frontier_id], result.final_pose)


if __name__ == "__main__":
    unittest.main()
