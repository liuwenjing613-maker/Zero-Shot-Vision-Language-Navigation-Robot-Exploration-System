"""A dependency-free, one-cycle V4 Decision demonstration.

Run from the repository root with:
    PYTHONPATH=code python -m v4.decision.demo_decision_loop
"""

from v4.common.schemas import FrontierCandidate, GoalSpec, WorldObject, WorldState
from v4.decision import HighLevelPlanner, build_planner_context


def main() -> None:
    goal = GoalSpec("demo-chair", "chair", {"color": "red"})
    frontier = FrontierCandidate(
        frontier_id="frontier-004", centroid=(4.0, 0.0, 2.0), safe_viewpoint=(3.5, 0.0, 2.0),
        cluster_size=12, reachable=True, path_length=3.7, clearance=0.8,
        information_gain=0.8, novelty=0.9, local_utility=0.71,
    )
    planner = HighLevelPlanner()

    # First observation: no chair has been localized, so exploration is legal.
    before = WorldState("demo-chair", 7, (0.0, 0.0, 0.0), frontiers=(frontier,))
    print(planner.step(build_planner_context(goal, before, (frontier,))))

    # After geometry + memory fuse a verified red chair, target approach wins.
    chair = WorldObject("obj-017", "chair", {"color": "red"}, (2.2, 0.0, 1.1), 0.92,
                        "CONFIRMED", 3, 0.18, True, ("obs-011:E02", "verify-003"))
    after = WorldState("demo-chair", 8, (1.0, 0.0, 0.0), objects=(chair,), frontiers=(frontier,))
    print(planner.step(build_planner_context(goal, after, (frontier,))))


if __name__ == "__main__":
    main()
