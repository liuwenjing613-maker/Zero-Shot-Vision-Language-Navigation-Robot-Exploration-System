import numpy as np

from record import Recorder
from vln_memory.geometry_memory import Intrinsics, OccupancyMemory
from vln_memory.topology_memory import TopologyMemory


def run(folder):
    grid = OccupancyMemory((17, 17), (-2.125, -2.125), 0.25)
    topo = TopologyMemory()
    rec = Recorder(folder, "controlled", grid, topo)
    home, dest = (0.0, 0.0), (1.0, 0.0)
    sample = lambda xy: int(grid.grid()[grid.cell(xy)])
    K = Intrinsics.from_hfov(1, 1, 80)

    def observe(xy, d, tick, path=None):
        pose = np.eye(4)
        pose[:3, 3] = [xy[0], 0.8, xy[1]]
        grid.update_depth(
            np.array([[d]], dtype=float),
            K,
            pose,
            (xy[0], 0, xy[1]),
            tick,
            floor_y=0,
            base_path=path,
        )

    a = topo.visit("start", home, 0, "start")
    rec.step(
        "初始化场景",
        "创建起点 P01，暂时没有深度输入。",
        [
            ("289 个栅格均为未知", (grid.grid() == -1).all()),
            ("只有一个节点且无边", len(topo.places) == 1 and not topo.edges),
        ],
        home,
    )
    observe(home, 1.0, 1)
    rec.step(
        "一条射线看见墙",
        "中心射线朝 −Z，墙面在 1 米处。点击墙前、墙面和墙后的格子可查看各自的证据。",
        [
            (
                "墙前空闲、终点占用、墙后未知",
                [sample((0, -z)) for z in (0.5, 1.0, 1.5)] == [0, 100, -1],
            ),
            ("底盘进入起点一次", grid.visit_count.sum() == 1),
        ],
        home,
        ray=[home, (0, -1)],
    )
    observe(home, 1.0, 2)
    rec.step(
        "原地再次观测",
        "相同墙面获得第二次命中。栅格分类可以保持不变，同时证据强度上升。",
        [
            (
                "墙面 log-odds 从 0.9 增至 1.8",
                np.isclose(grid.log_odds[grid.cell((0, -1))], 1.8),
            ),
            ("访问次数未增加", grid.visit_count.sum() == 1),
        ],
        home,
        ray=[home, (0, -1)],
    )
    old = grid.log_odds.copy()
    try:
        observe(home, 2.0, 2)
    except ValueError as exc:
        reason = str(exc)
    else:
        reason = ""
    rec.step(
        "拒绝重复时间戳",
        "用相同时间戳再次发送深度输入。模块返回的原因随本步保存。",
        [
            ("重复帧被拒收", bool(reason)),
            (
                "地图版本、访问次数和几何证据均保持",
                grid.map_version == 2
                and grid.visit_count.sum() == 1
                and np.array_equal(grid.log_odds, old),
            ),
        ],
        home,
        rejection=reason,
    )
    observe(dest, float("nan"), 3, [home, dest])
    b = topo.visit("arrive", dest, 3, "frontier_observed", ["obs-B"])
    topo.link_frontier("F01", b)
    rec.step(
        "经过四格，到达 P02",
        "输入已执行的一米轨迹和无效深度，再上报观察位置与 F01 引用。",
        [
            (
                "累计进入五格，原射线证据保留",
                grid.visit_count.sum() == 5 and np.array_equal(grid.log_odds, old),
            ),
            ("两个 Place 没有自动连边", len(topo.places) == 2 and not topo.edges),
            ("F01 指向 P02", topo.frontier_places.get("F01") == b),
        ],
        dest,
    )
    e = topo.record_navigation("plan", b, a, "planned", 1.0)
    rec.step(
        "规划返回 P01",
        "规划器确认 P02 到 P01 的一米路线。",
        [
            ("有向边为 PLANNED，成功次数为零", e.status == "PLANNED" and e.traversal_count == 0),
            ("反向边尚未建立", (a, b) not in topo.edges),
        ],
        dest,
    )
    for j in (1, 2):
        topo.record_navigation(f"fail{j}", b, a, "failed")
        rec.step(
            f"第{j}次执行失败",
            "模拟执行器上报失败，连续两次失败达到阻断阈值。",
            [
                ("失败次数与事件数量一致", e.failure_count == j),
                ("边状态符合阈值", e.status == ("PLANNED" if j == 1 else "BLOCKED")),
            ],
            dest,
        )
    observe(home, float("nan"), 4, [dest, home])
    actual = topo.visit("revisit", home, 4, "observation_place", ["obs-return"])
    topo.record_navigation("success", b, a, "succeeded", 1.0)
    rec.step(
        "走通并复用起点",
        "返回起点后上报成功，原有 P01 再次被访问，保留两次失败历史。",
        [
            (
                "同一 Place 的访问次数为 2",
                actual == a
                and topo.places[a].visit_count == 2
                and len(topo.places) == 2,
            ),
            ("回程后累计进入九次", grid.visit_count.sum() == 9),
            (
                "边恢复 TRAVERSED",
                e.status == "TRAVERSED"
                and e.traversal_count == 1
                and e.failure_count == 2
                and e.consecutive_failures == 0,
            ),
        ],
        home,
    )
    topo.record_navigation("success", b, a, "succeeded", 1.0)
    rec.step(
        "重发同一成功事件",
        "执行器重试发送相同事件 ID，检查幂等性。",
        [("成功仍为一次", e.traversal_count == 1), ("图仍只有一条有向边", len(topo.edges) == 1)],
        home,
    )
