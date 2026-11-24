"""模块三 Semantic Exploration 的演示。

构造一个合成场景（一个房间 + 两个未知方向 + 一个柱子），
跑完检测 → 聚类 → Viewpoint → 打分排序，把结果打印出来。

跑法：
    cd tls
    python code/exploration/demo_exploration.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from exploration import FrontierExplorer, ViewpointConfig  # noqa: E402
from exploration.detector import detect_regions  # noqa: E402
from exploration.gridops import FREE, OCCUPIED, UNKNOWN, clearance_map  # noqa: E402

RES = 0.1  # 0.1 米/格


def build_scene() -> np.ndarray:
    """一个 80×80（8m×8m）的场景：

        ┌──────────────────────────┐
        │  未知（右侧/上侧未探索）    │
        │                          │
        │   ┌──────────────┐       │
        │   │              │       │
        │   │   空闲房间    │  ●    │  ← ● 是柱子
        │   │              │       │
        │   └──────────────┘       │
        │      未知（下方）          │
        └──────────────────────────┘
    """
    values = np.full((80, 80), UNKNOWN, dtype=np.int16)
    # 房间主体
    values[25:60, 20:65] = FREE
    # 两道从边界伸进来的隔断墙：把一圈前沿切成左右两个区域
    # （墙没有贯穿整个房间，所以自由空间仍然连通）
    values[25:38, 42] = OCCUPIED
    values[52:60, 42] = OCCUPIED
    # 一根柱子，测试避障与视野遮挡
    values[40:46, 50:54] = OCCUPIED
    return values


def main() -> None:
    values = build_scene()
    print("场景：80x80 格（8m x 8m），分辨率 %.2f m/格" % RES)
    print(f"  空闲格：{int((values == FREE).sum())}")
    print(f"  未知格：{int((values == UNKNOWN).sum())}")
    print(f"  占用格：{int((values == OCCUPIED).sum())}")

    robot_cell = (55, 30)  # 机器人站在房间左下角
    print(f"\n机器人位置（栅格）：{robot_cell}")

    # ── 1. 检测与聚类 ────────────────────────────────────────
    regions = detect_regions(values, RES, min_cells=12)
    print(f"\n检测到 {len(regions)} 个前沿区域：")
    for i, r in enumerate(regions, 1):
        print(f"  R{i}: {r.cluster_size:>3} 格  质心(x,z)=({r.centroid_xy[0]:.2f}, {r.centroid_xy[1]:.2f})")

    clearance = clearance_map(values, RES)
    print(f"\n机器人处 clearance = {clearance[robot_cell]:.2f} m")

    # ── 2. 完整流程 ──────────────────────────────────────────
    cfg = ViewpointConfig(sample_radius=2.0, sensor_range=5.0, num_rays=72)
    explorer = FrontierExplorer(viewpoint_config=cfg, min_cells=12)

    snapshot = explorer.detect(
        values, RES, origin=(0.0, 0.0), robot_cell=robot_cell,
        map_version=1, floor_y=0.0,
    )
    # rank() 在没有 mapping 时把 local id "1" 映射成 "F1"，这里保持同样的键
    raw = {f"F{c['local_candidate_id']}": c for c in snapshot["candidates"]}
    candidates = explorer.rank(snapshot)

    print(f"\n{'=' * 84}")
    print(f"{'排名':<4}{'ID':<6}{'Cluster':>8}{'可见未知格':>11}{'路径(m)':>9}"
          f"{'余量(m)':>9}{'增益(归一)':>11}{'Utility':>9}")
    print(f"{'=' * 84}")

    for rank, c in enumerate(candidates, 1):
        n = raw.get(c.frontier_id, {})
        print(f"{rank:<4}{c.frontier_id:<6}{c.cluster_size:>8}"
              f"{n.get('gain_cells', 0):>11}"
              f"{c.path_length:>9.2f}{c.clearance:>9.2f}"
              f"{c.information_gain:>11.3f}{c.local_utility:>9.3f}")

    best = explorer.select_best(candidates)
    if best:
        print(f"\n{'=' * 78}")
        print("推荐探索目标：")
        print(f"  frontier_id   = {best.frontier_id}")
        print(f"  质心 (x,y,z)  = ({best.centroid[0]:.2f}, {best.centroid[1]:.2f}, {best.centroid[2]:.2f})")
        print(f"  观察点        = ({best.safe_viewpoint[0]:.2f}, {best.safe_viewpoint[1]:.2f}, {best.safe_viewpoint[2]:.2f})")
        print(f"  路径长度      = {best.path_length:.2f} m")
        print(f"  综合得分      = {best.local_utility:.4f}")
        print(f"\n  → 下一步应执行 ActionDecision(")
        print(f"        action_type='EXPLORE_FRONTIER', frontier_id='{best.frontier_id}')")
    else:
        print("\n没有可探索的前沿（地图已完全探索）")


if __name__ == "__main__":
    main()
