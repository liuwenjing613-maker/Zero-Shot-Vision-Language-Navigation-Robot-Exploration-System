"""语义探索的演示与消融实验（S0/S2）。

两部分：

  1. 演示：同一张栅格、同样两个前沿，只改变"记忆里有什么物体"，
     看语义项如何改变排序。
  2. 消融：扫描语义权重 w_S 与影响半径 σ，统计"双前沿二选一"的判别准确率。

⚠️ 这里用的是**合成场景 + 人工摆放的物体**，只能证明打分机制按设计工作，
   不能给出真实场景下的成功率。真实评测需要在 MP3D 上按 .house 标注构造
   决策点（见 docs/模块三_语义扩展设计.md 的 S1 阶段）。

用法： PYTHONPATH=code python -m exploration.ablate_semantic
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .demo_exploration import build_scene
from .explorer import FrontierExplorer
from .semantic import (
    MemoryObject,
    ObjectPriorConfig,
    ObjectPriorProvider,
    euclidean_object_query,
)
from .scorer import UtilityWeights
from .viewpoint import ViewpointConfig

RES = 0.1
ROBOT = (55, 30)

# 与 demo_exploration.py 使用同一套观察点采样参数，保证两个脚本报出的
# 绝对效用值可以直接对照（否则同一个场景会出现两套数字）。
DEMO_VIEWPOINT = ViewpointConfig(sample_radius=2.0, sensor_range=5.0, num_rays=72)

# 其他五项权重的相对比例（与 UtilityWeights 的默认值一致，和为 0.70）
_GEOM_RATIO = {
    "information_gain": 0.25,
    "novelty": 0.10,
    "path_cost": 0.15,
    "history_penalty": 0.10,
    "risk": 0.10,
}
_GEOM_SUM = sum(_GEOM_RATIO.values())


def weights_with_semantic(w_semantic: float) -> UtilityWeights:
    """给定语义权重，其余五项按固定比例缩放，保证总权重为 1。"""
    scale = (1.0 - w_semantic) / _GEOM_SUM
    return UtilityWeights(
        information_gain=_GEOM_RATIO["information_gain"] * scale,
        novelty=_GEOM_RATIO["novelty"] * scale,
        semantic=w_semantic,
        path_cost=_GEOM_RATIO["path_cost"] * scale,
        history_penalty=_GEOM_RATIO["history_penalty"] * scale,
        risk=_GEOM_RATIO["risk"] * scale,
    )


# ══════════════════════════════════════════════════════════════════
# 1. 演示
# ══════════════════════════════════════════════════════════════════


def _describe(ranked) -> str:
    return "  >  ".join(
        f"{c.frontier_id}(U={c.local_utility:.3f}, S={(c.semantic_utility or 0):.2f})"
        for c in ranked
    )


def run_demo() -> None:
    values = build_scene()
    explorer = FrontierExplorer(viewpoint_config=DEMO_VIEWPOINT)
    snapshot = explorer.detect(values, RES, (0.0, 0.0), ROBOT, 1)

    base = explorer.rank(snapshot)
    f2 = next(c for c in base if c.frontier_id == "F2")
    f2_xy = (f2.safe_viewpoint[0], f2.safe_viewpoint[2])

    provider = ObjectPriorProvider()
    print("场景：合成房间，左右两个前沿区域，几何上 F1 优于 F2")
    print(f"F2 观察点 = ({f2_xy[0]:.2f}, {f2_xy[1]:.2f})\n")

    scenarios: List[Tuple[str, Sequence[MemoryObject], str]] = [
        ("① 记忆里没有物体（纯几何 FBE）", [], "chair"),
        ("② F2 旁看见一把椅子（目标=椅子）",
         [MemoryObject("O1", "chair", f2_xy, reliability=1.0, status="CONFIRMED")], "chair"),
        ("③ F2 旁只看见一张床（目标=椅子）",
         [MemoryObject("O1", "bed", f2_xy, reliability=1.0, status="CONFIRMED")], "chair"),
        ("④ F2 旁的椅子只是低置信候选",
         [MemoryObject("O1", "chair", f2_xy, reliability=0.25, status="CANDIDATE")], "chair"),
        ("⑤ F2 旁有餐桌（椅子常在餐桌旁，目标=椅子）",
         [MemoryObject("O1", "dining_table", f2_xy, reliability=1.0)], "chair"),
    ]

    for title, objects, goal in scenarios:
        ranked = explorer.rank(
            snapshot, goal=goal, semantic=provider,
            object_query=euclidean_object_query(objects))
        print(title)
        print("    " + _describe(ranked))
        print()


# ══════════════════════════════════════════════════════════════════
# 2. 消融
# ══════════════════════════════════════════════════════════════════

_DIRECTIONS = [(math.cos(a), math.sin(a))
               for a in [k * math.pi / 4 for k in range(8)]]


def _build_cases(explorer: FrontierExplorer, snapshot) -> List[Dict]:
    """构造"两个前沿、其中一个是正确的"的判别用例。"""
    base = explorer.rank(snapshot)
    vp = {c.frontier_id: (c.safe_viewpoint[0], c.safe_viewpoint[2]) for c in base}
    cases: List[Dict] = []

    for correct in ("F1", "F2"):
        wrong = "F2" if correct == "F1" else "F1"
        (cx, cy), (wx, wy) = vp[correct], vp[wrong]
        # 沿"两个观察点连线"的垂直方向放置物体：保证物体离正确前沿更近，
        # 否则 2~3 m 的偏移可能让物体反而靠近另一个前沿，用例本身就不成立。
        ux, uy = cx - wx, cy - wy
        norm = math.hypot(ux, uy) or 1.0
        px, py = -uy / norm, ux / norm

        for dist in (0.0, 0.5, 1.0, 1.5):
            for reliability in (1.0, 0.7, 0.25):
                for distractor in (False, True):
                    objects = [MemoryObject(
                        "O_target", "chair", (cx + px * dist, cy + py * dist),
                        reliability=reliability, status="CONFIRMED")]
                    if distractor:
                        objects.append(MemoryObject(
                            "O_other", "bed", (wx, wy), reliability=1.0))
                    cases.append(dict(correct=correct, objects=objects))
    return cases


def run_ablation() -> None:
    values = build_scene()
    explorer = FrontierExplorer(viewpoint_config=DEMO_VIEWPOINT)
    snapshot = explorer.detect(values, RES, (0.0, 0.0), ROBOT, 1)
    cases = _build_cases(explorer, snapshot)

    print("=" * 68)
    print(f"双前沿判别消融：{len(cases)} 个用例（正确前沿 × 物体距离 × 可靠性 × 干扰物）")
    print("=" * 68)
    print("\nTop-1 准确率（选中“正确”前沿的比例）\n")
    sigmas = (1.0, 2.0, 4.0)
    header = "  w_S │" + "".join(f"   σ={s:.1f}m " for s in sigmas)
    print(header)
    print("──────┼" + "──────────" * len(sigmas))

    for w_s in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        weights = weights_with_semantic(w_s)
        expl = FrontierExplorer(weights=weights, viewpoint_config=DEMO_VIEWPOINT)
        row = f"  {w_s:.2f} │"
        for sigma in sigmas:
            provider = ObjectPriorProvider(config=ObjectPriorConfig(sigma=sigma))
            hit = 0
            for case in cases:
                ranked = expl.rank(
                    snapshot, goal="chair", semantic=provider,
                    object_query=euclidean_object_query(case["objects"]))
                if ranked and ranked[0].frontier_id == case["correct"]:
                    hit += 1
            row += f"  {hit / len(cases) * 100:6.1f}%  "
        print(row)

    print("\n注：w_S = 0 即纯几何 FBE 基线，此时 σ 无意义。")
    print("    合成用例只验证机制，不代表真实场景的成功率。")

    # 几何与语义冲突时，需要多大的 w_S 才能翻转
    print("\n" + "=" * 68)
    print("几何与语义冲突时，翻转排序所需的 w_S 下限")
    print("=" * 68)
    base = explorer.rank(snapshot)
    gap = base[0].local_utility - base[1].local_utility
    print(f"\n两个前沿的几何效用差 ΔU_geo = {gap:.4f}")
    print("若正确前沿的 S=1、错误前沿的 S=0，翻转条件为 w_S > ΔU_geo·scale\n")
    for w_s in (0.10, 0.20, 0.30, 0.40, 0.50):
        scale = (1 - w_s) / _GEOM_SUM
        new_gap = gap * scale          # 几何差距随权重重分配等比缩小
        verdict = "可翻转" if w_s > new_gap else "不足"
        print(f"  w_S={w_s:.2f}: 几何差缩为 {new_gap:.4f}，"
              f"语义可贡献 {w_s:.2f} → {verdict}")


if __name__ == "__main__":
    run_demo()
    run_ablation()
