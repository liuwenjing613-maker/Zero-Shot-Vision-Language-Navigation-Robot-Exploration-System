"""S1：双前沿判别评测（Frontier Choice Benchmark）。

回答一个问题：**当两个前沿里只有一个是"通向目标"的，打分函数能不能选对？**

与端到端 SR/SPL 不同，这个评测把问题限制在一个决策瞬间，因此：
  · 不需要跑完整回合，比跑 benchmark 便宜一到两个数量级
  · 指标直接反映打分函数本身，不会被感知错误、路径规划失败等下游问题稀释
  · 可以做"纯几何 vs 加语义"的严格消融

────────────────────────────────────────────────────────────────────
一、数据从哪来
────────────────────────────────────────────────────────────────────

每个"决策点"是一个 JSON 声明 + 一个 .npy 栅格：

    manifest.jsonl 每一行：
    {
      "dp_id": "2azQ1b91cZZ-0007",
      "scene": "2azQ1b91cZZ",
      "goal_category": "chair",
      "goal_xy": [3.42, -1.07],          # 目标实例的世界坐标（真值）
      "resolution": 0.1,
      "origin": [-2.0, -3.5],
      "robot_cell": [42, 31],
      "map_version": 7,
      "grid": "grids/dp0007.npy",
      "objects": [                        # 可选：当时 M2 里已观测到的物体
        {"object_id": "O000009", "category": "chair",
         "xy": [3.1, -0.8], "status": "CANDIDATE"}
      ]
    }

在回合脚本里用 `dump_decision_point()` 直接写出来即可（见文件末尾）。

**真值来源**：`goal_xy` 应当取自 Matterport3D 的语义标注（.house 的 objects 段，
或 Habitat 的 SemanticScene），而不是算法的输出。⚠️ 真值只能用于判分，
绝对不能喂给打分函数——否则就是特权信息泄漏。

────────────────────────────────────────────────────────────────────
二、正确答案怎么定
────────────────────────────────────────────────────────────────────

在同一个决策点的若干个前沿候选里，**代表观察点离目标最近的**那个就是正确前沿
（若多个候选与最近者的距离差在 --tie_tolerance 内，则都算对）。

理由：前沿的作用是"朝哪个方向继续走"，离目标最近的那条边界显然是应当优先去的。
若最近者距离超过 --max_range，说明目标远在本轮视野之外，该决策点无法判定，直接丢弃。

────────────────────────────────────────────────────────────────────
三、难点子集
────────────────────────────────────────────────────────────────────

`--hard_only` 只保留"纯几何选错"的决策点。此时几何基线准确率恒为 0，
任何提升都直接归因于语义层——这是最有说服力的一组数字。

────────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────────

    # 本地冒烟测试（合成数据，不需要 MP3D / Habitat）
    PYTHONPATH=code python evals/eval_frontier_choice.py --demo 40

    # 服务器上跑真实数据
    PYTHONPATH=code python evals/eval_frontier_choice.py \
        --manifest /root/autodl-tmp/data/frontier_choice/manifest.jsonl \
        --hard_only --out /root/autodl-tmp/data/frontier_choice/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from exploration import FrontierExplorer
from exploration.semantic import (
    MemoryObject,
    ObjectPriorConfig,
    ObjectPriorProvider,
    euclidean_object_query,
    reliability_from_status,
)
from exploration.scorer import UtilityWeights

# 其余五项权重的相对比例，与 scorer.UtilityWeights 的默认值一致（和为 0.70）
_GEOM_RATIO = {
    "information_gain": 0.25,
    "novelty": 0.10,
    "path_cost": 0.15,
    "history_penalty": 0.10,
    "risk": 0.10,
}
_GEOM_SUM = sum(_GEOM_RATIO.values())


def weights_with_semantic(w_semantic: float) -> UtilityWeights:
    """给定语义权重，其余五项按固定比例缩放，总权重保持为 1。"""
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
# 数据结构
# ══════════════════════════════════════════════════════════════════


@dataclass
class DecisionPoint:
    dp_id: str
    goal_category: str
    goal_xy: Tuple[float, float]
    resolution: float
    origin: Tuple[float, float]
    robot_cell: Tuple[int, int]
    map_version: int
    grid: np.ndarray
    scene: str = ""
    objects: List[MemoryObject] = field(default_factory=list)


def load_manifest(path: Path) -> List[DecisionPoint]:
    """读取 manifest.jsonl；`grid` 支持相对 manifest 所在目录的路径。"""
    points: List[DecisionPoint] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            grid_path = Path(rec["grid"])
            if not grid_path.is_absolute():
                grid_path = path.parent / grid_path
            values = np.load(grid_path)
            objects = [
                MemoryObject(
                    object_id=str(o.get("object_id", idx)),
                    category=str(o["category"]),
                    xy=(float(o["xy"][0]), float(o["xy"][1])),
                    reliability=float(o.get(
                        "reliability",
                        reliability_from_status(o.get("status", "CANDIDATE"),
                                                bool(o.get("localized", True))))),
                    status=str(o.get("status", "CANDIDATE")),
                )
                for idx, o in enumerate(rec.get("objects", []))
            ]
            points.append(DecisionPoint(
                dp_id=str(rec["dp_id"]),
                goal_category=str(rec["goal_category"]),
                goal_xy=(float(rec["goal_xy"][0]), float(rec["goal_xy"][1])),
                resolution=float(rec["resolution"]),
                origin=(float(rec["origin"][0]), float(rec["origin"][1])),
                robot_cell=(int(rec["robot_cell"][0]), int(rec["robot_cell"][1])),
                map_version=int(rec.get("map_version", 1)),
                grid=values,
                scene=str(rec.get("scene", "")),
                objects=objects,
            ))
    return points


# ══════════════════════════════════════════════════════════════════
# 真值
# ══════════════════════════════════════════════════════════════════


def ground_truth(candidates, goal_xy: Tuple[float, float],
                 max_range: float, tie_tolerance: float) -> Optional[set]:
    """返回"正确前沿"的 id 集合；若目标太远无法判定则返回 None。"""
    if len(candidates) < 2:
        return None

    dists = {}
    for cand in candidates:
        vx, vz = cand.safe_viewpoint[0], cand.safe_viewpoint[2]
        dists[cand.frontier_id] = math.hypot(vx - goal_xy[0], vz - goal_xy[1])

    nearest = min(dists.values())
    if nearest > max_range:
        return None
    return {fid for fid, d in dists.items() if d <= nearest + tie_tolerance}


# ══════════════════════════════════════════════════════════════════
# 评测
# ══════════════════════════════════════════════════════════════════


@dataclass
class Prepared:
    """一个决策点预计算好快照与真值，后续只变打分权重。"""

    dp: DecisionPoint
    snapshot: Dict
    truth: set
    geometric_top1: str
    geometric_rank: int


@dataclass
class PointResult:
    dp_id: str
    n_candidates: int
    truth: List[str]
    geometric_top1: str
    geometric_hit: bool
    geometric_rank: int


def rank_of(ranked, truth: set) -> int:
    for i, cand in enumerate(ranked, start=1):
        if cand.frontier_id in truth:
            return i
    return 0


def prepare(points: Sequence[DecisionPoint], max_range: float,
            tie_tolerance: float) -> List[Prepared]:
    """每个点只跑一次 detect()，后续所有权重配置复用同一份快照。"""
    prepared: List[Prepared] = []
    explorer = FrontierExplorer(weights=weights_with_semantic(0.0))
    for dp in points:
        snapshot = explorer.detect(dp.grid, dp.resolution, dp.origin,
                                   dp.robot_cell, dp.map_version)
        ranked = explorer.rank(snapshot)
        truth = ground_truth(ranked, dp.goal_xy, max_range, tie_tolerance)
        if truth is None:
            continue
        prepared.append(Prepared(
            dp=dp, snapshot=snapshot, truth=truth,
            geometric_top1=ranked[0].frontier_id,
            geometric_rank=rank_of(ranked, truth),
        ))
    return prepared


def evaluate(prepared: Sequence[Prepared], w_semantic: float, sigma: float
             ) -> Tuple[float, float]:
    """在给定权重下返回 (Top-1 准确率, MRR)。"""
    if not prepared:
        return 0.0, 0.0

    explorer = FrontierExplorer(weights=weights_with_semantic(w_semantic))
    provider = ObjectPriorProvider(config=ObjectPriorConfig(sigma=sigma))

    hits: List[bool] = []
    ranks: List[int] = []
    for item in prepared:
        if w_semantic <= 0.0:
            ranked = explorer.rank(item.snapshot)
        else:
            ranked = explorer.rank(
                item.snapshot, goal=item.dp.goal_category, semantic=provider,
                object_query=euclidean_object_query(item.dp.objects))
        hits.append(ranked[0].frontier_id in item.truth)
        ranks.append(rank_of(ranked, item.truth))

    top1 = sum(hits) / len(hits)
    mrr = sum(1.0 / r for r in ranks if r > 0) / len(ranks)
    return top1, mrr


def run(points: Sequence[DecisionPoint], hard_only: bool,
        max_range: float = 10.0, tie_tolerance: float = 0.5) -> Dict:
    prepared = prepare(points, max_range, tie_tolerance)
    if hard_only:
        prepared = [p for p in prepared if p.geometric_top1 not in p.truth]

    base_top1, base_mrr = evaluate(prepared, 0.0, 1.0)

    table = []
    for w_s in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        for sigma in (1.0, 2.0, 4.0):
            if w_s == 0.0 and sigma != 1.0:
                continue            # w_S=0 时 σ 无意义，只跑一次
            top1, mrr = evaluate(prepared, w_s, sigma)
            table.append(dict(w_semantic=w_s, sigma=sigma, n=len(prepared),
                              top1=top1, mrr=mrr))

    results = [
        PointResult(
            dp_id=p.dp.dp_id,
            n_candidates=len(p.snapshot.get("candidates", [])),
            truth=sorted(p.truth),
            geometric_top1=p.geometric_top1,
            geometric_hit=p.geometric_top1 in p.truth,
            geometric_rank=p.geometric_rank,
        )
        for p in prepared
    ]

    return {
        "n_points": len(prepared),
        "hard_only": hard_only,
        "max_range": max_range,
        "baseline_top1": base_top1,
        "baseline_mrr": base_mrr,
        "table": table,
        "points": [r.__dict__ for r in results],
    }


# ══════════════════════════════════════════════════════════════════
# 写入决策点（在回合脚本里调用）
# ══════════════════════════════════════════════════════════════════


def dump_decision_point(out_dir: Path, values: np.ndarray,
                        robot_cell: Tuple[int, int], resolution: float,
                        origin: Tuple[float, float], map_version: int,
                        goal_category: str, goal_xy: Tuple[float, float],
                        objects: Sequence[MemoryObject] = (),
                        scene: str = "", dp_id: Optional[str] = None) -> str:
    """把一个决策瞬间存成 manifest.jsonl + .npy，供本脚本离线评测。

    在 eval_tls_hyf_episode.py 的每轮决策前调用即可：

        from evals.eval_frontier_choice import dump_decision_point
        dump_decision_point(out_dir, occupancy.values, robot_cell=..., ...)
    """
    out_dir = Path(out_dir)
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)
    if dp_id is None:
        existing = len(list((out_dir / "grids").glob("dp*.npy")))
        dp_id = f"{scene or 'dp'}-{existing:04d}"

    grid_rel = Path("grids") / f"{dp_id}.npy"
    np.save(out_dir / grid_rel, np.asarray(values))

    record = {
        "dp_id": dp_id,
        "scene": scene,
        "goal_category": goal_category,
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "resolution": float(resolution),
        "origin": [float(origin[0]), float(origin[1])],
        "robot_cell": [int(robot_cell[0]), int(robot_cell[1])],
        "map_version": int(map_version),
        "grid": str(grid_rel).replace("\\", "/"),
        "objects": [
            {"object_id": o.object_id, "category": o.category,
             "xy": [float(o.xy[0]), float(o.xy[1])],
             "status": o.status, "reliability": float(o.reliability)}
            for o in objects
        ],
    }
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dp_id


# ══════════════════════════════════════════════════════════════════
# 合成演示（本地冒烟用）
# ══════════════════════════════════════════════════════════════════


def make_demo_points(n: int, seed: int = 0) -> List[DecisionPoint]:
    """用合成场景生成决策点，仅用于验证脚本本身能跑通。

    真值（goal_xy）被放在某个前沿区的后方，交替选左右两侧，
    并随机给 0~2 个"已观测物体"（其中一半是目标类别）。
    """
    from exploration.demo_exploration import build_scene

    rng = np.random.default_rng(seed)
    values = build_scene()
    explorer = FrontierExplorer()
    snap = explorer.detect(values, 0.1, (0.0, 0.0), (55, 30), 1)
    base = explorer.rank(snap)

    # 机器人世界坐标（栅格 (55,30)、分辨率 0.1、原点 0）
    robot_world = ((30 + 0.5) * 0.1, (55 + 0.5) * 0.1)

    points: List[DecisionPoint] = []
    for i in range(n):
        target = base[i % len(base)]
        cx, cz = target.centroid[0], target.centroid[2]
        # 沿"机器人 → 区域质心"的方向把目标放到质心之外，保证它在该前沿的后方
        ux, uz = cx - robot_world[0], cz - robot_world[1]
        norm = math.hypot(ux, uz) or 1.0
        dist = 1.5 + 1.5 * float(rng.random())
        goal = (cx + ux / norm * dist, cz + uz / norm * dist)

        # 目标物体：多数用例高可靠，少数用例模拟"只瞥见一眼"
        objects: List[MemoryObject] = [
            MemoryObject(
                "O1", "chair",
                (goal[0] - 0.3, goal[1] + 0.2),
                reliability=float(rng.choice([1.0, 1.0, 0.7, 0.25])),
                status="CANDIDATE")
        ]
        if rng.random() < 0.4:
            other = base[(i + 1) % len(base)]
            objects.append(MemoryObject(
                "O2", "bed",
                (other.centroid[0], other.centroid[2]),
                reliability=1.0, status="CONFIRMED"))

        points.append(DecisionPoint(
            dp_id=f"demo-{i:04d}", goal_category="chair", goal_xy=goal,
            resolution=0.1, origin=(0.0, 0.0), robot_cell=(55, 30),
            map_version=1, grid=values, scene="demo", objects=objects))
    return points


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="双前沿判别评测")
    parser.add_argument("--manifest", type=Path, help="manifest.jsonl 路径")
    parser.add_argument("--demo", type=int, default=0, help="用合成数据跑 N 个决策点")
    parser.add_argument("--hard_only", action="store_true",
                        help="只保留纯几何选错的决策点")
    parser.add_argument("--max_range", type=float, default=10.0,
                        help="目标距最近观察点超过此值时丢弃该决策点（米）")
    parser.add_argument("--out", type=Path, help="把结果写成 JSON")
    args = parser.parse_args(argv)

    if args.manifest:
        points = load_manifest(args.manifest)
    elif args.demo:
        points = make_demo_points(args.demo)
    else:
        parser.print_help()
        return 2

    result = run(points, hard_only=args.hard_only, max_range=args.max_range)

    print(f"决策点：{result['n_points']} 个"
          f"（{'仅纯几何选错的难点' if args.hard_only else '全部'}）")
    print(f"几何基线 Top-1 = {result['baseline_top1'] * 100:.1f}%   "
          f"MRR = {result['baseline_mrr']:.3f}\n")

    header = f"{'w_S':>5} {'σ/m':>5} {'Top-1':>8} {'MRR':>8} {'n':>5}"
    print(header)
    print("-" * len(header))
    for row in result["table"]:
        print(f"{row['w_semantic']:>5.2f} {row['sigma']:>5.1f} "
              f"{row['top1'] * 100:>7.1f}% {row['mrr']:>8.3f} {row['n']:>5}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
