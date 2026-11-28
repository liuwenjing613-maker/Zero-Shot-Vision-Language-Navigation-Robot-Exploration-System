"""前沿候选的本地 Utility 打分与排序。

公式：

    U(F) = w_IG·IG + w_N·N + w_S·S − w_D·D − w_H·H − w_R·R

     IG 信息增益（可见未知格，比值归一化）
     N  新颖性（1/(1+访问次数)）
     S  语义收益（由 semantic.SemanticProvider 提供，见 semantic.py）
     D  路径成本（测地距离，比值归一化）
     H  历史惩罚（失败占比）
     R  风险（离障碍太近 / 不可达）

S 由语义层注入；未接线时 provider 为 NullProvider，S 恒为 0，
排序结果与纯几何 FBE 完全一致 —— 语义是加分项，不是必需品。
所有权重、归一化后的指标都在 [0,1]，权重通过实验调（见 S2 消融）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from common.schemas import FrontierCandidate
from .gridops import ratio


@dataclass
class UtilityWeights:
    """默认权重。六项之和为 1.0，全部作用于 [0,1] 归一化后的分量。

    相比纯几何版本，语义项拿走 0.30，几何/历史各项按比例压缩：
    语义足够强时可以翻转排序，但无法单独决定结果（S=1 时贡献 0.30，
    仍小于"信息增益 0.25 + 新颖性 0.10 + 路径 0.15"的合计 0.50）。
    """

    information_gain: float = 0.25
    novelty: float = 0.10
    semantic: float = 0.30
    path_cost: float = 0.15
    history_penalty: float = 0.10
    risk: float = 0.10


def novelty(visit_count: int) -> float:
    """越少被访问越新：0 次 → 1.0，1 次 → 0.5，2 次 → 0.33 …"""
    return 1.0 / (1.0 + max(0, int(visit_count)))


def history_penalty(failed_count: int, attempt_count: int) -> float:
    """失败占比。从没尝试过 → 0（不惩罚）。"""
    attempts = max(0, int(attempt_count))
    if attempts == 0:
        return 0.0
    return min(1.0, max(0, int(failed_count)) / attempts)


def score_candidates(
    candidates: Sequence[FrontierCandidate],
    weights: Optional[UtilityWeights] = None,
) -> List[FrontierCandidate]:
    """给候选打 local_utility 并按分数降序排列（原地修改并返回）。

    归一化在这一步统一做（跨所有候选），保证不同场景下分数可比。
    """
    w = weights or UtilityWeights()
    if not candidates:
        return []

    # 用比值归一化而不是 min-max：保留候选间的真实差异
    gains = ratio([c.information_gain for c in candidates])
    costs = ratio([c.path_length for c in candidates])

    for cand, g_n, d_n in zip(candidates, gains, costs):
        n = novelty(cand.visit_count)
        # 用「访问次数 + 失败次数」近似尝试次数：第一版 FrontierCandidate 里没有单独字段
        h = history_penalty(cand.failed_count, cand.visit_count + cand.failed_count)

        # 风险：不可达或没有安全余量 → 满风险；否则按 clearance 反推
        if not cand.reachable or cand.clearance <= 0:
            risk = 1.0
        else:
            risk = max(0.0, 1.0 - cand.clearance / 0.6)

        # 语义分由 semantic provider 预先写入；未接线时为 None，按 0 处理
        s = cand.semantic_utility if cand.semantic_utility is not None else 0.0
        s = min(1.0, max(0.0, float(s)))

        utility = (
            w.information_gain * g_n
            + w.novelty * n
            + w.semantic * s
            - w.path_cost * d_n
            - w.history_penalty * h
            - w.risk * risk
        )

        cand.information_gain = round(float(g_n), 4)   # 存归一化后的值
        cand.local_utility = round(float(utility), 4)

    candidates.sort(key=lambda c: (-c.local_utility, c.frontier_id))
    return list(candidates)
