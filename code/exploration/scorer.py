"""前沿候选的本地 Utility 打分与排序。

公式（队长设计文档 7.1 节）：

    U(F) = w1·IG − w2·D + w3·S − w4·H − w5·R

     IG 信息增益（可见未知格）
     D  路径成本（测地距离）
     S  语义收益（第一版不做，置 0，留接口）
     H  历史惩罚（失败/重复探索）
     R  风险（离障碍太近 / 不可达）

第一版五个分量除 S 外都实现；S 由后续 Qwen 语义重排序补上。
所有权重、归一化后的指标都在 [0,1]，权重通过实验调（文档 4.4 节）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from common.schemas import FrontierCandidate
from .gridops import ratio


@dataclass
class UtilityWeights:
    """默认权重。方案要求「通过 Habitat 实验调整」，这里先给一组保守初值。"""

    information_gain: float = 0.35
    novelty: float = 0.15
    path_cost: float = 0.20
    history_penalty: float = 0.15
    risk: float = 0.15


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

        utility = (
            w.information_gain * g_n
            + w.novelty * n
            - w.path_cost * d_n
            - w.history_penalty * h
            - w.risk * risk
        )

        cand.information_gain = round(float(g_n), 4)   # 存归一化后的值
        cand.local_utility = round(float(utility), 4)

    candidates.sort(key=lambda c: (-c.local_utility, c.frontier_id))
    return list(candidates)
