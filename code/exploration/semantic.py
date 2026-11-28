"""语义收益 S(F, g) 的可插拔实现。

设计原则：**语义只做本地确定性计算，不调用生成式大模型**。

    S(F,g) = max_{o ∈ O_F} [ A(o,g) · c_o · exp(−d_geo(o,F)/σ) ]

      O_F    与前沿 F 的观察点有空间关联的记忆物体（由 object_query 提供）
      A(o,g) 知识库中的物体关联得分（房间分布点积 / 共现先验，见 knowledge.py）
      c_o    该物体的观测可靠性（由 M2 状态映射而来）
      d_geo  物体到观察点的测地距离（由调用方提供；缺省用欧氏距离近似）
      σ      语义影响距离（米）

取 max 而不是求和：大量无关物体不应该稀释一条强证据。没有有效证据时 S = 0，
排序自动退化为原来的几何 FBE —— 语义是加分项，不是必需品。

历史调制：被访问过却没有发现目标的区域，语义分按 (1 − decay)^visit_count 衰减，
避免机器人因为"知识库说椅子常在餐厅"而反复回餐厅。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .knowledge import KnowledgeBase, default_knowledge

DEFAULT_SIGMA = 4.0          # 语义影响距离（米）
DEFAULT_QUERY_RADIUS = 8.0   # 只查询观察点附近的物体


# ══════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MemoryObject:
    """探索模块眼中的"记忆里的一个物体"（由 M2 适配而来，不依赖 M2 实现）。"""

    object_id: str
    category: str
    xy: Tuple[float, float]
    reliability: float = 0.7
    status: str = "CANDIDATE"


@dataclass(frozen=True)
class CandidateContext:
    """一个前沿候选在做语义评分时需要知道的全部信息。"""

    frontier_id: str
    viewpoint_xy: Tuple[float, float]
    centroid_xy: Tuple[float, float]
    visit_count: int = 0
    failed_count: int = 0
    status: str = "NEW"


# (观察点 xy, 查询半径) → [(物体, 测地距离), ...]
ObjectQuery = Callable[[Tuple[float, float], float], Sequence[Tuple[MemoryObject, float]]]


# ══════════════════════════════════════════════════════════════════
# 可靠性映射
# ══════════════════════════════════════════════════════════════════

_RELIABILITY_BY_STATUS = {
    "CONFIRMED": 1.0,
    "CANDIDATE": 0.7,
    "UNLOCATED": 0.25,
    "REJECTED": 0.0,
    "LOST": 0.0,
}


def reliability_from_status(status: str, has_position: bool = True) -> float:
    """把 M2 的对象状态映射成 c_o ∈ [0,1]。"""
    key = (status or "CANDIDATE").upper()
    if key == "CANDIDATE" and not has_position:
        key = "UNLOCATED"
    return _RELIABILITY_BY_STATUS.get(key, 0.3)


# ══════════════════════════════════════════════════════════════════
# Provider
# ══════════════════════════════════════════════════════════════════


class NullProvider:
    """不做任何语义打分 —— 行为与未接入语义时完全一致。"""

    name = "null"

    def score(self, goal: str, contexts: Sequence[CandidateContext],
              query: ObjectQuery) -> Dict[str, float]:
        return {c.frontier_id: 0.0 for c in contexts}


@dataclass(frozen=True)
class ObjectPriorConfig:
    sigma: float = DEFAULT_SIGMA
    query_radius: float = DEFAULT_QUERY_RADIUS
    history_decay: float = 0.6      # 每被访问一次，先验保留 (1 - decay)
    min_affinity: float = 0.10      # 关联太弱的物体直接忽略


class ObjectPriorProvider:
    """路径 A：用**已观测到的物体**把语义传播到附近前沿。

    只需要 M2 里已有的信息，不依赖任何视觉模型。
    局限：目标从未被任何帧检出过时，所有前沿的语义分都是 0，退化为几何 FBE。
    """

    name = "object_prior"

    def __init__(self, kb: Optional[KnowledgeBase] = None, *,
                 config: Optional[ObjectPriorConfig] = None) -> None:
        self.kb = kb or default_knowledge()
        self.config = config or ObjectPriorConfig()

    def score(self, goal: str, contexts: Sequence[CandidateContext],
              query: ObjectQuery) -> Dict[str, float]:
        cfg = self.config
        if not goal:
            return {c.frontier_id: 0.0 for c in contexts}

        scores: Dict[str, float] = {}
        for ctx in contexts:
            best = 0.0
            for obj, dist in query(ctx.viewpoint_xy, cfg.query_radius):
                if dist > cfg.query_radius:
                    continue
                affinity = self.kb.affinity(obj.category, goal)
                if affinity < cfg.min_affinity:
                    continue
                value = affinity * obj.reliability * math.exp(-dist / cfg.sigma)
                if value > best:
                    best = value

            # 历史调制：访问过但没找到 → 先验打折，避免反复回访
            if ctx.visit_count > 0:
                best *= (1.0 - cfg.history_decay) ** ctx.visit_count

            scores[ctx.frontier_id] = max(0.0, min(1.0, best))
        return scores


class CompositeProvider:
    """把多个 provider 的分数按 max 融合（后续接入路径 B 时用）。"""

    def __init__(self, providers: Sequence[object]) -> None:
        self.providers = list(providers)
        self.name = "+".join(getattr(p, "name", "?") for p in self.providers) or "composite"

    def score(self, goal: str, contexts: Sequence[CandidateContext],
              query: ObjectQuery) -> Dict[str, float]:
        out: Dict[str, float] = {c.frontier_id: 0.0 for c in contexts}
        for provider in self.providers:
            part = provider.score(goal, contexts, query)
            for key, value in part.items():
                if value > out.get(key, 0.0):
                    out[key] = value
        return out


# ══════════════════════════════════════════════════════════════════
# 便捷构造
# ══════════════════════════════════════════════════════════════════


def euclidean_object_query(objects: Sequence[MemoryObject]) -> ObjectQuery:
    """用欧氏距离做查询的简易实现（单元测试 / 演示用）。

    生产环境应当注入基于测地距离的查询：欧氏距离会穿过墙体，
    把隔壁房间的物体算成"很近"。见 explorer.rank 的 object_query 参数。
    """

    def query(xy: Tuple[float, float], radius: float) -> List[Tuple[MemoryObject, float]]:
        out: List[Tuple[MemoryObject, float]] = []
        for obj in objects:
            d = math.hypot(obj.xy[0] - xy[0], obj.xy[1] - xy[1])
            if d <= radius:
                out.append((obj, d))
        return out

    return query


def build_candidate_contexts(candidates: Sequence[object]) -> List[CandidateContext]:
    """从 FrontierCandidate 列表构造评分上下文（保持探索模块与 schemas 解耦）。"""
    contexts: List[CandidateContext] = []
    for cand in candidates:
        contexts.append(CandidateContext(
            frontier_id=cand.frontier_id,
            viewpoint_xy=(cand.safe_viewpoint[0], cand.safe_viewpoint[2]),
            centroid_xy=(cand.centroid[0], cand.centroid[2]),
            visit_count=int(getattr(cand, "visit_count", 0)),
            failed_count=int(getattr(cand, "failed_count", 0)),
            status=str(getattr(cand, "status", "NEW")),
        ))
    return contexts
