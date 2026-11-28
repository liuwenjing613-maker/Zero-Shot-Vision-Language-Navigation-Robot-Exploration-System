"""语义探索模块（模块三）。

    gridops.py   栅格运算：掩码 / 距离变换 / 测地距离 / 射线扫描
    detector.py  Frontier Cell 检测 + 区域聚类
    viewpoint.py Viewpoint 生成与评分
    scorer.py    本地 Utility 打分与排序
    explorer.py  对外入口 FrontierExplorer

与队长的 Memory 模块分工：
    Exploration（这里）：检测 / 聚类 / Viewpoint / 打分
    Memory（vln_memory）：区域身份 / 历史状态 / 访问与失败计数
"""

from .detector import FrontierRegion, cluster_frontier_cells, detect_frontier_cells, detect_regions
from .explorer import FrontierExplorer
from .knowledge import KnowledgeBase, default_knowledge
from .scorer import UtilityWeights, score_candidates
from .semantic import (
    CandidateContext,
    CompositeProvider,
    MemoryObject,
    NullProvider,
    ObjectPriorConfig,
    ObjectPriorProvider,
    build_candidate_contexts,
    euclidean_object_query,
    reliability_from_status,
)
from .viewpoint import Viewpoint, ViewpointConfig, best_viewpoint, generate_viewpoints

__all__ = [
    "FrontierExplorer",
    "FrontierRegion",
    "Viewpoint",
    "ViewpointConfig",
    "UtilityWeights",
    "detect_frontier_cells",
    "cluster_frontier_cells",
    "detect_regions",
    "generate_viewpoints",
    "best_viewpoint",
    "score_candidates",
    # ── 语义扩展 ──
    "KnowledgeBase",
    "default_knowledge",
    "MemoryObject",
    "CandidateContext",
    "NullProvider",
    "ObjectPriorProvider",
    "ObjectPriorConfig",
    "CompositeProvider",
    "euclidean_object_query",
    "reliability_from_status",
    "build_candidate_contexts",
]
