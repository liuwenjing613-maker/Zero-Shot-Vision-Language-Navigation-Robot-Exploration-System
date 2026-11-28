"""离线常识知识库：物体—房间先验 + 物体—物体共现。

设计依据（见 docs/模块三_语义扩展设计.md）：

  · 不在线调用大模型做语义推理 —— 延迟、抖动、重复计算都不可接受
  · 常识以 JSON 形式离线固化，运行时只做查表和几次乘加
  · 物体关联 A(o,g) 取两条路径的最大值：
      路径 1（房间）：A = Σ_r P(r|o)·P(r|g)
      路径 2（共现）：A = cooccurrence(o, g)
    房间路径解决"马桶↔洗手池"，共现路径解决"遥控器↔电视"这类
    房间粒度太粗的情况。

知识库文件放在 `code/knowledge/`，与代码一起版本管理。
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def normalize_category(name: str) -> str:
    """统一类别名的写法：小写、全角转半角、空格/连字符转下划线。"""
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()
    for ch in (" ", "-", "/", "\\"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


@dataclass(frozen=True)
class KnowledgeBase:
    """不可变知识库。`affinity()` 是唯一对外的语义查询入口。"""

    room_prior: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cooccurrence: Dict[Tuple[str, str], float] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    rooms: Tuple[str, ...] = ()

    # ── 构造 ──────────────────────────────────────────────
    @classmethod
    def empty(cls) -> "KnowledgeBase":
        return cls()

    @classmethod
    def load(cls, kb_dir: Optional[Path] = None) -> "KnowledgeBase":
        root = Path(kb_dir) if kb_dir is not None else KB_DIR
        prior_doc = json.loads((root / "object_room_prior.json").read_text(encoding="utf-8"))
        cooc_doc = json.loads((root / "object_cooccurrence.json").read_text(encoding="utf-8"))

        floor = float(prior_doc.get("floor", 0.02))
        rooms = tuple(prior_doc.get("default_rooms", ()))
        aliases = {normalize_category(k): normalize_category(v)
                   for k, v in prior_doc.get("aliases", {}).items()}

        room_prior: Dict[str, Dict[str, float]] = {}
        for obj, spec in prior_doc.get("objects", {}).items():
            key = normalize_category(obj)
            row = {normalize_category(r): float(v) for r, v in spec.items()}
            # 未列出的房间补 floor，避免"某个房间完全不可能"这种过强先验
            for r in rooms:
                row.setdefault(r, floor)
            total = sum(row.values())
            if total <= 0:
                continue
            room_prior[key] = {r: v / total for r, v in row.items()}

        cooc: Dict[Tuple[str, str], float] = {}
        for a, partners in cooc_doc.get("pairs", {}).items():
            ka = normalize_category(a)
            for b, w in partners.items():
                kb_ = normalize_category(b)
                if ka == kb_:
                    continue
                cooc[(ka, kb_)] = max(cooc.get((ka, kb_), 0.0), float(w))
                cooc[(kb_, ka)] = max(cooc.get((kb_, ka), 0.0), float(w))

        return cls(room_prior=room_prior, cooccurrence=cooc,
                   aliases=aliases, rooms=rooms)

    # ── 查询 ──────────────────────────────────────────────
    def canonical(self, name: str) -> str:
        key = normalize_category(name)
        return self.aliases.get(key, key)

    def room_distribution(self, category: str) -> Dict[str, float]:
        """P(r | object)。未知类别返回空字典（即"没有先验"）。"""
        return self.room_prior.get(self.canonical(category), {})

    def affinity(self, obj_category: str, goal_category: str) -> float:
        """A(o, g) ∈ [0, 1]：物体 o 对目标 g 的语义关联强度。

        同类直接取 1；否则取房间分布点积与共现先验的较大者。
        未知类别一律返回 0（宁可不给先验，也不要瞎猜）。
        """
        a = self.canonical(obj_category)
        b = self.canonical(goal_category)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        pa = self.room_prior.get(a)
        pb = self.room_prior.get(b)
        if pa and pb:
            room_score = sum(p * pb[r] for r, p in pa.items() if r in pb)
        else:
            room_score = 0.0

        cooc_score = self.cooccurrence.get((a, b), 0.0)
        return min(1.0, max(room_score, cooc_score))

    def is_known(self, category: str) -> bool:
        key = self.canonical(category)
        return key in self.room_prior


_DEFAULT_KB: Optional[KnowledgeBase] = None


def default_knowledge() -> KnowledgeBase:
    """进程内只加载一次的知识库。加载失败时退化为空库（语义分恒为 0）。"""
    global _DEFAULT_KB
    if _DEFAULT_KB is None:
        try:
            _DEFAULT_KB = KnowledgeBase.load()
        except (OSError, ValueError):
            _DEFAULT_KB = KnowledgeBase.empty()
    return _DEFAULT_KB
