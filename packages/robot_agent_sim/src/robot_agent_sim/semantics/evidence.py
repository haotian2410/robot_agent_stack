"""Deterministic facts extracted from user language."""
from __future__ import annotations
from dataclasses import dataclass
import re

_NUMBERS = {"一个": 1, "一": 1, "两个": 2, "两": 2, "二": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_OBJECT_WORDS = ("苹果", "香蕉", "棒球", "球", "方块", "盒子", "篮子", "apple", "banana", "baseball", "ball", "box", "basket")

@dataclass(frozen=True)
class ExplicitQuantityEvidence:
    phrase: str
    semantic_name: str | None
    count: int
    explicit_all: bool
    start: int
    end: int

@dataclass(frozen=True)
class ExplicitAssignmentEvidence:
    pairwise: bool
    token: str
    start: int
    end: int

@dataclass(frozen=True)
class ExplicitSemanticEvidence:
    quantities: list[ExplicitQuantityEvidence]
    assignments: list[ExplicitAssignmentEvidence]

def extract_semantic_evidence(text: str) -> ExplicitSemanticEvidence:
    words = "|".join(map(re.escape, sorted(_OBJECT_WORDS, key=len, reverse=True)))
    number = r"(一个|两个|三|四|五|六|七|八|九|十|[1-9][0-9]*)"
    quantities = []
    for match in re.finditer(rf"(?P<num>{number})\s*(?P<object>{words})", text):
        token = match.group("num")
        count = _NUMBERS.get(token, int(token) if token.isdigit() else 1)
        quantities.append(ExplicitQuantityEvidence(
            match.group(0), match.group("object"), count,
            not bool(re.search(r"中|其中", text[match.end():match.end() + 8])),
            match.start(), match.end(),
        ))
    assignments = [ExplicitAssignmentEvidence(True, token, m.start(), m.end())
                   for token in ("分别", "各自", "一一对应", "respectively")
                   for m in re.finditer(re.escape(token), text, re.IGNORECASE)]
    return ExplicitSemanticEvidence(quantities, assignments)
