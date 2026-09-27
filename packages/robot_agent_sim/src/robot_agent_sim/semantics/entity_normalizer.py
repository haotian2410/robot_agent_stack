"""Python-owned semantic entity identity and duplicate-role checks.

The model may split one counted phrase into several identical entities.  This
module defines the comparison rule used by normalization so that quantity is
not mistaken for semantic identity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticEntityKey:
    canonical_name: str
    category: str | None = None
    color: str | None = None
    dialogue_ref: bool = False
    dialogue_ref_set: bool = False


def semantic_entity_key(entity) -> SemanticEntityKey:
    return SemanticEntityKey(
        canonical_name=str(entity.name if hasattr(entity, "name") else entity.semantic_name).casefold(),
        category=getattr(entity, "category", None),
        color=getattr(entity, "color", None),
        dialogue_ref=bool(getattr(entity, "dialogue_ref", False)),
        dialogue_ref_set=bool(getattr(entity, "dialogue_ref_set", False)),
    )


def same_semantic_role(left, right) -> bool:
    """Whether two model entities are duplicate decomposition of one role."""
    return semantic_entity_key(left) == semantic_entity_key(right)


__all__ = ["SemanticEntityKey", "semantic_entity_key", "same_semantic_role"]
