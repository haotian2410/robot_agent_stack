"""Bind explicitly marked dialogue entities to stable scene object IDs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..models.task_understanding import TaskParseLLMOutput


class DialogueBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    semantic_label: str


class ReferentBinder:
    """Turn parser-owned entity markers into grounding constraints.

    The binder never infers an operation role from words such as ``放到它``.
    The parser expresses the role through its normal entity references, so
    the same mechanism covers operation roles and spatial-relation references.
    """

    @staticmethod
    def bind(parsed: TaskParseLLMOutput, binding: DialogueBinding | None) -> dict[str, str]:
        marked = [entity.id for entity in parsed.entities if entity.dialogue_ref]
        if binding is None:
            if marked:
                raise ValueError("dialogue_binding_invalid: marked entity without dialogue referent")
            return {}
        if len(marked) != 1:
            raise ValueError(
                "dialogue_binding_unresolved: expected exactly one dialogue_ref entity, "
                f"got {marked}"
            )
        return {marked[0]: binding.object_id}


__all__ = ["DialogueBinding", "ReferentBinder"]
