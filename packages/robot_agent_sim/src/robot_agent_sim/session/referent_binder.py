"""Bind explicitly marked dialogue entities to stable scene object IDs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..models.task_understanding import TaskParseLLMOutput


class DialogueBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_ids: list[str]
    semantic_label: str
    plural: bool = False

    @property
    def object_id(self) -> str:
        if len(self.object_ids) != 1:
            raise ValueError("dialogue_binding_is_plural")
        return self.object_ids[0]


class ReferentBinder:
    """Turn parser-owned entity markers into grounding constraints.

    The binder never infers an operation role from words such as ``放到它``.
    The parser expresses the role through its normal entity references, so
    the same mechanism covers operation roles and spatial-relation references.
    """

    @staticmethod
    def bind(parsed: TaskParseLLMOutput, binding: DialogueBinding | None) -> dict[str, list[str]]:
        marked = [entity.id for entity in parsed.entities if entity.dialogue_ref or entity.dialogue_ref_set]
        if binding is None:
            if marked:
                raise ValueError("dialogue_binding_invalid: marked entity without dialogue referent")
            return {}
        if len(marked) != 1:
            raise ValueError(
                "dialogue_binding_unresolved: expected exactly one dialogue_ref entity, "
                f"got {marked}"
            )
        marked_entity = next(entity for entity in parsed.entities if entity.id == marked[0])
        if binding.plural != bool(marked_entity.dialogue_ref_set):
            raise ValueError("dialogue_binding_number_mismatch")
        return {marked[0]: list(binding.object_ids)}


__all__ = ["DialogueBinding", "ReferentBinder"]
