"""Compact, Python-owned semantic facts for the optional skill planner."""
from __future__ import annotations

from ..contracts.grounded_task import GroundedTask


def build_semantic_summary(task: GroundedTask) -> str:
    entities = {entity.entity_id: entity.semantic_name for entity in task.entities}
    lines = [f"用户要求：{task.instruction}"]
    for operation in task.operations:
        source = entities.get(operation.source, operation.source)
        destination = entities.get(operation.destination, operation.destination)
        if operation.task_type.value == "pick_and_place":
            placement = operation.placement_target
            placement_text = ""
            if placement is not None:
                reference = entities.get(placement.reference, placement.reference)
                placement_text = f"；placement={placement.kind.value}/{placement.relation.value if placement.relation else 'none'}"
                if reference:
                    placement_text += f"；reference={reference}"
            lines.append(f"{operation.operation_id}: pick_and_place；source={source}；destination={destination}{placement_text}。")
        else:
            roles = ", ".join(f"{role}={entities.get(getattr(operation, role), getattr(operation, role))}" for role in ("source", "destination", "target", "reference") if getattr(operation, role))
            lines.append(f"{operation.operation_id}: {operation.task_type.value}；{roles}。")
    return "\n".join(lines)[:1000]


__all__ = ["build_semantic_summary"]
