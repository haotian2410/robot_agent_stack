"""Project runtime interaction metadata into a compact planner world model."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..contracts.grounded_task import GroundedTask


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannerEntity(_StrictModel):
    id: str
    name: str
    category: str
    affordances: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()


class PlannerRoleBindings(_StrictModel):
    source: str | None = None
    destination: str | None = None
    target: str | None = None
    reference: str | None = None


class PlannerOperation(_StrictModel):
    id: str
    type: str
    semantic_intent: str = Field(min_length=1, max_length=300)
    role_bindings: PlannerRoleBindings = Field(default_factory=PlannerRoleBindings)
    valid_roles: tuple[str, ...] = ()
    source: str | None = None
    destination: str | None = None
    target: str | None = None
    reference: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    motion_direction: str | None = None
    distance_m: float | None = None
    placement_target: dict[str, Any] | None = None


class PlannerGoal(_StrictModel):
    relation: str
    subject: str | None = None
    reference: str | None = None


class PlannerInitialState(_StrictModel):
    held_entity: str | None = None


class PlannerContext(_StrictModel):
    instruction: str
    semantic_summary: str = Field(min_length=1, max_length=1000)
    operations: list[PlannerOperation]
    entities: list[PlannerEntity]
    goals: list[PlannerGoal] = Field(default_factory=list)
    initial_state: PlannerInitialState = Field(default_factory=PlannerInitialState)


def build_planner_context(
    task: GroundedTask,
    interaction_registry: str | Path | dict[str, Any] | None = None,
    initial_state: PlannerInitialState | None = None,
) -> PlannerContext:
    """Return only operation-relevant semantic facts, never physical metadata."""
    registry = _load_registry(interaction_registry)
    objects = registry.get("objects", {}) if registry else {}
    relevant = {
        value
        for operation in task.operations
        for value in (operation.source, operation.destination, operation.target, operation.reference)
        if value
    }
    grounded = {entity.entity_id: entity for entity in task.entities}
    object_to_entity = {entity.object_id: entity.entity_id for entity in task.entities}
    projected: dict[str, dict[str, Any]] = {}

    for entity_id in relevant:
        entity = grounded[entity_id]
        metadata = _metadata_for(objects, entity.object_id)
        category = entity.category or _infer_category(entity.semantic_name, entity.model_name, entity.object_id)
        affordances, regions = _semantic_capabilities(metadata, category)
        projected[entity_id] = {
            "id": entity_id,
            "name": entity.semantic_name,
            "category": category,
            "affordances": affordances,
            "regions": regions,
            "relations": [],
        }

    for relation in task.spatial_relations:
        if relation.subject in relevant and (relation.reference is None or relation.reference in relevant):
            value = relation.relation.value
            if relation.reference:
                value += f":{relation.reference}"
            projected[relation.subject]["relations"].append(value)

    # Runtime mechanism metadata states which contact object acts on which
    # articulated object.  Expose that as a semantic relationship only.
    for raw_object_id, metadata in objects.items():
        if not isinstance(metadata, dict):
            continue
        for contract in metadata.get("affordances", {}).values():
            if not isinstance(contract, dict):
                continue
            acting = object_to_entity.get(str(contract.get("acting_target")))
            contact = object_to_entity.get(str(contract.get("contact_target")))
            if acting in relevant and contact in relevant:
                projected[contact]["relations"].append(f"part_of:{acting}")

    operations = []
    for operation in task.operations:
        bindings = {
            role: getattr(operation, role)
            for role in ("source", "destination", "target", "reference")
        }
        valid_roles = tuple(role for role, value in bindings.items() if value is not None)
        source_name = grounded.get(operation.source).semantic_name if operation.source in grounded else operation.source
        destination_name = grounded.get(operation.destination).semantic_name if operation.destination in grounded else operation.destination
        target_name = grounded.get(operation.target).semantic_name if operation.target in grounded else operation.target
        if operation.task_type.value == "pick_and_place":
            semantic_intent = f"搬运 {source_name}，并将其放置到 {destination_name}。"
            if operation.placement_target is not None:
                placement = operation.placement_target
                reference_name = grounded.get(placement.reference).semantic_name if placement.reference in grounded else placement.reference
                semantic_intent = f"搬运 {source_name}，并将其放置到 {reference_name or destination_name} 附近。" if placement.relation and placement.relation.value == "near" else semantic_intent
        elif operation.task_type.value in {"open", "close"}:
            semantic_intent = f"对 {target_name} 执行 {operation.task_type.value} 操作。"
        else:
            semantic_intent = f"对 {target_name or source_name or destination_name} 执行 {operation.task_type.value} 操作。"
        operations.append(PlannerOperation(
            id=operation.operation_id,
            type=operation.task_type.value,
            semantic_intent=semantic_intent,
            role_bindings=PlannerRoleBindings(**bindings),
            valid_roles=valid_roles,
            source=operation.source,
            destination=operation.destination,
            target=operation.target,
            reference=operation.reference,
            depends_on=list(operation.depends_on),
            motion_direction=operation.motion_direction.value if operation.motion_direction else None,
            distance_m=operation.distance_m,
            placement_target=operation.placement_target.model_dump(mode="json") if operation.placement_target else None,
        ))
    goals = [PlannerGoal(
        relation=relation.relation.value,
        subject=relation.subject,
        reference=relation.reference,
    ) for relation in task.spatial_relations if relation.scope == "goal"]
    entities = [PlannerEntity(**{
        **projected[entity_id],
        "relations": tuple(dict.fromkeys(projected[entity_id]["relations"])),
    }) for entity_id in grounded if entity_id in projected]
    from .semantic_summary import build_semantic_summary
    return PlannerContext(
        instruction=task.instruction,
        semantic_summary=build_semantic_summary(task),
        operations=operations,
        entities=entities,
        goals=goals,
        initial_state=initial_state or PlannerInitialState(),
    )


def _load_registry(value: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    data = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("objects"), dict):
        raise ValueError("interaction registry must contain an objects mapping")
    return data


def _metadata_for(objects: dict[str, Any], object_id: str) -> dict[str, Any]:
    direct = objects.get(object_id)
    if isinstance(direct, dict):
        return direct
    return next((item for item in objects.values()
                 if isinstance(item, dict) and item.get("object_id") == object_id), {})


def _semantic_capabilities(metadata: dict[str, Any], category: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    action_names = set(metadata.get("action_requests", {}))
    action_names.update(metadata.get("default_interactions", {}))
    action_names.update(metadata.get("affordances", {}))
    anchors = set(metadata.get("spatial", {}).get("anchors", {}))
    affordances = set()
    if "grasp" in action_names or "grasp" in anchors:
        affordances.add("graspable")
    if "press" in action_names or "button_surface" in anchors:
        affordances.add("pressable")
    if "pull" in action_names:
        affordances.add("pullable")
    if "push" in action_names:
        affordances.add("pushable")
    if "interior" in anchors or category == "location":
        affordances.add("placeable")
    if "support_surface" in anchors or category == "support_surface":
        affordances.add("placeable")

    regions = set()
    if "grasp" in anchors or "graspable" in affordances:
        regions.add("grasp_region")
    if "interior" in anchors or "placeable" in affordances:
        regions.add("container_interior")
    if "button_surface" in anchors or "pressable" in affordances:
        regions.add("button_surface")
    if "support_surface" in anchors or category == "support_surface":
        regions.add("support_surface")

    # Visual-grounding Route B may have no authored sidecar.  Use only the
    # grounded semantic category as a conservative fallback.
    if not metadata:
        if category in {"object", "ball", "cube", "handle", "fruit", "sponge", "utensil", "package"}:
            affordances.add("graspable"); regions.add("grasp_region")
        elif category == "button":
            affordances.add("pressable"); regions.add("button_surface")
        elif category in {"location", "container", "box", "basket"}:
            affordances.add("placeable"); regions.add("container_interior")
        elif category == "support_surface":
            affordances.add("placeable"); regions.add("support_surface")
    return tuple(sorted(affordances)), tuple(sorted(regions))


def _infer_category(name: str, model_name: str | None, object_id: str) -> str:
    value = " ".join(filter(None, (name, model_name, object_id))).casefold()
    rules = (
        (("handle", "把手", "手柄"), "handle"),
        (("door", "柜门", "门"), "door"),
        (("button", "按钮"), "button"),
        (("compartment", "shelf", "interior", "container", "box", "柜子上层", "盒"), "location"),
        (("ball", "球"), "ball"),
        (("cube", "方块", "魔方"), "cube"),
    )
    for tokens, category in rules:
        if any(token in value for token in tokens):
            return category
    token = re.sub(r"[^a-z0-9]+", "_", (model_name or "object").casefold()).strip("_")
    return token or "object"
