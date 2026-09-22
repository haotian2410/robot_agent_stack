"""Generic semantic validation for model-composed atomic skill plans."""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan
from ..skills.registry import AtomicSkillRegistry, REGISTRY
from .context_builder import PlannerContext


@dataclass(frozen=True)
class SemanticEvent:
    sequence: int
    operation_id: str
    skill: str
    target: str | None
    reference: str | None
    region: str | None


def validate_semantic_plan(
    plan: SkillPlan,
    task: GroundedTask,
    context: PlannerContext,
    skill_registry: AtomicSkillRegistry = REGISTRY,
) -> None:
    """Validate grounding, affordances and a minimal sequential world state."""
    try:
        _validate(plan, task, context, skill_registry)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("semantic_plan_invalid:"):
            raise
        raise ValueError(f"semantic_plan_invalid: {message}") from exc


def _validate(plan, task, context, skill_registry) -> None:
    object_to_entity = {entity.object_id: entity.entity_id for entity in task.entities}
    entity_context = {entity.id: entity for entity in context.entities}
    operation_ids = [operation.operation_id for operation in task.operations]
    positions = {operation_id: index for index, operation_id in enumerate(operation_ids)}
    by_operation = {operation_id: [] for operation_id in operation_ids}
    last_position = -1
    operation_events: dict[str, list[SemanticEvent]] = {operation_id: [] for operation_id in operation_ids}
    operation_initial_held: dict[str, str | None] = {}
    operation_reached: dict[str, set[tuple[str, str | None]]] = {operation_id: set() for operation_id in operation_ids}

    located: set[str] = set()
    reached: set[tuple[str, str | None]] = set()
    held: str | None = context.initial_state.held_entity

    active_operation: str | None = None

    def check_outcome(operation_id: str | None, held_state: str | None) -> None:
        if operation_id is None:
            return
        operation = next(item for item in task.operations if item.operation_id == operation_id)
        events = operation_events[operation_id]
        event_names = {event.skill for event in events}
        reached_for_operation = operation_reached[operation_id]
        target_entity = operation.target or operation.source
        if operation.task_type.value == "grasp" and held_state != target_entity:
            raise ValueError(f"grasp operation did not end holding {target_entity}")
        if operation.task_type.value == "pick_and_place":
            destination = operation.destination
            source_initially_held = operation_initial_held.get(operation_id) == operation.source
            grasped_source = any(event.skill == "grasp" and event.target == operation.source for event in events)
            valid_release = any(
                event.skill == "release"
                and event.target == operation.source
                and event.reference == destination
                and event.region == "container_interior"
                for event in events
            )
            if (not source_initially_held and not grasped_source) or not valid_release or (destination, "container_interior") not in reached_for_operation:
                raise ValueError("pick_and_place operation outcome is incomplete")
            if held_state == operation.source:
                raise ValueError("pick_and_place operation did not release source")
        if operation.task_type.value == "release" and not any(event.skill == "release" and event.target == target_entity for event in events):
            raise ValueError("release operation did not execute release")
        if operation.task_type.value == "press" and not any(event.skill == "press" and event.target == operation.target for event in events):
            raise ValueError("press operation did not execute press")
        if operation.task_type.value == "open":
            mechanism = [event for event in events if event.skill in {"pull", "push"}]
            if not mechanism or mechanism[-1].skill != "pull" or mechanism[-1].target != operation.target or mechanism[-1].reference != operation.reference:
                raise ValueError("open operation did not end with matching pull")
        if operation.task_type.value == "close":
            mechanism = [event for event in events if event.skill in {"pull", "push"}]
            if not mechanism or mechanism[-1].skill != "push" or mechanism[-1].target != operation.target or mechanism[-1].reference != operation.reference:
                raise ValueError("close operation did not end with matching push")
        if operation.task_type.value == "locate" and not any(event.skill == "locate" and event.target == target_entity for event in events):
            raise ValueError("locate operation did not locate target")
        if operation.task_type.value == "move" and not any(event.skill == "move" and event.target == target_entity for event in events):
            raise ValueError("move operation did not execute move")
        if operation.task_type.value == "move" and operation.motion_direction and not any(event.skill == "move" and event.region == "relative_motion" and event.target == target_entity for event in events):
            raise ValueError("directional move operation did not perform relative_motion")

    for step in plan.steps:
        definition = skill_registry.require(step.skill_name)
        if step.operation_id not in by_operation:
            raise ValueError(f"skill references unknown operation: {step.operation_id}")
        current_position = positions[step.operation_id]
        if current_position < last_position:
            raise ValueError("skill steps must preserve operation order")
        if active_operation != step.operation_id:
            check_outcome(active_operation, held)
            active_operation = step.operation_id
            operation_initial_held.setdefault(step.operation_id, held)
        last_position = current_position
        by_operation[step.operation_id].append(step)
        if definition.requires_target and not step.target_object:
            raise ValueError(f"skill {step.skill_name} requires target_object")
        target = _resolve(step.target_object, object_to_entity, "target")
        reference = _resolve(step.reference_object, object_to_entity, "reference")
        if step.semantic_target is not None and step.semantic_target not in definition.allowed_regions:
            raise ValueError(f"unsupported region for {step.skill_name}: {step.semantic_target}")
        target_facts = entity_context.get(target) if target else None
        if target and target_facts is None:
            raise ValueError(f"skill target is not in planner context: {target}")
        for affordance in definition.required_affordances:
            if target_facts is None or affordance not in target_facts.affordances:
                raise ValueError(f"{step.skill_name} requires {affordance} affordance on {target}")

        operation_events[step.operation_id].append(SemanticEvent(
            sequence=len(operation_events[step.operation_id]),
            operation_id=step.operation_id,
            skill=step.skill_name,
            target=target,
            reference=reference,
            region=step.semantic_target,
        ))

        if step.skill_name == "search":
            raise ValueError("search is unavailable after grounding")
        if step.skill_name == "locate":
            located.add(target)
        elif step.skill_name == "move":
            if target not in located:
                raise ValueError(f"move requires located target: {target}")
            region = step.semantic_target
            if region in {"grasp_region", "container_interior", "button_surface"}:
                if target_facts is None or region not in target_facts.regions:
                    raise ValueError(f"target {target} has no semantic region {region}")
            reached.add((target, region))
            operation_reached[step.operation_id].add((target, region))
            if region == "relative_motion" and held != target:
                raise ValueError(f"relative_motion requires held target: {target}")
        elif step.skill_name == "grasp":
            if target not in located:
                raise ValueError(f"grasp requires located target: {target}")
            if (target, "grasp_region") not in reached:
                raise ValueError(f"grasp requires reached grasp_region for {target}")
            if held is not None and held != target:
                raise ValueError(f"cannot grasp {target} while holding {held}")
            held = target
        elif step.skill_name == "release":
            if held != target:
                raise ValueError(f"release requires held target: {target}")
            if step.semantic_target:
                destination = reference or target
                if (destination, step.semantic_target) not in reached:
                    raise ValueError(
                        f"release requires reached {step.semantic_target} for {destination}"
                    )
            held = None
        elif step.skill_name == "press":
            if target not in located:
                raise ValueError(f"press requires located target: {target}")
            if (target, "button_surface") not in reached:
                raise ValueError(f"press requires reached button_surface for {target}")
        elif step.skill_name in {"pull", "push"}:
            contact = reference or target
            if held != contact:
                raise ValueError(
                    f"{step.skill_name} requires established grasp/contact on {contact}"
                )
            if reference and f"part_of:{target}" not in entity_context[reference].relations:
                raise ValueError(
                    f"{step.skill_name} contact {reference} is not related to target {target}"
                )

    check_outcome(active_operation, held)

    for operation in task.operations:
        if not by_operation[operation.operation_id]:
            raise ValueError(f"operation has no skill steps: {operation.operation_id}")
        for dependency in operation.depends_on:
            if dependency not in positions or positions[dependency] >= positions[operation.operation_id]:
                raise ValueError("operation dependency order violated")
            if not by_operation[dependency]:
                raise ValueError("operation dependency has no steps")


def _resolve(object_id: str | None, mapping: dict[str, str], role: str) -> str | None:
    if object_id is None:
        return None
    if object_id not in mapping:
        raise ValueError(f"skill {role} is not grounded: {object_id}")
    return mapping[object_id]
