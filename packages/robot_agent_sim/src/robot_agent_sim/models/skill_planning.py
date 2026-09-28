"""Minimal planning-model contract; enrichment stays deterministic."""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan, SkillStep
from ..planning.context_builder import PlannerContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMPlanStep(StrictModel):
    skill: str
    target_role: Literal["source", "destination", "target", "reference"] | None = None
    reference_role: Literal["source", "destination", "target", "reference"] | None = None
    region: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_role_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "target_role" not in value and "target" in value:
                value["target_role"] = value.pop("target")
            if "reference_role" not in value and "reference" in value:
                value["reference_role"] = value.pop("reference")
        return value


class LLMOperationPlan(StrictModel):
    id: str
    intent: str = Field(min_length=1, max_length=300)
    steps: list[LLMPlanStep]


class SkillPlanLLMOutput(StrictModel):
    operations: list[LLMOperationPlan]

    @model_validator(mode="after")
    def unique_operations(self):
        ids = [operation.id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("planner operation ids must be unique")
        return self


class SkillPlanningRequest(StrictModel):
    context: PlannerContext
    skill_catalog: str


class SkillPlanningProvider(Protocol):
    def plan(self, request: SkillPlanningRequest) -> SkillPlanLLMOutput: ...


_PLACEMENT_REGIONS = {
    "placement_region", "container_interior", "support_surface",
    "relative_region", "semantic_region",
}


def _normalize_pick_and_place_steps(operation, raw_steps: list[LLMPlanStep]) -> list[LLMPlanStep]:
    """Normalize the operation-owned placement tail before validation.

    The Qwen planner occasionally omits the destination ``locate`` step or
    emits only the payload-style ``move(source, reference=destination)``.
    Those are understandable descriptions, but they cannot be compiled: the
    live placement anchor belongs to the destination host and ``move`` requires
    that host to have been located.  Keep the planner's grasp prefix, then
    make the placement tail explicit and deterministic.
    """
    if operation.task_type.value != "pick_and_place":
        return raw_steps
    first_placement = next(
        (
            index
            for index, step in enumerate(raw_steps)
            if step.skill in {"move", "release"} and step.region in _PLACEMENT_REGIONS
        ),
        len(raw_steps),
    )
    prefix = list(raw_steps[:first_placement])
    if not any(step.skill == "locate" and step.target_role == "destination" for step in prefix):
        prefix.append(LLMPlanStep(skill="locate", target_role="destination"))
    prefix.extend([
        LLMPlanStep(skill="move", target_role="destination", reference_role="source", region="placement_region"),
        LLMPlanStep(skill="release", target_role="source", reference_role="destination", region="placement_region"),
    ])
    return prefix


def operation_valid_roles(operation) -> tuple[str, ...]:
    return tuple(role for role in ("source", "destination", "target", "reference") if getattr(operation, role) is not None)


class PlannerRoleError(ValueError):
    pass


def _role_error(operation, role: str, step: LLMPlanStep) -> PlannerRoleError:
    valid = list(operation_valid_roles(operation))
    return PlannerRoleError(
        f"planner_role_invalid: operation={operation.operation_id} "
        f"used_role={role} valid_roles={valid} step={step.skill}"
    )


def enrich_skill_plan(output: SkillPlanLLMOutput, task: GroundedTask, repairs: list[dict] | None = None) -> SkillPlan:
    grounded = {entity.entity_id: entity.object_id for entity in task.entities}
    operations = {operation.operation_id: operation for operation in task.operations}
    expected_ids = [operation.operation_id for operation in task.operations]
    if [operation.id for operation in output.operations] != expected_ids:
        raise ValueError("planner must preserve the exact operation order")
    steps: list[SkillStep] = []
    for operation_plan in output.operations:
        operation = operations[operation_plan.id]
        raw_steps = operation_plan.steps
        if operation.motion_direction and operation.task_type.value == "move":
            primary_role = "target" if operation.target is not None else "source"
            raw_steps = [
                LLMPlanStep(skill="locate", target_role=primary_role),
                LLMPlanStep(skill="move", target_role=primary_role, region="grasp_region"),
                LLMPlanStep(skill="grasp", target_role=primary_role),
                LLMPlanStep(skill="move", target_role=primary_role, region="relative_motion"),
                LLMPlanStep(skill="release", target_role=primary_role),
            ]
        raw_steps = _normalize_pick_and_place_steps(operation, raw_steps)
        for index, raw in enumerate(raw_steps):
            target_role = raw.target_role
            reference_role = raw.reference_role
            # A few local Qwen checkpoints use the generic literal ``target``
            # for the first locate/grasp step.  For pick_and_place this is
            # uniquely recoverable because the operation has source and
            # destination but no target role.  Never apply this repair to
            # open/close or to an operation with an actual target role.
            if (
                operation.task_type.value == "pick_and_place"
                and operation.target is None
                and target_role == "target"
                and raw.skill in {"locate", "move", "grasp"}
                and raw.region in {None, "grasp_region"}
            ):
                target_role = "source"
                if repairs is not None:
                    repairs.append({
                        "operation_id": operation.operation_id,
                        "step_index": index,
                        "type": "invalid_role_repair",
                        "field": "target_role",
                        "from": "target",
                        "to": "source",
                        "reason": "unique_role_from_pick_and_place_grasp_prefix",
                    })
            # Placement semantics belong to the operation, not to an LLM
            # choice of which role is being moved.  Qwen sometimes expresses
            # the placement move as ``move(source, reference=destination)``
            # because the payload is the object physically carried by the
            # gripper.  The execution contract, however, resolves the
            # placement anchor on the destination host, so normalize every
            # pick-and-place placement step to the canonical roles before
            # validation/compilation.
            if operation.task_type.value == "pick_and_place" and raw.region in _PLACEMENT_REGIONS:
                if raw.skill == "move":
                    target_role, reference_role = "destination", "source"
                elif raw.skill == "release":
                    target_role, reference_role = "source", "destination"
            for role in (target_role, reference_role):
                if role is not None and role not in operation_valid_roles(operation):
                    raise _role_error(operation, role, raw)
            target_entity = getattr(operation, target_role) if target_role else None
            reference_entity = getattr(operation, reference_role) if reference_role else None
            target = grounded.get(target_entity) if target_entity else None
            reference = grounded.get(reference_entity) if reference_entity else None
            if target_role and target is None:
                raise _role_error(operation, target_role, raw)
            if reference_role and reference is None:
                raise _role_error(operation, reference_role, raw)
            step_id = f"step-{len(steps) + 1}"
            steps.append(SkillStep(
                step_id=step_id, operation_id=operation.operation_id, skill_name=raw.skill,
                target_object=target, reference_object=reference, semantic_target=raw.region,
                motion_direction=operation.motion_direction if raw.region == "relative_motion" else None,
                distance_m=operation.distance_m if raw.region == "relative_motion" else None,
                depends_on=[steps[-1].step_id] if steps else [],
            ))
    return SkillPlan(task_types=task.task_types, steps=steps)
