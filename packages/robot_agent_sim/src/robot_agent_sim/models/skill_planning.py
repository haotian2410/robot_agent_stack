"""Minimal planning-model contract; enrichment stays deterministic."""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan, SkillStep
from ..planning.context_builder import PlannerContext


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMPlanStep(StrictModel):
    skill: str
    target: Literal["source", "destination", "target", "reference"] | None = None
    reference: Literal["source", "destination", "target", "reference"] | None = None
    region: str | None = None


class LLMOperationPlan(StrictModel):
    id: str
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


def enrich_skill_plan(output: SkillPlanLLMOutput, task: GroundedTask) -> SkillPlan:
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
                LLMPlanStep(skill="locate", target=primary_role),
                LLMPlanStep(skill="move", target=primary_role, region="grasp_region"),
                LLMPlanStep(skill="grasp", target=primary_role),
                LLMPlanStep(skill="move", target=primary_role, region="relative_motion"),
                LLMPlanStep(skill="release", target=primary_role),
            ]
        for raw in raw_steps:
            target_entity = getattr(operation, raw.target) if raw.target else None
            reference_entity = getattr(operation, raw.reference) if raw.reference else None
            target = grounded.get(target_entity) if target_entity else None
            reference = grounded.get(reference_entity) if reference_entity else None
            if raw.target and target is None:
                raise ValueError(f"operation {operation.operation_id} has no {raw.target} object")
            if raw.reference and reference is None:
                raise ValueError(f"operation {operation.operation_id} has no {raw.reference} object")
            step_id = f"step-{len(steps) + 1}"
            steps.append(SkillStep(
                step_id=step_id, operation_id=operation.operation_id, skill_name=raw.skill,
                target_object=target, reference_object=reference, semantic_target=raw.region,
                motion_direction=operation.motion_direction if raw.region == "relative_motion" else None,
                distance_m=operation.distance_m if raw.region == "relative_motion" else None,
                depends_on=[steps[-1].step_id] if steps else [],
            ))
    return SkillPlan(task_types=task.task_types, steps=steps)
