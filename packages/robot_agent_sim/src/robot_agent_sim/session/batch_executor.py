"""Batch execution helpers for concrete operation sub-plans."""
from __future__ import annotations

from ..contracts.skill_plan import SkillPlan


def slice_skill_plan_for_operation(skill: SkillPlan, operation_id: str) -> SkillPlan:
    """Return a locally numbered, dependency-free operation slice."""
    steps = [
        step.model_copy(
            update={
                "step_id": f"step-{index}",
                "depends_on": [f"step-{index - 1}"] if index > 1 else [],
            }
        )
        for index, step in enumerate((item for item in skill.steps if item.operation_id == operation_id), 1)
    ]
    return SkillPlan(task_types=skill.task_types, steps=steps)


__all__ = ["slice_skill_plan_for_operation"]
