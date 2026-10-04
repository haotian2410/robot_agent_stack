from importlib.resources import files

import pytest

from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.task_intent import Direction, MotionScale, Operation, TaskType
from robot_agent_sim.models.motion_policy import MotionPolicy
from robot_agent_sim.models.prompts import (
    SKILL_PLANNING_PROMPT,
    TASK_UNDERSTANDING_PROMPT,
    VISION_GROUNDING_PROMPT,
)
from robot_agent_sim.planning.motion_scale import resolve_motion_scales
from robot_agent_sim.scene.registry import SceneObject, SceneRegistry
from robot_agent_sim.scene.support_surfaces import SupportSurface, TableCameraSpec


def test_runtime_prompts_are_loaded_from_packaged_text_files():
    prompt_package = files("robot_agent_sim.models.prompt_templates")
    assert TASK_UNDERSTANDING_PROMPT == prompt_package.joinpath("task_understanding_v2.txt").read_text(encoding="utf-8")
    assert VISION_GROUNDING_PROMPT == prompt_package.joinpath("vision_grounding_v1.txt").read_text(encoding="utf-8")
    assert SKILL_PLANNING_PROMPT == prompt_package.joinpath("skill_planning_v2.txt").read_text(encoding="utf-8")


def _move_task(motion_scale=MotionScale.SMALL):
    return GroundedTask(
        instruction="move apple right",
        task_types=[TaskType.MOVE],
        entities=[GroundedEntity(entity_id="apple", semantic_name="apple", object_id="apple-01", grounding_method="asset_scene_binding")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.MOVE, target="apple", motion_direction=Direction.RIGHT, motion_scale=motion_scale)],
        spatial_relations=[],
        scene_id="scene",
    )


def test_motion_scale_uses_configured_percentage_of_object_axis():
    registry = SceneRegistry(scene_id="scene", robot="ur5e", objects=[SceneObject(
        object_id="apple-01", body_name="apple", role="target", semantic_name="apple", dimensions_m=(0.2, 0.1, 0.1)
    )])
    task, records = resolve_motion_scales(_move_task(MotionScale.MEDIUM), registry, MotionPolicy(medium_scale_factor=0.5))
    assert task.operations[0].distance_m == pytest.approx(0.1)
    assert records[0]["source"] == "object_axis_dimension"


def test_motion_scale_requires_geometry_instead_of_fixed_meter_fallback():
    registry = SceneRegistry(scene_id="scene", robot="ur5e", objects=[SceneObject(
        object_id="apple-01", body_name="apple", role="target", semantic_name="apple", dimensions_m=None
    )])
    with pytest.raises(ValueError, match="motion_scale_geometry_missing"):
        resolve_motion_scales(_move_task(), registry)


def test_table_camera_position_is_surface_percentage_based():
    surface = SupportSurface("table", "table", (1.0, 2.0, 0.0), (0.8, 1.2, 0.03))
    assert TableCameraSpec().world_position(surface) == pytest.approx((1.0, 2.0, 1.2))
    assert TableCameraSpec(xy_ratio=(0.25, 0.75), height_ratio=0.5).world_position(surface) == pytest.approx((0.8, 2.3, 0.6))
