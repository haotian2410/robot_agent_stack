from pathlib import Path

from robot_agent_sim.execution.control_adapter import execute_bundle
from robot_agent_sim.execution.compiler import compile_directory
from robot_agent_sim.pipeline.engine import PipelineEngine


def test_route_a_headless_pick_and_place(tmp_path):
    result = PipelineEngine().plan(
        "把红色方块放进蓝色盒子", robot="ur5e", output_dir=tmp_path
    )
    assert result.status == "accepted"
    bundle = compile_directory(tmp_path)
    report = execute_bundle(bundle, viewer_mode="headless")
    assert report.success
    assert report.commands_completed == report.commands_total


def test_route_b_scene_001_headless_full_flow(tmp_path):
    root = Path(__file__).parents[2]
    scene = root / "packages/robot_agent_control/world_model/robotsim/scene_001.xml"
    registry = root / "packages/robot_agent_control/demo/common/scenes/scene_001.interactions.json"
    result = PipelineEngine().plan(
        "打开柜门，把红球放到柜子上层，然后关闭柜门",
        robot="ur5e", scene=scene, interaction_registry=registry, output_dir=tmp_path,
    )
    assert result.status == "accepted"
    assert [item["task_type"] for item in result.grounded_task["operations"]] == ["open", "pick_and_place", "close"]
    skills_by_operation = {}
    for step in result.skill_plan["steps"]:
        skills_by_operation.setdefault(step["operation_id"], []).append(step["skill_name"])
    assert "pull" in skills_by_operation["op-1"]
    assert "release" in skills_by_operation["op-2"]
    assert "push" in skills_by_operation["op-3"]
    assert result.route == "B"
    bundle = compile_directory(tmp_path)
    report = execute_bundle(bundle, viewer_mode="headless")
    assert report.success
    assert report.commands_completed == report.commands_total


def test_route_a_button_press_has_structured_result(tmp_path):
    result = PipelineEngine().plan("按按钮", robot="ur5e", output_dir=tmp_path)
    assert result.status == "accepted"
    bundle = compile_directory(tmp_path)
    report = execute_bundle(bundle, viewer_mode="headless")
    # The generated primitive is intentionally fail-closed until a target
    # sensor/mechanism is authored; either outcome must remain structured.
    assert report.failure is None or report.failure.error_code in {
        "TARGET_DEPTH_NOT_REACHED", "TARGET_NOT_ACTUATED", "LOCAL_COLLISION_DETECTED",
    }
