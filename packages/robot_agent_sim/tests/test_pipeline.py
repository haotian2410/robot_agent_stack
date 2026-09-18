from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.contracts.task_intent import TaskStatus
from robot_agent_sim.grounding.iou import iou,match_detections
from robot_agent_sim.models.vision_grounding import VisionDetection
def test_pick_and_place_route_a():
    r=PipelineEngine().plan("把左边的红色方块放进右边蓝色盒子",seed=7); assert r.status=="accepted" and r.model_call_count==1; assert r.planner == "recipe"; assert [x["skill_name"] for x in r.skill_plan["steps"]]==["locate","move","grasp","locate","move","release"]

def test_generated_generic_grasp_has_provenance(tmp_path):
    import json
    result = PipelineEngine().plan("抓取红色方块", seed=7, output_dir=tmp_path)
    registry = json.loads((tmp_path / "interaction_registry.json").read_text())
    assert registry["objects"]["red_cube_01"]["interaction_metadata"]["grasp_source"] == "generic_default"
    assert all("interaction_metadata" not in item for item in registry["objects"].values() if item.get("model_name") in {"open_box", "button_basic"})
def test_seed_is_reproducible():
    assert PipelineEngine().plan("抓取红方块",seed=4).scene_registry==PipelineEngine().plan("抓取红方块",seed=4).scene_registry
def test_unsupported_direction(): assert PipelineEngine().plan("把东北角红方块抓起来").status==TaskStatus.DIRECTION_CLARIFICATION_REQUIRED
def test_unsupported_task(): assert PipelineEngine().plan("拧紧螺丝").status==TaskStatus.UNSUPPORTED_TASK
def test_iou_and_ambiguity():
    assert iou((0,0,100,100),(0,0,100,100))==1; d=VisionDetection(entity_id="x",bbox=[0,0,100,100]); m,u,a=match_detections([d],[{"object_id":"a","bbox":[0,0,100,100]},{"object_id":"b","bbox":[0,0,100,100]}]); assert not m and a==["x"]


def test_nearest_selection_binds_the_actual_nearest_candidate():
    import math

    result = PipelineEngine().plan("把离黄色方块最近的红色方块放进盒子", seed=2)
    objects = {item["object_id"]: item for item in result.scene_registry["objects"]}
    yellow = objects["yellow_cube_01"]["position"]
    candidates = [item for item in result.scene_registry["objects"] if item["candidate_for"] == "red_cube_01"]
    selected_id = result.scene_registry["bindings"]["red_cube_01"]
    selected_distance = math.dist(yellow[:2], objects[selected_id]["position"][:2])
    assert selected_distance == min(math.dist(yellow[:2], item["position"][:2]) for item in candidates)


def test_route_a_home_keyframe_preserves_generated_free_object_positions(tmp_path):
    import mujoco
    import pytest

    from robot_agent_control.utils.simulation_runtime import SceneRobotRuntime

    result = PipelineEngine().plan(
        "把离黄色方块最近的红色方块放进盒子",
        robot="ur5e", seed=2, output_dir=tmp_path,
    )
    runtime = SceneRobotRuntime(result.source_scene)
    movable = [
        item for item in result.scene_registry["objects"]
        if item["model_name"] not in {"open_box", "button_basic"}
    ]
    assert len(movable) >= 2
    for item in movable:
        body_id = mujoco.mj_name2id(
            runtime.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
        )
        assert tuple(runtime.data.xpos[body_id]) == pytest.approx(tuple(item["position"]))
    assert len({tuple(runtime.data.xpos[mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"])]) for item in movable}) == len(movable)


def test_route_a_quantity_generates_three_nearest_candidates(tmp_path):
    result = PipelineEngine().plan(
        "把三个苹果中靠近篮子的苹果放到篮子里",
        robot="ur5e", planner="recipe", output_dir=tmp_path,
    )
    candidates = [
        item for item in result.scene_registry["objects"]
        if item["candidate_for"] == "apple_01"
    ]
    assert len(candidates) == 3
    assert result.scene_registry["bindings"]["apple_01"] == min(
        candidates,
        key=lambda item: (item["position"][0] - 0.213542) ** 2 + (item["position"][1] - 0.3405) ** 2,
    )["object_id"]


def test_directional_move_recipe_grasps_moves_and_releases(tmp_path):
    result = PipelineEngine().plan(
        "把苹果向右移动一点", robot="ur5e", planner="recipe", output_dir=tmp_path,
    )
    assert result.task_intent["raw_direction"] == "right"
    assert result.task_intent["operations"][0]["distance_m"] == 0.1
    assert [step["skill_name"] for step in result.skill_plan["steps"]] == [
        "locate", "move", "grasp", "move", "release"
    ]
    assert result.skill_plan["steps"][3]["motion_direction"] == "right"
