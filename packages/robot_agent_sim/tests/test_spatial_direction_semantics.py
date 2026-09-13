from pathlib import Path
import pytest

from robot_agent_sim.models.fake import FakeTaskUnderstandingProvider
from robot_agent_sim.models.task_understanding import TaskUnderstandingRequest, enrich_task
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.contracts.task_intent import SpatialRelationType, TaskStatus, TaskType


PARSER = FakeTaskUnderstandingProvider()


def parse(text):
    parsed = PARSER.understand(TaskUnderstandingRequest(instruction=text))
    return parsed, enrich_task(parsed, text)


def test_motion_direction_is_kept_separate_from_selection():
    parsed, intent = parse("机械臂向右移动")
    assert parsed.status == "accepted"
    assert intent.raw_direction == "right"
    assert intent.spatial_relations == []


def test_compound_motion_direction_still_requests_clarification():
    parsed, intent = parse("机械臂向左上方移动")
    assert parsed.status == "direction_clarification_required"
    assert intent.status == TaskStatus.DIRECTION_CLARIFICATION_REQUIRED


def test_left_baseball_is_entity_selection_not_motion():
    parsed, intent = parse("抓取左边的棒球")
    assert parsed.status == "accepted"
    assert intent.raw_direction is None
    assert [(r.subject, r.relation) for r in intent.spatial_relations] == [("baseball_01", SpatialRelationType.LEFT)]


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("左上角", {SpatialRelationType.LEFT, SpatialRelationType.FRONT}),
        ("右上角", {SpatialRelationType.RIGHT, SpatialRelationType.FRONT}),
        ("左下角", {SpatialRelationType.LEFT, SpatialRelationType.BACK}),
        ("右下角", {SpatialRelationType.RIGHT, SpatialRelationType.BACK}),
    ],
)
def test_corner_entity_selectors_are_planar(phrase, expected):
    parsed, intent = parse(f"抓取{phrase}的棒球")
    assert parsed.status == "accepted"
    assert intent.raw_direction is None
    relations = {r.relation for r in intent.spatial_relations if r.subject == "baseball_01"}
    assert relations == expected
    assert SpatialRelationType.UP not in relations
    assert SpatialRelationType.DOWN not in relations


@pytest.mark.parametrize(("phrase", "expected"), [("向上", "up"), ("向下", "down")])
def test_motion_direction_does_not_become_spatial_selector(phrase, expected):
    parsed, intent = parse(f"把棒球{phrase}移动")
    assert parsed.status == "accepted"
    assert intent.raw_direction == expected
    assert intent.spatial_relations == []


def test_corner_pick_and_place_layout_uses_both_axes(tmp_path):
    parsed, intent = parse("把左上角的棒球放到右下角的篮子里")
    assert parsed.status == "accepted"
    assert intent.raw_direction is None
    assert len(intent.operations) == 1
    assert intent.operations[0].task_type == TaskType.PICK_AND_PLACE
    assert intent.operations[0].source == "baseball_01"
    assert intent.operations[0].destination == "basket_01"
    selectors = {(r.subject, r.relation) for r in intent.spatial_relations if r.scope == "selection"}
    assert selectors == {
        ("baseball_01", SpatialRelationType.LEFT),
        ("baseball_01", SpatialRelationType.FRONT),
        ("basket_01", SpatialRelationType.RIGHT),
        ("basket_01", SpatialRelationType.BACK),
    }
    result = PipelineEngine().plan("把左上角的棒球放到右下角的篮子里", planner="recipe", output_dir=tmp_path)
    assert result.status == "accepted"
    objects = {item["entity_id"]: item for item in result.scene_registry["objects"]}
    assert objects["baseball_01"]["position"][0] < 0
    assert objects["baseball_01"]["position"][1] > 0
    assert objects["basket_01"]["position"][0] > 0
    assert objects["basket_01"]["position"][1] < 0


def test_selection_and_motion_direction_can_coexist():
    parsed, intent = parse("把左边的棒球向右移动")
    assert parsed.status == "accepted"
    assert intent.raw_direction == "right"
    assert intent.operations[0].source == "baseball_01"
    assert any(r.subject == "baseball_01" and r.relation == SpatialRelationType.LEFT for r in intent.spatial_relations)


def test_relative_locate_uses_selection_relations_not_raw_direction():
    parsed, intent = parse("在机械臂末端左上方找个点")
    assert parsed.status == "accepted"
    assert intent.raw_direction is None
    assert intent.operations[0].task_type == TaskType.LOCATE
    assert {(r.relation, r.subject) for r in intent.spatial_relations} == {
        (SpatialRelationType.LEFT, "target_01"),
        (SpatialRelationType.FRONT, "target_01"),
    }


def test_relative_locate_right_side_is_not_motion_direction():
    parsed, intent = parse("在盒子右侧找个位置")
    assert parsed.status == "accepted"
    assert intent.raw_direction is None
    assert intent.operations[0].task_type == TaskType.LOCATE
    assert any(r.relation == SpatialRelationType.RIGHT for r in intent.spatial_relations)
