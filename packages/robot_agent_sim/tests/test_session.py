import json

import pytest

from robot_agent_sim.contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskEntity, TaskIntent, TaskStatus, TaskType
from robot_agent_sim.grounding.world_relation import RelationNotSatisfied, WorldRelationResolver
from robot_agent_sim.session import SceneSession


def test_scene_session_reuses_runtime_and_live_world_relations(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("把两个苹果中靠近篮子的苹果放到篮子里")
        assert first["report"]["success"] is True
        assert first["result"]["grounded_task"]["entities"][0]["object_id"] == "apple_02"
        process_id = session.control.process.pid
        moved_position = session.world_state.objects["apple_02"].position
        basket_position = session.world_state.objects["basket_01"].position
        assert abs(moved_position[0] - basket_position[0]) < 0.02
        assert abs(moved_position[1] - basket_position[1]) < 0.02

        second = session.run_turn("把现在离篮子最远的苹果抓起来")
        assert session.control.process.pid == process_id
        assert second["report"]["success"] is True
        assert second["result"]["grounded_task"]["entities"][0]["object_id"] == "apple_01"
        assert session.origin == "generated"
        assert session.scene_version == 1
        assert session.world_version == 2
        assert session.world_state.held_object == "apple_01"
        assert second["result"]["route"] == "A"
        assert (session.output_root / "turns/0001/execution_report.json").is_file()
        assert (session.output_root / "turns/0002/execution_report.json").is_file()
        saved = json.loads((session.output_root / "state/world_state.json").read_text())
        assert saved["turn_index"] == 2
    finally:
        session.close()


def test_dialogue_referent_keeps_stable_object_id(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("把两个苹果中靠近篮子的苹果抓起来")
        selected = first["result"]["grounded_task"]["entities"][0]["object_id"]
        second = session.run_turn("把它放进篮子")
        rebound = next(entity for entity in second["result"]["grounded_task"]["entities"] if entity["entity_id"].startswith("apple"))
        assert rebound["object_id"] == selected
        assert rebound["grounding_method"] == "dialogue_binding"
    finally:
        session.close()


def test_remove_held_object_is_rejected(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("抓苹果")
        assert first["report"]["success"] is True
        assert session.world_state.held_object == "apple_01"
        with pytest.raises(ValueError, match="HELD_OBJECT_REMOVE_FORBIDDEN"):
            session.run_turn("删除苹果")
    finally:
        session.close()


def test_scene_edit_reloads_runtime_and_continues_with_new_object(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("把两个苹果中靠近篮子的苹果放进篮子")
        assert first["report"]["success"] is True
        apple_position = session.world_state.objects["apple_02"].position
        robot_qpos = tuple(session.world_state.robot_qpos)
        updated = session.run_turn("在篮子右边增加一个香蕉")
        assert updated["status"] == "scene_updated"
        assert session.scene_version == 2
        assert session.world_state.objects["apple_02"].position == apple_position
        assert tuple(session.world_state.robot_qpos) == robot_qpos
        final = session.run_turn("把刚才那个香蕉放进篮子")
        assert final["report"]["success"] is True
        assert session.world_state.objects["banana_01"].object_id == "banana_01"
        assert (session.output_root / "turns/0002/scene_patch.json").is_file()
    finally:
        session.close()


def test_unique_candidate_must_satisfy_explicit_unary_relation():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="抓右边的苹果",
        task_types=[TaskType.GRASP],
        entities=[TaskEntity(entity_id="apple", semantic_name="apple", category="apple")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.RIGHT, scope="selection")],
    )
    with pytest.raises(RelationNotSatisfied, match="relation_not_satisfied"):
        WorldRelationResolver().resolve(intent, {"apple": [{"object_id": "apple_01"}]}, {"apple_01": (-0.4, 0.0, 0.0)})
