import json

import pytest

from robot_agent_sim.contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskEntity, TaskIntent, TaskStatus, TaskType
from robot_agent_sim.contracts.turn import SceneQueryIntent, SceneQueryType
from robot_agent_sim.grounding.world_relation import RelationNotSatisfied, WorldRelationResolver
from robot_agent_sim.session import SceneSession
from robot_agent_sim.session.contracts import ObjectWorldState, SemanticObject, WorldState
from robot_agent_sim.models.fake import FakeTaskUnderstandingProvider
from robot_agent_sim.pipeline.engine import PipelineEngine


def test_scene_session_reuses_runtime_and_live_world_relations(tmp_path, monkeypatch):
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
        monkeypatch.setattr(session.engine.backend, "load_uploaded", lambda *_args, **_kwargs: pytest.fail("later turns must not reinitialize a session scene"))

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


def test_scene_query_existence_is_structured(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        session.run_turn("抓苹果")
        answer = session.run_turn("场景里有没有苹果")
        assert answer["status"] == "query_answer"
        assert "存在苹果" in answer["answer"]
    finally:
        session.close()


@pytest.mark.parametrize(
    ("query", "existing"),
    [("apple", "pineapple"), ("ball", "baseball"), ("cup", "cupcake")],
)
def test_scene_query_and_edit_resolution_reject_substring_matches(tmp_path, query, existing):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    object_id = f"{existing}_01"
    session.semantic_map.objects[object_id] = SemanticObject(
        object_id=object_id, labels=[existing], category="object",
    )
    session.scene_registry = {
        "objects": [{"object_id": object_id, "semantic_name": existing, "model_name": existing}],
    }
    session.world_state = WorldState(
        world_version=1, scene_version=1, turn_index=1, sim_time=0.0,
        objects={object_id: ObjectWorldState(object_id=object_id, position=(0.0, 0.0, 0.0))},
    )

    answer = session._run_scene_query(SceneQueryIntent(
        query_type=SceneQueryType.EXISTENCE, semantic_name=query,
    ))
    assert "不存在" in answer
    with pytest.raises(ValueError, match="scene edit reference is missing"):
        session._resolve_semantic_object(query)


def test_dialogue_referent_keeps_stable_object_id(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("把两个苹果中靠近篮子的苹果抓起来")
        selected = first["result"]["grounded_task"]["entities"][0]["object_id"]
        second = session.run_turn("把它放进篮子")
        rebound = next(entity for entity in second["result"]["grounded_task"]["entities"] if entity["entity_id"].startswith("apple"))
        assert rebound["object_id"] == selected
        assert rebound["grounding_method"] == "dialogue_binding"
        assert [step["skill_name"] for step in second["result"]["skill_plan"]["steps"]] == ["locate", "move", "release"]
    finally:
        session.close()


def test_dialogue_referent_binds_spatial_relation_reference(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("把两个苹果中靠近篮子的苹果放进篮子")
        referent_id = next(
            entity["object_id"] for entity in first["result"]["grounded_task"]["entities"]
            if entity["entity_id"] == "apple_01"
        )

        second = session.run_turn("把它左边的苹果抓起来")

        assert second["status"] == "accepted"
        relation = second["result"]["task_intent"]["spatial_relations"][0]
        assert relation == {
            "subject": "apple_01",
            "relation": "left_of",
            "reference": "apple_dialogue_ref",
            "scope": "selection",
        }
        grounded = {
            entity["entity_id"]: entity
            for entity in second["result"]["grounded_task"]["entities"]
        }
        assert grounded["apple_dialogue_ref"]["object_id"] == referent_id
        assert grounded["apple_dialogue_ref"]["grounding_method"] == "dialogue_binding"
        assert grounded["apple_01"]["object_id"] != referent_id
    finally:
        session.close()


def test_session_control_pause_resume_and_close(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    paused = session.run_turn("暂停")
    assert paused["status"] == "session_paused"
    blocked = session.run_turn("抓苹果")
    assert blocked["status"] == "session_paused"
    resumed = session.run_turn("继续")
    assert resumed["status"] == "session_resumed"
    closed = session.run_turn("关闭会话")
    assert closed["status"] == "session_closed"
    assert session.control is None
    assert json.loads((session.output_root / "session.json").read_text(encoding="utf-8"))["closed"] is True
    with pytest.raises(RuntimeError, match="SESSION_CLOSED"):
        session.run_turn("抓苹果")


def test_scene_query_uses_dialogue_binding_and_world_state(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("抓苹果")
        selected = first["result"]["grounded_task"]["entities"][0]["object_id"]
        query = session.run_turn("它在哪里")
        assert query["turn_type"] == "scene_query"
        assert str(session.world_state.objects[selected].position) in query["answer"]
    finally:
        session.close()


def test_another_excludes_previous_dialogue_object(tmp_path):
    session = SceneSession(robot="ur5e", output_root=tmp_path, viewer_mode="headless")
    try:
        first = session.run_turn("把两个苹果中靠近篮子的苹果抓起来")
        first_id = first["result"]["grounded_task"]["entities"][0]["object_id"]
        second = session.run_turn("抓另一个苹果")
        second_id = second["result"]["grounded_task"]["entities"][0]["object_id"]
        assert second_id != first_id
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


def test_turn_kind_routes_scene_edit_with_one_understanding_call(tmp_path):
    class CountingUnderstanding:
        def __init__(self):
            self.delegate = FakeTaskUnderstandingProvider()
            self.call_count = 0
            self.calls = self.delegate.calls if hasattr(self.delegate, "calls") else []

        def understand(self, request):
            self.call_count += 1
            return self.delegate.understand(request)

    understanding = CountingUnderstanding()
    session = SceneSession(
        robot="ur5e", output_root=tmp_path, viewer_mode="headless",
        engine=PipelineEngine(understanding=understanding),
    )
    try:
        session.run_turn("把苹果放进篮子")
        assert understanding.call_count == 1
        edit = session.run_turn("在篮子右边增加一个香蕉")
        assert edit["turn_type"] == "scene_edit"
        assert understanding.call_count == 2
        session.run_turn("把刚才那个香蕉放进篮子")
        assert understanding.call_count == 3
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
