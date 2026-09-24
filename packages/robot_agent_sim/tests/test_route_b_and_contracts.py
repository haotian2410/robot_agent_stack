import json
from pathlib import Path

import httpx
import pytest

from robot_agent_sim.backends.mujoco.backend import MujocoSceneBackend
from robot_agent_sim.models.fake import FakeVisionGroundingProvider
from robot_agent_sim.models.qwen_http import QwenHTTPProvider
from robot_agent_sim.models.task_understanding import TaskUnderstandingRequest
from robot_agent_sim.models.vision_grounding import VisionDetection
from robot_agent_sim.models.budget import ModelCallBudget, ModelCallMode
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.grounding.interaction_registry import ground_partial_with_interaction_registry
from robot_agent_sim.contracts.task_intent import TaskEntity, TaskIntent, TaskStatus, TaskType, Operation, SpatialRelation, SpatialRelationType
from robot_agent_sim.scene.registry import SceneObject, SceneRegistry
from robot_agent_sim.grounding.segmentation import InstanceObservation, SceneObservation
from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.task_intent import Operation, TaskType
from robot_agent_sim.grounding.candidates import GroundingCandidate


SCENE_003 = Path(__file__).parents[1] / "assets/robots/ur5e/scenes/scene_003.xml"


def test_mujoco_headless_rgb_segmentation_smoke(tmp_path):
    registry = MujocoSceneBackend().load_uploaded(SCENE_003, "ur5e")
    observation = MujocoSceneBackend().renderer.render(SCENE_003, registry, tmp_path)
    assert observation.rgb_path.is_file()
    assert observation.segmentation_path.is_file()
    assert observation.segmentation_visualization_path.is_file()


def test_route_b_auto_discovers_task_body_and_grounding_artifact(tmp_path):
    registry = MujocoSceneBackend().load_uploaded(SCENE_003, "ur5e")
    assert [item.body_name for item in registry.objects] == ["push_button_base"]

    provider = FakeVisionGroundingProvider(
        [VisionDetection(entity_id="button_01", bbox=[710, 412, 867, 525])]
    )
    result = PipelineEngine(vision=provider).plan(
        "按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path
    )
    assert result.status == "accepted"
    assert result.model_call_count == 2
    assert Path(result.artifacts["visual_grounding.json"]).is_file()
    assert Path(result.artifacts["grounding_candidates.json"]).is_file()
    assert Path(result.artifacts["grounding_decision.json"]).is_file()
    assert Path(result.artifacts["raw_vision_grounding.json"]).is_file()
    assert Path(result.artifacts["semantic_validation.json"]).is_file()
    decision = json.loads(Path(result.artifacts["grounding_decision.json"]).read_text())
    assert decision["entities"][0]["object_id"] == "scene_object_001"
    assert decision["entities"][0]["grounding_method"] == "vlm_iou"
    assert result.grounded_task["entities"][0]["object_id"] == "scene_object_001"
    assert [step["skill_name"] for step in result.skill_plan["steps"]] == [
        "locate", "move", "press"
    ]
    validation = json.loads(Path(result.artifacts["semantic_validation.json"]).read_text())
    assert validation["task_intent_validation"]["status"] == "accepted"
    assert validation["grounding_validation"]["status"] == "accepted"
    assert validation["skill_plan_validation"]["status"] == "accepted"
    assert validation["skill_plan_validation"]["operation_outcomes"] == [{"operation_id": "op-1", "result": "plan_validated"}]
    assert validation["skill_plan_validation"]["semantic_plan_outcomes"] == [{"operation_id": "op-1", "status": "plan_validated"}]
    assert validation["skill_plan_validation"]["execution_goal_status"] == "not_verified"
    goals = json.loads(Path(result.artifacts["goal_conditions.json"]).read_text())
    assert goals == []


def test_route_b_sidecar_takes_precedence(tmp_path):
    scene = tmp_path / "custom.xml"
    scene.write_text(SCENE_003.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar = scene.with_name("custom.scene_registry.json")
    sidecar.write_text(json.dumps({
        "scene_id": "declared-scene",
        "robot": "ur5e",
        "objects": [{
            "object_id": "button-semantic",
            "body_name": "push_button_base",
            "semantic_name": "red button",
            "role": "target",
            "model_id": "button-model",
            "model_name": "button",
        }],
    }), encoding="utf-8")
    registry = MujocoSceneBackend().load_uploaded(scene, "panda")
    assert registry.scene_id == "declared-scene"
    assert registry.robot == "ur5e"
    assert registry.objects[0].object_id == "button-semantic"
    assert registry.objects[0].semantic_name == "red button"


def test_route_b_partial_sidecar_merges_undeclared_bodies(tmp_path):
    scene = tmp_path / "partial.xml"
    scene.write_text(
        """<mujoco><worldbody><body name="body_a"><geom type="sphere" size="0.02"/></body><body name="body_b"><geom type="sphere" size="0.02"/></body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    scene.with_name("partial.scene_registry.json").write_text(json.dumps({
        "scene_id": "partial",
        "robot": "ur5e",
        "objects": [{"object_id": "red_ball", "body_name": "body_a", "semantic_name": "red ball", "role": "target"}],
    }), encoding="utf-8")
    registry = MujocoSceneBackend().load_uploaded(scene, "ur5e")
    by_body = {item.body_name: item for item in registry.objects}
    assert by_body["body_a"].object_id == "red_ball"
    assert by_body["body_a"].semantic_name == "red ball"
    assert by_body["body_b"].object_id.startswith("scene_object_")


def test_partial_interaction_registry_returns_known_and_unresolved_entities(tmp_path):
    registry = tmp_path / "partial.interactions.json"
    registry.write_text(json.dumps({
        "objects": {
            "red_ball": {
                "aliases": ["red ball", "红球"],
                "body_name": "push_button_base",
                "spatial": {"source": {"type": "body", "name": "push_button_base"}},
            }
        }
    }), encoding="utf-8")
    entities = [
        TaskEntity(entity_id="red_ball", semantic_name="red ball", category="ball"),
        TaskEntity(entity_id="blue_box", semantic_name="blue box", category="container"),
    ]
    known, missing = ground_partial_with_interaction_registry(entities, registry, SCENE_003)
    assert [item.object_id for item in known] == ["red_ball"]
    assert [item.entity_id for item in missing] == ["blue_box"]


def test_partial_interaction_registry_defers_cross_entity_relation(tmp_path):
    registry = tmp_path / "partial-relations.interactions.json"
    registry.write_text(json.dumps({
        "objects": {
            "red_ball": {
                "aliases": ["red ball"],
                "spatial": {"source": {"type": "body", "name": "push_button_base"}},
            }
        }
    }), encoding="utf-8")
    entities = [
        TaskEntity(entity_id="red_ball", semantic_name="red ball", category="ball"),
        TaskEntity(entity_id="blue_box", semantic_name="blue box", category="container"),
    ]
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="把红球放到蓝色盒子",
        task_types=[TaskType.PICK_AND_PLACE],
        entities=entities,
        operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="red_ball", destination="blue_box")],
        spatial_relations=[SpatialRelation(subject="red_ball", relation=SpatialRelationType.LEFT_OF, reference="blue_box")],
    )
    known, missing = ground_partial_with_interaction_registry(entities, registry, SCENE_003, intent=intent, positions={"red_ball": (0, 0, 0), "blue_box": (1, 0, 0)})
    assert known == []
    assert {item.entity_id for item in missing} == {"red_ball", "blue_box"}


def test_route_b_default_fake_detection_fails_without_forcing_binding(tmp_path):
    result = PipelineEngine().plan("按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path)
    assert result.status == "grounding_failed"
    assert result.grounded_task is None
    assert result.visual_grounding["unmatched"] == ["button_01"]
    assert "rgb.png" in result.artifacts


def test_route_b_semantic_cache_skips_second_vision_call(tmp_path):
    class CountingVision(FakeVisionGroundingProvider):
        def __init__(self):
            super().__init__([VisionDetection(entity_id="button_01", bbox=[710, 412, 867, 525])])
            self.calls_count = 0

        def detect(self, request):
            self.calls_count += 1
            return super().detect(request)

    vision = CountingVision()
    engine = PipelineEngine(vision=vision)
    first = engine.plan("按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path / "first")
    assert first.status == "accepted"
    semantic_map = {"objects": {"scene_object_001": {"labels": ["button", "按钮"]}}}
    second = engine.plan("按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path / "second", semantic_map=semantic_map)
    assert second.status == "accepted"
    assert second.visual_grounding["method"] == "semantic_cache"
    assert vision.calls_count == 1


def test_candidate_pool_cache_is_completed_by_vision_before_ranking(tmp_path):
    class CountingVision(FakeVisionGroundingProvider):
        def __init__(self):
            super().__init__([
                VisionDetection(entity_id="apple", bbox=[200, 200, 300, 300]),
                VisionDetection(entity_id="apple", bbox=[400, 400, 500, 500]),
            ])
            self.requests = []

        def detect(self, request):
            self.requests.append(request)
            return super().detect(request)

    from robot_agent_sim.contracts.task_intent import QuantityMode, SpatialRelation, SpatialRelationType

    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="three apples, rightmost",
        task_types=[TaskType.GRASP],
        entities=[TaskEntity(entity_id="apple", semantic_name="apple", category="fruit", count=3, quantity_mode=QuantityMode.CANDIDATE_POOL)],
        operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.RIGHTMOST)],
    )
    observation = SceneObservation(
        scene_id="three", camera_id="scene", image_width_px=640, image_height_px=480,
        rgb_path=tmp_path / "rgb.png", segmentation_path=tmp_path / "seg.npy",
        segmentation_visualization_path=tmp_path / "seg.png", instance_index_path=tmp_path / "instances.json",
        instances=[
            InstanceObservation(object_id="apple_01", body_name="apple_01", bbox=(100, 100, 180, 180), visible_pixel_count=10, world_position=(-0.2, 0, 0)),
            InstanceObservation(object_id="apple_02", body_name="apple_02", bbox=(200, 200, 300, 300), visible_pixel_count=10, world_position=(0.0, 0, 0)),
            InstanceObservation(object_id="apple_03", body_name="apple_03", bbox=(400, 400, 500, 500), visible_pixel_count=10, world_position=(0.2, 0, 0)),
        ],
    )
    vision = CountingVision()
    engine = PipelineEngine(vision=vision)
    grounded, _ = engine._resolve_candidate_map(
        intent,
        intent.entities,
        {"apple": [GroundingCandidate(object_id="apple_01", sources={"semantic_map"}).model_dump(mode="json")]},
        SceneRegistry(scene_id="three", robot="ur5e", objects=[]),
        observation,
        {"apple_01": (-0.2, 0, 0), "apple_02": (0.0, 0, 0), "apple_03": (0.2, 0, 0)},
        ModelCallBudget.for_mode(ModelCallMode.CURRENT_SCENE, "recipe"),
    )
    assert len(vision.requests) == 1
    assert vision.requests[0].entities[0].all is True
    assert grounded[0].object_id == "apple_03"

    too_many = {
        "apple": [
            {"object_id": "apple_01", "sources": ["semantic_map"]},
            {"object_id": "apple_02", "sources": ["vision"]},
            {"object_id": "apple_03", "sources": ["vision"]},
            {"object_id": "apple_04", "sources": ["vision"]},
        ]
    }
    with pytest.raises(ValueError, match="grounding_candidate_count_mismatch"):
        engine._resolve_candidate_map(
            intent,
            intent.entities,
            too_many,
            SceneRegistry(scene_id="three", robot="ur5e", objects=[]),
            observation,
            {"apple_01": (-0.2, 0, 0), "apple_02": (0.0, 0, 0), "apple_03": (0.2, 0, 0), "apple_04": (0.3, 0, 0)},
            ModelCallBudget.for_mode(ModelCallMode.CURRENT_SCENE, "recipe"),
        )


def _two_object_route_b(monkeypatch, tmp_path, vision):
    scene = tmp_path / "two.xml"
    scene.write_text('<mujoco><worldbody><body name="body_a"><geom type="sphere" size="0.02"/></body><body name="body_b" pos="0.2 0 0"><geom type="box" size="0.03 0.03 0.03"/></body></worldbody></mujoco>', encoding="utf-8")
    registry = SceneRegistry(scene_id="two", robot="ur5e", objects=[
        SceneObject(object_id="scene_object_001", body_name="body_a", role="target", semantic_name="body_a", source="uploaded"),
        SceneObject(object_id="scene_object_002", body_name="body_b", role="target", semantic_name="body_b", source="uploaded"),
    ])
    observation = SceneObservation(
        scene_id="two", camera_id="-1", image_width_px=640, image_height_px=480,
        rgb_path=tmp_path / "rgb.png", segmentation_path=tmp_path / "seg.npy",
        segmentation_visualization_path=tmp_path / "seg.png", instance_index_path=tmp_path / "instances.json",
        instances=[
            InstanceObservation(object_id="scene_object_001", body_name="body_a", bbox=(100, 100, 200, 200), visible_pixel_count=100, world_position=(0.0, 0.0, 0.0)),
            InstanceObservation(object_id="scene_object_002", body_name="body_b", bbox=(300, 300, 400, 400), visible_pixel_count=100, world_position=(0.2, 0.0, 0.0)),
        ],
    )
    engine = PipelineEngine(vision=vision)
    monkeypatch.setattr(engine.backend, "load_uploaded", lambda path, robot: registry)
    monkeypatch.setattr(engine.backend.renderer, "render", lambda path, value, out: observation)
    return engine, scene


def test_mixed_cached_and_uncached_entities_only_send_uncached_to_vision(monkeypatch, tmp_path):
    class RecordingVision(FakeVisionGroundingProvider):
        def __init__(self):
            super().__init__([VisionDetection(entity_id="blue_box_01", bbox=[300, 300, 400, 400])])
            self.requests = []

        def detect(self, request):
            self.requests.append(request)
            return super().detect(request)

    vision = RecordingVision()
    engine, scene = _two_object_route_b(monkeypatch, tmp_path, vision)
    result = engine.plan(
        "把红球放到蓝色盒子", robot="ur5e", scene=scene, output_dir=tmp_path / "out",
        semantic_map={"objects": {"scene_object_001": {"labels": ["red ball"]}}},
    )
    assert result.status == "accepted"
    assert [[entity.id for entity in request.entities] for request in vision.requests] == [["blue_box_01"]]
    methods = {entity["entity_id"]: entity["grounding_method"] for entity in result.grounded_task["entities"]}
    assert methods == {"red_ball_01": "semantic_cache", "blue_box_01": "vlm_iou"}


def test_partial_interaction_registry_uses_cached_missing_entity_without_vision(monkeypatch, tmp_path):
    class FailingVision(FakeVisionGroundingProvider):
        def __init__(self):
            super().__init__([])
            self.calls_count = 0

        def detect(self, request):
            self.calls_count += 1
            raise AssertionError("vision should not be called")

    vision = FailingVision()
    engine, scene = _two_object_route_b(monkeypatch, tmp_path, vision)
    interactions = tmp_path / "partial.json"
    interactions.write_text(json.dumps({"objects": {"red_ball": {"aliases": ["red ball"], "spatial": {"source": {"type": "body", "name": "body_a"}}}}}), encoding="utf-8")
    result = engine.plan(
        "把红球放到蓝色盒子", robot="ur5e", scene=scene, interaction_registry=interactions,
        output_dir=tmp_path / "out", semantic_map={"objects": {"scene_object_002": {"labels": ["blue box"]}}},
    )
    assert result.status == "accepted"
    assert vision.calls_count == 0


def test_partial_interaction_registry_only_sends_unresolved_entity_to_vision(monkeypatch, tmp_path):
    class RecordingVision(FakeVisionGroundingProvider):
        def __init__(self):
            super().__init__([VisionDetection(entity_id="blue_box_01", bbox=[300, 300, 400, 400])])
            self.requests = []

        def detect(self, request):
            self.requests.append(request)
            return super().detect(request)

    vision = RecordingVision()
    engine, scene = _two_object_route_b(monkeypatch, tmp_path, vision)
    interactions = tmp_path / "partial-vision.json"
    interactions.write_text(json.dumps({
        "objects": {
            "red_ball": {
                "aliases": ["red ball"],
                "spatial": {"source": {"type": "body", "name": "body_a"}},
            }
        }
    }), encoding="utf-8")

    result = engine.plan(
        "把红球放到蓝色盒子", robot="ur5e", scene=scene,
        interaction_registry=interactions, output_dir=tmp_path / "out",
    )

    assert result.status == "accepted"
    assert [[entity.id for entity in request.entities] for request in vision.requests] == [["blue_box_01"]]
    assert result.visual_grounding["providers"] == {
        "red_ball_01": ["interaction_registry"],
        "blue_box_01": ["vision"],
    }
    methods = {entity["entity_id"]: entity["grounding_method"] for entity in result.grounded_task["entities"]}
    assert methods == {"red_ball_01": "interaction_registry", "blue_box_01": "vlm_iou"}


def test_current_scene_allows_optional_vision_fallback_with_qwen_budget(monkeypatch, tmp_path):
    class RecordingVision(FakeVisionGroundingProvider):
        def __init__(self):
            super().__init__([VisionDetection(entity_id="blue_box_01", bbox=[300, 300, 400, 400])])
            self.calls_count = 0

        def detect(self, request):
            self.calls_count += 1
            return super().detect(request)

    vision = RecordingVision()
    engine, scene = _two_object_route_b(monkeypatch, tmp_path, vision)
    registry = engine.backend.load_uploaded(scene, "ur5e")
    result = engine.plan_current_scene(
        "把红球放到蓝色盒子", scene_path=scene, scene_registry=registry, origin="uploaded",
        robot="ur5e", planner="qwen", output_dir=tmp_path / "current", semantic_map={"objects": {"scene_object_001": {"labels": ["red ball"]}}},
    )
    assert result.status == "accepted"
    assert result.model_call_count == 3
    assert vision.calls_count == 1


def test_current_scene_semantic_cache_hit_skips_optional_vision(monkeypatch, tmp_path):
    class FailingVision(FakeVisionGroundingProvider):
        def detect(self, request):
            raise AssertionError("semantic cache hit must not call vision")

    vision = FailingVision([])
    engine, scene = _two_object_route_b(monkeypatch, tmp_path, vision)
    registry = engine.backend.load_uploaded(scene, "ur5e")
    result = engine.plan_current_scene(
        "把红球放到蓝色盒子", scene_path=scene, scene_registry=registry, origin="uploaded",
        robot="ur5e", planner="qwen", output_dir=tmp_path / "current-cache",
        semantic_map={"objects": {
            "scene_object_001": {"labels": ["red ball"]},
            "scene_object_002": {"labels": ["blue box"]},
        }},
    )
    assert result.status == "accepted"
    assert result.model_call_count == 2
    assert result.visual_grounding["method"] == "semantic_cache"


def test_qwen_provider_sends_fixed_stage_and_extracts_json(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"status":"unsupported_task","raw_task":"x","instruction":"x"}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenHTTPProvider("http://localhost:8000/v1", "qwen-test")
    intent = provider.understand(TaskUnderstandingRequest(instruction="x"))
    assert intent.status == "unsupported_task"
    assert len(calls) == 1
    assert calls[0][0].endswith("/chat/completions")
    body = calls[0][1]["json"]
    assert body["temperature"] == 0
    assert body["messages"][1]["content"] == '{"instruction":"x"}'
    assert provider.calls[0]["stage"] == "task_understanding"
    assert body["response_format"]["type"] == "json_schema"
    assert "json_schema" in body["response_format"]
    assert provider.last_raw_values["task_understanding"]["instruction"] == "x"
    assert "instruction" not in intent.model_dump(mode="json")


@pytest.mark.parametrize(
    ("mode", "has_response_format"),
    [("json_schema", True), ("json_object", True), ("off", False)],
)
def test_qwen_structured_output_modes(monkeypatch, mode, has_response_format):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"status":"unsupported_task","raw_task":"x"}'}}]}

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenHTTPProvider("http://localhost:8000/v1", "qwen-test", use_structured_output=mode)
    provider.understand(TaskUnderstandingRequest(instruction="x"))
    payload = calls[0]
    assert ("response_format" in payload) is has_response_format
    if mode == "json_schema":
        assert payload["response_format"]["type"] == "json_schema"
        assert "schema" in payload["response_format"]["json_schema"]
        assert "只输出一个 JSON 对象" not in payload["messages"][0]["content"]
    elif mode == "json_object":
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0]["content"].endswith("只输出一个 JSON 对象，不要 markdown 或额外文字。")
    else:
        assert "只输出一个 JSON 对象，不要 markdown 或额外文字。" in payload["messages"][0]["content"]


def test_qwen_http_pipeline_level4_shape_with_local_transport(monkeypatch, tmp_path):
    """Exercise task -> vision -> skill planning through the real HTTP adapter.

    The transport is local and deterministic; this verifies the same request
    and response contracts used by a real Qwen-compatible server without
    pretending that a real model endpoint is available in CI.
    """
    responses = {
        "task_understanding": {
            "status": "accepted",
            "entities": [{"id": "button_01", "name": "red button", "category": "button", "color": "red"}],
            "operations": [{"type": "press", "target": "button_01"}],
            "relations": [],
        },
        "vision_grounding": {
            "detections": [{"entity": "button_01", "bbox": [710, 412, 867, 525]}],
        },
        "skill_planning": {
            "operations": [{"id": "op-1", "steps": [
                {"skill": "locate", "target": "target"},
                {"skill": "move", "target": "target", "region": "button_surface"},
                {"skill": "press", "target": "target"},
            ]}],
        },
    }

    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(self.body)}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        stage = kwargs["json"]["response_format"]["json_schema"]["name"]
        return Response(responses[stage])

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    provider = QwenHTTPProvider("http://local-mock/v1", "qwen-test")
    result = PipelineEngine(understanding=provider, vision=provider, planner=provider).plan(
        "按红色按钮", robot="ur5e", scene=SCENE_003,
        planner="qwen", output_dir=tmp_path,
    )
    assert result.status == "accepted"
    assert result.route == "B"
    assert result.planner == "qwen"
    assert result.model_call_count == 3
    assert [step["skill_name"] for step in result.skill_plan["steps"]] == ["locate", "move", "press"]
    assert [item["stage"] for item in provider.calls] == ["task_understanding", "vision_grounding", "skill_planning"]
    assert len(calls) == 3


def test_skill_plan_rejects_unknown_target_and_wrong_order():
    from robot_agent_sim.contracts.skill_plan import SkillPlan, SkillStep
    from robot_agent_sim.planning.recipes import validate_plan
    task = GroundedTask(
        instruction="按按钮",
        task_types=[TaskType.PRESS],
        entities=[GroundedEntity(entity_id="button", semantic_name="button", object_id="button-1", grounding_method="asset_scene_binding")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.PRESS, target="button")],
        spatial_relations=[],
        scene_id="scene",
    )
    with pytest.raises(ValueError):
        validate_plan(SkillPlan(task_types=[TaskType.PRESS], steps=[SkillStep(step_id="step-1", operation_id="op-1", skill_name="press", target_object="missing")]), task)
