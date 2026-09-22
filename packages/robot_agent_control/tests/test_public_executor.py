from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from robot_agent_control import CommandDocument, ControlExecutor, load_command_document
from robot_agent_control.session import ControlSession
from robot_agent_control.contracts import scene_sha256


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "demo" / "skill_command" / "config_001.commands.json"


def test_legacy_document_is_enriched_to_versioned_absolute_contract():
    document = load_command_document(COMMANDS)
    assert document.schema_version == "1.0"
    assert document.robot == "ur5e"
    assert Path(document.scene).is_absolute()
    assert Path(document.registry).is_absolute()
    assert document.scene_fingerprint == scene_sha256(document.scene)
    assert document.commands[0].command_id == "command-001"
    assert len(document.commands) == 18


def test_contract_rejects_unknown_fields():
    value = load_command_document(COMMANDS).model_dump()
    value["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        CommandDocument.model_validate(value)


def test_scene_fingerprint_is_checked_before_model_load():
    document = load_command_document(COMMANDS).model_copy(
        update={"scene_fingerprint": "0" * 64}
    )
    report = ControlExecutor().execute(document, viewer_mode="headless")
    assert not report.success
    assert report.failure.error_code == "SCENE_FINGERPRINT_MISMATCH"
    assert "fingerprint mismatch" in report.failure.error_message


def test_malformed_command_file_returns_process_failure(tmp_path):
    bad = tmp_path / "bad.commands.json"
    bad.write_text("{not-json", encoding="utf-8")
    result = ControlExecutor().execute(bad, viewer_mode="headless")
    assert result.success is False
    assert result.error_code == "INVALID_REQUEST"
    assert result.report is None


def test_headless_executor_writes_report_and_trace(tmp_path):
    original = load_command_document(COMMANDS)
    home = original.commands[-1].model_copy(
        update={"command_id": "home-command", "source_skill_step_id": "step-1"}
    )
    document = original.model_copy(update={"commands": [home]})
    report = ControlExecutor().execute(document, viewer_mode="headless", output_dir=tmp_path)
    assert report.success
    assert report.commands_total == report.commands_completed == 1
    assert len(report.steps) == 1
    assert report.steps[0].source_skill_step_id == "step-1"
    saved = json.loads((tmp_path / "execution_report.json").read_text(encoding="utf-8"))
    assert saved["success"] is True
    trace = (tmp_path / "skill_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace) == 1
    event = json.loads(trace[0])
    assert event["runtime_step_id"] == "runtime-step-001"
    assert event["started"] is event["completed"] is event["success"] is True
    assert event["started_at"] <= event["finished_at"]


def test_session_reload_preserves_realtime_for_auto_viewer(monkeypatch):
    # Keep this regression display-independent while exercising the same
    # preflight flag used by the persistent GUI session.
    monkeypatch.setattr(ControlSession, "_open_viewer", lambda self: None)
    monkeypatch.setattr(ControlSession, "_close_viewer", lambda self: None)
    document = load_command_document(COMMANDS)
    session = ControlSession(document, headless=False, viewer_mode="auto")
    try:
        assert session.runtime.runtime.realtime is True
        session.reload_scene(document.scene, document.registry, robot=document.robot)
        assert session.runtime.runtime.realtime is True
    finally:
        session.close()


def test_session_reload_accepts_zero_task_object_scene(tmp_path):
    from robot_agent_sim.backends.mujoco.backend import MujocoSceneBackend
    from robot_agent_sim.execution.interaction_registry_builder import build_generated_registry
    from robot_agent_sim.scene.registry import SceneRegistry

    scene = MujocoSceneBackend().compose(SceneRegistry(scene_id="zero", robot="ur5e", objects=[]), {}, tmp_path)
    interactions = build_generated_registry(scene, SceneRegistry(scene_id="zero", robot="ur5e", objects=[]), tmp_path / "interaction_registry.json")
    document = load_command_document(COMMANDS)
    session = ControlSession(document, headless=True)
    try:
        snapshot = session.reload_scene(scene, interactions, robot="ur5e")
        assert snapshot["objects"] == {}
    finally:
        session.close()


def test_registry_source_names_are_checked_before_execution(tmp_path):
    document = load_command_document(COMMANDS)
    registry = json.loads(Path(document.registry).read_text(encoding="utf-8"))
    registry["objects"]["red_ball"]["spatial"]["source"] = {
        "type": "body",
        "name": "body_that_does_not_exist",
    }
    registry_path = tmp_path / "interactions.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    document = document.model_copy(update={"registry": str(registry_path)})
    report = ControlExecutor().execute(document, viewer_mode="headless")
    assert not report.success
    assert report.failure.error_code == "REGISTRY_INVALID"
    assert "body_that_does_not_exist" in report.failure.error_message


def test_panda_execution_is_rejected_structurally():
    document = load_command_document(COMMANDS).model_copy(update={"robot": "panda"})
    report = ControlExecutor().execute(document, viewer_mode="headless")
    assert not report.success
    assert report.failure.error_code == "CONTROL_BACKEND_UNSUPPORTED_ROBOT"
    assert report.failure.command_id == document.commands[0].command_id


def test_missing_actuator_is_robot_model_incompatible(monkeypatch):
    from dataclasses import replace
    from robot_agent_control.robot_profile import RobotProfile

    original = RobotProfile.load_for_robot
    monkeypatch.setattr(
        RobotProfile, "load_for_robot",
        classmethod(lambda cls, robot: replace(original(robot), actuator_names=("missing_actuator",))),
    )
    report = ControlExecutor().execute(load_command_document(COMMANDS), viewer_mode="headless")
    assert not report.success
    assert report.failure.error_code == "ROBOT_MODEL_INCOMPATIBLE"
    assert "missing_actuator" in report.failure.error_message


def test_unsupported_skill_has_structured_trace_context():
    document = load_command_document(COMMANDS)
    command = document.commands[0].model_copy(
        update={"command_id": "bad-command", "source_skill_step_id": "bad-step", "skill_name": "teleport"}
    )
    report = ControlExecutor().execute(
        document.model_copy(update={"commands": [command]}), viewer_mode="headless"
    )
    assert not report.success
    assert report.failure.error_code == "UNSUPPORTED_SKILL"
    assert report.failure.command_id == "bad-command"
    assert report.failure.source_skill_step_id == "bad-step"
    assert report.failure.skill_name == "teleport"
