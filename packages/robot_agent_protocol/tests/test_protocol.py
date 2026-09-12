import json

import pytest
from pydantic import ValidationError

from robot_agent_protocol import CommandDocument, ExecutionBundle, scene_sha256, validate_bundle_consistency
from robot_agent_protocol.legacy_loader import load_legacy_command_document


def test_v1_rejects_unknown_fields(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    value = {
        "schema_version": "1.0", "robot": "ur5e", "scene": str(scene),
        "registry": str(tmp_path / "registry.json"), "scene_fingerprint": scene_sha256(scene),
        "commands": [{"command_id": "c-1", "skill_name": "move", "parameters": {"target": "home"}}],
        "unknown": True,
    }
    with pytest.raises(ValidationError):
        CommandDocument.model_validate(value)


def test_bundle_paths_are_strictly_absolute(tmp_path):
    with pytest.raises(ValidationError):
        ExecutionBundle(
            robot="ur5e", route="A", task_dir="relative", scene="relative",
            scene_fingerprint="0" * 64, interaction_registry="relative", commands="relative",
        )


def test_legacy_loader_is_explicit(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"scene": str(scene), "scene_fingerprint": scene_sha256(scene)}), encoding="utf-8")
    command = tmp_path / "commands.json"
    command.write_text(json.dumps({"registry": "registry.json", "commands": [{"skill_name": "move", "parameters": {"target": "home"}}]}), encoding="utf-8")
    assert load_legacy_command_document(command).commands[0].command_id == "command-001"


def test_bundle_consistency_is_fail_closed(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    commands = tmp_path / "commands.json"
    commands.write_text(json.dumps({
        "scene": str(scene), "registry": str(tmp_path / "right.json"),
        "scene_fingerprint": scene_sha256(scene), "commands": [],
    }), encoding="utf-8")
    bundle = ExecutionBundle(
        robot="ur5e", route="A", task_dir=str(tmp_path), scene=str(scene),
        scene_fingerprint=scene_sha256(scene), interaction_registry=str(tmp_path / "wrong.json"),
        commands=str(commands),
    )
    with pytest.raises(ValueError, match="EXECUTION_BUNDLE_INCONSISTENT"):
        validate_bundle_consistency(bundle)
