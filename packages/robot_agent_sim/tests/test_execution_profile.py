import json

import pytest

from robot_agent_sim.execution.execution_profile import load_execution_profile


def test_builtin_execution_profile():
    profile = load_execution_profile()
    assert profile.post_grasp_lift_m > 0
    assert profile.retreat_frame in {"world", "tool"}


def test_execution_profile_override_and_invalid_frame(tmp_path, monkeypatch):
    root = tmp_path / "configs" / "execution_profiles"
    root.mkdir(parents=True)
    payload = {"post_grasp_lift_m": 0.1, "lift_relation": "above", "lift_frame": "object_local", "post_release_retreat_m": 0.2, "retreat_relation": "behind", "retreat_frame": "world"}
    root.joinpath("default.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("ROBOT_AGENT_CONFIG_ROOT", str(root.parent))
    with pytest.raises(ValueError, match="must be one"):
        load_execution_profile()
