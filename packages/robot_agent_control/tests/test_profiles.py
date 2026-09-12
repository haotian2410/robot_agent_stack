import json
from pathlib import Path

import mujoco

from robot_agent_control.robot_profile import RobotProfile


def test_builtin_ur5e_profile_loads():
    profile = RobotProfile.load_for_robot("ur5e")
    assert profile.end_effector_site == "robotiq_2f85_pinch"


def test_home_target_matches_scene_keyframe():
    profile = RobotProfile.load_for_robot("ur5e")
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).parents[1] / "world_model/robotsim/scene_001.xml")
    )
    values = profile.home_joint_positions(model)
    assert values == [float(model.key_qpos[0, model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]) for name in profile.joint_names]


def test_robot_profile_override(tmp_path, monkeypatch):
    root = tmp_path / "configs" / "robots"
    root.mkdir(parents=True)
    root.joinpath("ur5e.json").write_text(json.dumps({
        "name": "override", "joint_names": [], "actuator_names": [],
        "end_effector_site": "site", "home_keyframe": "home",
        "ik_backend": "test", "execution_backend": "test",
    }), encoding="utf-8")
    monkeypatch.setenv("ROBOT_AGENT_CONFIG_ROOT", str(root.parent))
    assert RobotProfile.load_for_robot("ur5e").name == "override"
