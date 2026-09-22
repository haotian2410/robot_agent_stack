from types import SimpleNamespace

import numpy as np

from robot_agent_control.collision.mujoco_checker import MujocoCollisionChecker


def test_held_object_collision_with_non_target_obstacle_is_not_whitelisted(monkeypatch):
    monkeypatch.setattr("robot_agent_control.collision.mujoco_checker.mujoco.mj_forward", lambda model, data: None)
    controller = SimpleNamespace(_held_body_id=10, sync_held_object=lambda data: None)
    runtime = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(6)),
        qpos_indices=np.arange(6),
        gripper_controller=controller,
    )
    checker = MujocoCollisionChecker.__new__(MujocoCollisionChecker)
    checker.runtime = runtime
    checker.model = SimpleNamespace(ngeom=4, geom_bodyid=np.array([1, 10, 20, 30]))
    checker._data = SimpleNamespace(
        qpos=np.zeros(6),
        ncon=1,
        contact=[SimpleNamespace(geom1=1, geom2=2, dist=-0.01)],
    )
    checker.robot_geom_ids = {0}
    checker.robot_body_ids = {1}
    checker.baseline_allowed_pairs = set()
    checker.allowed_target_geom_ids = {3}
    checker.allowed_held_contact_geom_ids = set()
    checker._geom_name = lambda geom_id: f"geom_{geom_id}"
    checker._is_held_object_pad_contact = lambda *args: False

    result = checker.check(np.zeros(6))

    assert result.in_collision is True
    assert result.contacts[0]["geom2"] == "geom_2"


def test_only_explicit_container_content_contact_is_whitelisted(monkeypatch):
    monkeypatch.setattr("robot_agent_control.collision.mujoco_checker.mujoco.mj_forward", lambda model, data: None)
    controller = SimpleNamespace(_held_body_id=10, sync_held_object=lambda data: None)
    runtime = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(6)), qpos_indices=np.arange(6), gripper_controller=controller)
    checker = MujocoCollisionChecker.__new__(MujocoCollisionChecker)
    checker.runtime = runtime
    checker.model = SimpleNamespace(ngeom=4, geom_bodyid=np.array([1, 10, 20, 30]))
    checker._data = SimpleNamespace(qpos=np.zeros(6), ncon=1, contact=[SimpleNamespace(geom1=1, geom2=2, dist=-0.01)])
    checker.robot_geom_ids = {0}
    checker.robot_body_ids = {1}
    checker.baseline_allowed_pairs = set()
    checker.allowed_target_geom_ids = {3}
    checker.allowed_held_contact_geom_ids = {2}
    checker._geom_name = lambda geom_id: f"geom_{geom_id}"
    checker._is_held_object_pad_contact = lambda *args: False

    assert checker.check(np.zeros(6)).in_collision is False
