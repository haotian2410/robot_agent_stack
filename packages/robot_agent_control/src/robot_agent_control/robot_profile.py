"""Robot profile loading and model-contract validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco


@dataclass(frozen=True)
class RobotProfile:
    name: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    end_effector_site: str
    home_keyframe: str
    ik_backend: str
    execution_backend: str

    @classmethod
    def load(cls, path: str | Path) -> "RobotProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(data.get("name", Path(path).stem)),
            joint_names=tuple(data["joint_names"]),
            actuator_names=tuple(data.get("actuator_names", ())),
            end_effector_site=str(data["end_effector_site"]),
            home_keyframe=str(data["home_keyframe"]),
            ik_backend=str(data["ik_backend"]),
            execution_backend=str(data["execution_backend"]),
        )

    def validate_model(self, model: mujoco.MjModel, requested_site: str) -> None:
        missing = [n for n in self.joint_names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
        missing += [n for n in self.actuator_names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) < 0]
        site = requested_site or self.end_effector_site
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            missing.append(site)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, self.home_keyframe) < 0:
            missing.append(self.home_keyframe)
        if missing:
            raise ValueError(f"missing robot profile names: {missing}")
