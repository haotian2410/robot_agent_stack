"""Portable execution-profile resource loader."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionProfile:
    post_grasp_lift_m: float
    lift_relation: str
    lift_frame: str
    post_release_retreat_m: float
    retreat_relation: str
    retreat_frame: str


def load_execution_profile(name: str = "default") -> ExecutionProfile:
    override = os.environ.get("ROBOT_AGENT_CONFIG_ROOT")
    path = Path(override) / "execution_profiles" / f"{name}.json" if override else None
    if path is not None and path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        resource = files("robot_agent_sim").joinpath("resources", "execution_profiles", f"{name}.json")
        raw = json.loads(resource.read_text(encoding="utf-8"))
    required = ("post_grasp_lift_m", "lift_relation", "lift_frame", "post_release_retreat_m", "retreat_relation", "retreat_frame")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"EXECUTION_PROFILE_INVALID: missing required fields {missing}")
    allowed_frames = {"world", "tool"}
    for key in ("lift_frame", "retreat_frame"):
        if raw[key] not in allowed_frames:
            raise ValueError(f"EXECUTION_PROFILE_INVALID: {key} must be one of {sorted(allowed_frames)}")
    return ExecutionProfile(
        post_grasp_lift_m=float(raw["post_grasp_lift_m"]), lift_relation=str(raw["lift_relation"]), lift_frame=str(raw["lift_frame"]),
        post_release_retreat_m=float(raw["post_release_retreat_m"]), retreat_relation=str(raw["retreat_relation"]), retreat_frame=str(raw["retreat_frame"]),
    )
