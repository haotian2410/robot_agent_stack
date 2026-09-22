"""State contracts for a persistent robot-agent scene session."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from ..contracts.turn import SceneEditIntent, SceneEditType, SceneQueryIntent, SceneQueryType, TurnKind


class ObjectWorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    body_name: str | None = None
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] | None = None


class JointWorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    position: float
    velocity: float | None = None


class WorldState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world_version: int = Field(ge=0)
    scene_version: int = Field(ge=1)
    turn_index: int = Field(ge=0)
    sim_time: float = Field(ge=0)
    objects: dict[str, ObjectWorldState] = {}
    joints: dict[str, JointWorldState] = {}
    robot_qpos: list[float] = []
    robot_qvel: list[float] = []
    held_object: str | None = None


class SemanticObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    labels: list[str] = []
    category: str | None = None
    attributes: dict[str, str] = {}
    source: str = "authored"
    confidence: float | None = None
    semantic_confidence: float | None = None
    identity_iou: float | None = None
    last_verified_world_version: int | None = None


class SemanticMap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: dict[str, SemanticObject] = {}


class DialogueState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recent_turns: list[str] = []
    referents: dict[str, str] = {}
    last_grounded_objects: dict[str, str] = {}
