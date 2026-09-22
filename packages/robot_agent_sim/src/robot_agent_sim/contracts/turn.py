"""Single-pass dialogue turn classification contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TurnKind(StrEnum):
    ROBOT_TASK = "robot_task"
    SCENE_EDIT = "scene_edit"
    SCENE_QUERY = "scene_query"
    SESSION_CONTROL = "session_control"


class SceneEditType(StrEnum):
    ADD = "add"
    REMOVE = "remove"


class SceneEditRelation(StrEnum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind"
    ABOVE = "above"


class SceneEditIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: SceneEditType
    semantic_name: str
    category: str
    count: int = Field(default=1, ge=1, le=20)
    relation: SceneEditRelation | None = None
    reference: str | None = None


class SceneQueryType(StrEnum):
    COUNT = "count"
    POSITION = "position"
    STATE = "state"
    EXISTENCE = "existence"


class SceneQueryIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_type: SceneQueryType
    semantic_name: str | None = None
    category: str | None = None
    referent: bool = False
