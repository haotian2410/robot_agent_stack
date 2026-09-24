"""Single-pass dialogue turn classification contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def validate_edit_relation(self):
        if self.operation == SceneEditType.ADD and ((self.relation is None) != (self.reference is None)):
            raise ValueError("scene_edit add relation/reference must be supplied together")
        return self


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

    @model_validator(mode="after")
    def requires_target(self):
        if not self.referent and not self.semantic_name and not self.category:
            raise ValueError("scene_query requires semantic_name/category or a dialogue referent")
        return self


class SessionControlType(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    CLOSE = "close"


class SessionControlIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: SessionControlType
