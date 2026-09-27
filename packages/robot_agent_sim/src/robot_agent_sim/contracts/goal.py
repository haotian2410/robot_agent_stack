"""Lightweight goal-condition records reserved for post-execution verification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from enum import StrEnum
from typing import Literal


class GoalRelation(StrEnum):
    HELD = "held"
    NOT_HELD = "not_held"
    INSIDE = "inside"
    OPEN = "open"
    CLOSED = "closed"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind"
    ABOVE = "above"
    BELOW = "below"


class GoalSource(StrEnum):
    EXPLICIT_GOAL = "explicit_goal"
    OPERATION_INFERRED = "operation_inferred"
    EXPLICIT_AND_INFERRED = "explicit_and_inferred"


class VerificationMode(StrEnum):
    WORLD_STATE = "world_state"
    GEOMETRY = "geometry"
    ARTICULATION = "articulation"
    VISION = "vision"


class GoalCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str | None = None
    parent_operation_id: str | None = None
    subject_entity_id: str | None = None
    subject_object_id: str | None = None
    reference_entity_id: str | None = None
    reference_object_id: str | None = None
    relation: GoalRelation
    subject: str = Field(min_length=1)
    reference: str | None = None
    source: GoalSource
    verification_mode: VerificationMode


__all__ = ["GoalCondition", "GoalRelation", "GoalSource", "VerificationMode"]
