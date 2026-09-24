"""Lightweight goal-condition records reserved for post-execution verification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class GoalCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str | None = None
    relation: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    reference: str | None = None
    source: Literal["explicit_goal", "operation_inferred"]
    verification_mode: Literal["world_state", "geometry", "articulation", "vision"]


__all__ = ["GoalCondition"]
