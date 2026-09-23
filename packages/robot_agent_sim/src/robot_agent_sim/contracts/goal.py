"""Lightweight goal-condition records reserved for post-execution verification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GoalCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    reference: str | None = None


__all__ = ["GoalCondition"]
