from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from .task_intent import Operation, QuantityMode, SpatialRelation, TaskType

class GroundedMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    body_name: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    grounding_method: str
    detection_bbox: tuple[int, int, int, int] | None = None
    instance_bbox: tuple[int, int, int, int] | None = None
    bbox_iou: float | None = Field(default=None, ge=0, le=1)

class GroundedEntitySet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_id: str
    semantic_name: str
    category: str | None = None
    color: str | None = None
    quantity_mode: QuantityMode
    expected_count: int | None = None
    members: list[GroundedMember]

class GroundedSemanticTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str
    task_types: list[TaskType]
    entity_sets: list[GroundedEntitySet]
    operations: list[Operation]
    spatial_relations: list[SpatialRelation]
    scene_id: str
