"""Resolve vague motion scales from grounded object geometry."""
from __future__ import annotations

from ..contracts.task_intent import MotionScale
from ..models.motion_policy import MotionPolicy


def resolve_motion_scales(task, scene_registry, policy: MotionPolicy | None = None):
    """Convert semantic scales to metres after concrete instance grounding.

    The characteristic length is the object's extent on the requested motion
    axis: X for left/right, Y for front/back, and Z for up/down.  The model
    never chooses the final metric distance.
    """
    policy = policy or MotionPolicy()
    entities = {entity.entity_id: entity for entity in task.entities}
    objects = {item.object_id: item for item in scene_registry.objects}
    updated = []
    records = []
    for operation in task.operations:
        if operation.task_type.value != "move" or operation.distance_m is not None or operation.motion_scale is None:
            updated.append(operation)
            continue
        actor_id = operation.target or operation.source
        entity = entities.get(actor_id)
        item = objects.get(entity.object_id) if entity is not None else None
        dimensions = item.dimensions_m if item is not None else None
        direction = operation.motion_direction.value if operation.motion_direction else None
        axis = 0 if direction in {"left", "right"} else 1 if direction in {"front", "back"} else 2
        factor = policy.scale_factor(operation.motion_scale)
        if dimensions is not None and len(dimensions) == 3 and dimensions[axis] > 0:
            characteristic_length_m = float(dimensions[axis])
            distance_m = characteristic_length_m * factor
            source = "object_axis_dimension"
        else:
            raise ValueError(
                "motion_scale_geometry_missing: "
                f"operation={operation.operation_id} object={actor_id} "
                "requires dimensions_m or another reliable geometry source"
            )
        distance_m = min(max(distance_m, 1e-6), 2.0)
        updated.append(operation.model_copy(update={"distance_m": distance_m, "motion_scale": None}))
        records.append({
            "operation_id": operation.operation_id,
            "object_id": entity.object_id if entity is not None else None,
            "direction": direction,
            "axis": ("x", "y", "z")[axis],
            "scale": operation.motion_scale.value,
            "factor": factor,
            "characteristic_length_m": characteristic_length_m,
            "distance_m": distance_m,
            "source": source,
        })
    return task.model_copy(update={"operations": updated}), records


__all__ = ["resolve_motion_scales"]
