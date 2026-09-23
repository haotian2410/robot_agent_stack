"""Post-compose validation for generated scene semantics."""

from __future__ import annotations

import math

from ..contracts.task_intent import SpatialRelationType


class SceneConstraintError(ValueError):
    pass


def validate_generated_scene(intent, registry) -> None:
    objects = {item.object_id: item for item in registry.objects}
    positions = {object_id: item.position for object_id, item in objects.items()}
    for relation in intent.spatial_relations:
        if relation.scope == "goal":
            continue
        subject_id = registry.bindings.get(relation.subject)
        reference_id = registry.bindings.get(relation.reference) if relation.reference else None
        if subject_id is None or (relation.reference and reference_id is None):
            raise SceneConstraintError(f"scene_generation_constraint_failed: missing binding for {relation.subject}")
        if relation.relation in {
            SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST,
            SpatialRelationType.LEFTMOST, SpatialRelationType.RIGHTMOST,
            SpatialRelationType.FRONTMOST, SpatialRelationType.BACKMOST,
            SpatialRelationType.HIGHEST, SpatialRelationType.LOWEST,
        }:
            candidates = [item for item in registry.objects if item.candidate_for == relation.subject]
            if not candidates:
                candidates = [objects[subject_id]]
            if relation.relation in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST}:
                reference_position = positions[reference_id]
                values = [math.dist(item.position[:2], reference_position[:2]) for item in candidates]
                selected_value = math.dist(positions[subject_id][:2], reference_position[:2])
                expected = min(values) if relation.relation == SpatialRelationType.NEAREST else max(values)
                if sum(abs(value - expected) <= 1e-6 for value in values) != 1:
                    raise SceneConstraintError(f"scene_generation_constraint_failed: {relation.relation} is not unique")
            else:
                axis, high = {
                    SpatialRelationType.LEFTMOST: (0, False), SpatialRelationType.RIGHTMOST: (0, True),
                    SpatialRelationType.FRONTMOST: (1, True), SpatialRelationType.BACKMOST: (1, False),
                    SpatialRelationType.HIGHEST: (2, True), SpatialRelationType.LOWEST: (2, False),
                }[relation.relation]
                values = [item.position[axis] for item in candidates]
                selected_value = positions[subject_id][axis]
                expected = max(values) if high else min(values)
                if sum(abs(value - expected) <= 1e-6 for value in values) != 1:
                    raise SceneConstraintError(f"scene_generation_constraint_failed: {relation.relation} is not unique")
            if abs(selected_value - expected) > 1e-6:
                raise SceneConstraintError(f"scene_generation_constraint_failed: {relation.relation} not satisfied")
            continue
        if relation.relation == SpatialRelationType.INSIDE:
            container = objects[reference_id]
            half = tuple(dimension / 2 for dimension in (container.dimensions_m or (0, 0, 0)))
            subject = objects[subject_id]
            point = positions[subject_id]
            subject_half = tuple(dimension / 2 for dimension in (subject.dimensions_m or (0, 0, 0)))
            if not all(
                abs(point[index] - container.position[index]) + subject_half[index] <= half[index]
                for index in range(3)
            ):
                raise SceneConstraintError("scene_generation_constraint_failed: inside relation not satisfied")
            continue
        subject = positions[subject_id]
        reference = positions[reference_id] if reference_id else (0.0, 0.0, 0.0)
        checks = {
            SpatialRelationType.LEFT: subject[0] < reference[0], SpatialRelationType.RIGHT: subject[0] > reference[0],
            SpatialRelationType.FRONT: subject[1] > reference[1], SpatialRelationType.BACK: subject[1] < reference[1],
            SpatialRelationType.UP: subject[2] > reference[2], SpatialRelationType.DOWN: subject[2] < reference[2],
            SpatialRelationType.LEFT_OF: subject[0] < reference[0], SpatialRelationType.RIGHT_OF: subject[0] > reference[0],
            SpatialRelationType.FRONT_OF: subject[1] > reference[1], SpatialRelationType.BEHIND: subject[1] < reference[1],
            SpatialRelationType.ABOVE: subject[2] > reference[2], SpatialRelationType.BELOW: subject[2] < reference[2],
        }
        if relation.relation not in checks or not checks[relation.relation]:
            raise SceneConstraintError(f"scene_generation_constraint_failed: {relation.relation} not satisfied")
