from __future__ import annotations

import math

from ..contracts.task_intent import SpatialRelationType


DISTANCE_TIE_EPSILON = 1e-6


class WorldRelationError(ValueError):
    pass


class RelationNotSatisfied(WorldRelationError):
    pass


class RelationAmbiguous(WorldRelationError):
    pass


class WorldRelationResolver:
    """Resolve every selection relation before ranking candidates."""

    def resolve(self, intent, candidates, positions, bounds=None):
        selected = {}
        resolving = set()
        bounds = bounds or {}

        def choose(entity_id):
            if entity_id in selected:
                return selected[entity_id]
            if entity_id in resolving:
                raise WorldRelationError("cyclic selection relations")
            resolving.add(entity_id)
            values = list(candidates.get(entity_id, []))
            if not values:
                raise WorldRelationError(f"missing visual candidates for {entity_id}")
            relations = [item for item in intent.spatial_relations if item.scope == "selection" and item.subject == entity_id]
            hard = [item for item in relations if item.relation not in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST}]
            ranking = [item for item in relations if item.relation in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST}]
            for relation in hard:
                reference_position = None
                reference_bounds = None
                if relation.reference:
                    reference = choose(relation.reference)
                    reference_id = reference["object_id"]
                    reference_position = positions[reference_id]
                    reference_bounds = bounds.get(reference_id)
                values = [value for value in values if self._satisfies(relation.relation, positions[value["object_id"]], reference_position, reference_bounds)]
                if not values:
                    raise RelationNotSatisfied(f"relation_not_satisfied: {entity_id} {relation.relation} {relation.reference or ''}".strip())
            for relation in ranking:
                reference = choose(relation.reference)
                ref_pos = positions[reference["object_id"]]
                distances = [math.dist(positions[value["object_id"]][:2], ref_pos[:2]) for value in values]
                best = min(distances) if relation.relation == SpatialRelationType.NEAREST else max(distances)
                tied = [value for value, distance in zip(values, distances) if abs(distance - best) <= DISTANCE_TIE_EPSILON]
                if len(tied) != 1:
                    raise RelationAmbiguous(f"grounding_ambiguous: distance tie for {entity_id} {relation.relation}")
                values = tied
            if len(values) != 1:
                raise RelationAmbiguous(f"grounding_ambiguous: multiple candidates remain for {entity_id}")
            result = values[0]
            resolving.remove(entity_id)
            selected[entity_id] = result
            return result

        for entity in intent.entities:
            choose(entity.entity_id)
        return selected

    @staticmethod
    def _satisfies(relation, position, reference_position, reference_bounds):
        if relation == SpatialRelationType.INSIDE:
            if reference_position is None or reference_bounds is None:
                raise RelationNotSatisfied("relation_not_satisfied: inside requires container bounds")
            minimum, maximum = reference_bounds
            return all(minimum[index] <= position[index] <= maximum[index] for index in range(3))
        if reference_position is None:
            axis_sign = {
                SpatialRelationType.LEFT: (0, -1), SpatialRelationType.RIGHT: (0, 1),
                SpatialRelationType.FRONT: (1, 1), SpatialRelationType.BACK: (1, -1),
                SpatialRelationType.UP: (2, 1), SpatialRelationType.DOWN: (2, -1),
            }.get(relation)
            if axis_sign is None:
                raise WorldRelationError(f"unsupported selection relation: {relation}")
            axis, sign = axis_sign
            return position[axis] * sign > 0
        axis_sign = {
            SpatialRelationType.LEFT_OF: (0, -1), SpatialRelationType.RIGHT_OF: (0, 1),
            SpatialRelationType.FRONT_OF: (1, 1), SpatialRelationType.BEHIND: (1, -1),
            SpatialRelationType.ABOVE: (2, 1), SpatialRelationType.BELOW: (2, -1),
        }.get(relation)
        if axis_sign is None:
            raise WorldRelationError(f"unsupported selection relation: {relation}")
        axis, sign = axis_sign
        return (position[axis] - reference_position[axis]) * sign > 0
