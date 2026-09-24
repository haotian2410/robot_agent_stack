from __future__ import annotations

import math
import random
import re

from ..contracts.task_intent import Direction, QuantityMode, SpatialRelationType
from .constraints import SceneConstraintError
from .registry import SceneObject, SceneRegistry
from .support_surfaces import WORKSPACE_X, WORKSPACE_Y

MIN_GAP_M = 0.04


class SceneComposer:
    def compose(self, intent, assets, robot: str = "panda", seed: int = 0) -> SceneRegistry:
        rng = random.Random(seed)
        ranking_relations = {
            SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST,
            SpatialRelationType.LEFTMOST, SpatialRelationType.RIGHTMOST,
            SpatialRelationType.FRONTMOST, SpatialRelationType.BACKMOST,
            SpatialRelationType.HIGHEST, SpatialRelationType.LOWEST,
        }
        selection_subjects = {
            entity.entity_id for entity in intent.entities
            if entity.quantity_mode == QuantityMode.CANDIDATE_POOL
        } | {
            r.subject for r in intent.spatial_relations
            if r.scope in {"scene", "selection"} and r.relation in ranking_relations
        }
        objects: list[SceneObject] = []
        bindings: dict[str, str] = {}

        by_entity = {entity.entity_id: entity for entity in intent.entities}
        preferred_by_entity: dict[str, tuple[float, float, float]] = {}
        for relation in intent.spatial_relations:
            if relation.scope not in {"scene", "selection"}:
                continue
            if not relation.reference or relation.relation not in {
                SpatialRelationType.LEFT_OF,
                SpatialRelationType.RIGHT_OF,
                SpatialRelationType.FRONT_OF,
                SpatialRelationType.BEHIND,
                SpatialRelationType.ABOVE,
                SpatialRelationType.BELOW,
            }:
                continue
            subject_dimensions = assets[relation.subject].dimensions_m or _primitive_dimensions(assets[relation.subject].model_name)
            reference_dimensions = assets[relation.reference].dimensions_m or _primitive_dimensions(assets[relation.reference].model_name)
            subject_position, reference_position = self._relation_pair_positions(
                relation.relation, subject_dimensions, reference_dimensions
            )
            preferred_by_entity.setdefault(relation.subject, subject_position)
            preferred_by_entity.setdefault(relation.reference, reference_position)
        placed: set[str] = set()
        placing: set[str] = set()

        def place(entity_id: str) -> None:
            if entity_id in placed or entity_id in selection_subjects:
                return
            entity = by_entity[entity_id]
            if entity_id in placing:
                raise ValueError("cyclic spatial relations are not supported")
            placing.add(entity_id)
            relation = next(
                (
                    relation for relation in intent.spatial_relations
                    if relation.scope in {"scene", "selection"}
                    and relation.subject == entity_id
                    and relation.reference
                    and relation.relation in {
                        SpatialRelationType.LEFT_OF,
                        SpatialRelationType.RIGHT_OF,
                        SpatialRelationType.FRONT_OF,
                        SpatialRelationType.BEHIND,
                        SpatialRelationType.ABOVE,
                        SpatialRelationType.BELOW,
                        SpatialRelationType.INSIDE,
                    }
                ),
                None,
            )
            preferred = preferred_by_entity.get(entity_id)
            if relation is not None:
                place(relation.reference)
                if relation.relation == SpatialRelationType.INSIDE:
                    reference_item = next(item for item in objects if item.object_id == bindings[relation.reference])
                    preferred = (
                        reference_item.position[0],
                        reference_item.position[1],
                        reference_item.position[2],
                    )
            item = self._make_object(entity, assets[entity_id], 1, objects, rng, intent, preferred=preferred, allow_overlap_object=(bindings[relation.reference] if relation and relation.relation == SpatialRelationType.INSIDE else None))
            objects.append(item)
            bindings[entity.entity_id] = item.object_id
            placed.add(entity_id)
            placing.discard(entity_id)

        for entity in intent.entities:
            place(entity.entity_id)

        for entity in [e for e in intent.entities if e.entity_id in selection_subjects]:
            relation = next((
                r for r in intent.spatial_relations
                if r.scope in {"scene", "selection"}
                and r.subject == entity.entity_id
                and r.relation in ranking_relations
            ), None)
            reference = next((item for item in objects if relation and relation.reference and item.object_id == bindings[relation.reference]), None)
            if relation and relation.relation in {SpatialRelationType.HIGHEST, SpatialRelationType.LOWEST}:
                raise SceneConstraintError(
                    f"scene_generation_constraint_failed: vertical ranking requires supported height levels for {entity.entity_id}"
                )
            # Generate the requested number of candidates.  Selection tasks
            # need at least two candidates even when the language omits an
            # explicit count; an explicit count such as “三个苹果” is kept.
            candidate_count = max(int(entity.count), 2)
            candidates = []
            ranking_positions = self._candidate_ranking_positions(relation, candidate_count, reference, assets[entity.entity_id].dimensions_m or _primitive_dimensions(assets[entity.entity_id].model_name))
            for index in range(candidate_count):
                if ranking_positions is not None:
                    preferred = ranking_positions[index]
                elif index == 0:
                    preferred = ((reference.position[0] + 0.12) if reference else 0.12, reference.position[1] if reference else 0.0)
                else:
                    preferred = (
                        (reference.position[0] if reference else 0.0) - 0.26 - 0.12 * (index - 1),
                        (reference.position[1] if reference else 0.0) + 0.18 * (index - 1),
                    )
                candidate = self._make_object(
                    entity, assets[entity.entity_id], index + 1, objects, rng,
                    intent, preferred=preferred,
                )
                objects.append(candidate)
                candidates.append(candidate)
            ranked = self._rank_candidates(candidates, reference, relation)
            # Preserve the natural-language selection in the registry.
            bindings[entity.entity_id] = ranked[0].object_id

        return SceneRegistry(scene_id=f"{robot}_generated_{seed}", robot=robot, objects=objects, bindings=bindings)

    @staticmethod
    def _candidate_ranking_positions(relation, count, reference, dimensions):
        if relation is None or relation.relation not in {
            SpatialRelationType.LEFTMOST, SpatialRelationType.RIGHTMOST,
            SpatialRelationType.FRONTMOST, SpatialRelationType.BACKMOST,
        }:
            return None
        width, depth, _ = dimensions
        spacing = max(width if relation.relation in {SpatialRelationType.LEFTMOST, SpatialRelationType.RIGHTMOST} else depth, MIN_GAP_M + 0.02, 0.12)
        if relation.relation in {SpatialRelationType.LEFTMOST, SpatialRelationType.RIGHTMOST}:
            center = reference.position[0] if reference else 0.0
            start = center - spacing * (count - 1) / 2
            return [(round(start + spacing * index, 6), round(reference.position[1] if reference else 0.0, 6), 0.0) for index in range(count)]
        center = reference.position[1] if reference else 0.0
        start = center - spacing * (count - 1) / 2
        return [(round(reference.position[0] if reference else 0.0, 6), round(start + spacing * index, 6), 0.0) for index in range(count)]

    @staticmethod
    def _rank_candidates(candidates, reference, relation):
        if relation is None:
            return candidates
        if relation.relation in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST}:
            key = lambda item: _distance(item.position, reference.position)
        else:
            axis = {
                SpatialRelationType.LEFTMOST: 0, SpatialRelationType.RIGHTMOST: 0,
                SpatialRelationType.FRONTMOST: 1, SpatialRelationType.BACKMOST: 1,
                SpatialRelationType.HIGHEST: 2, SpatialRelationType.LOWEST: 2,
            }[relation.relation]
            key = lambda item: item.position[axis]
        reverse = relation.relation in {SpatialRelationType.FARTHEST, SpatialRelationType.RIGHTMOST, SpatialRelationType.FRONTMOST, SpatialRelationType.HIGHEST}
        return sorted(candidates, key=key, reverse=reverse)

    @staticmethod
    def _relation_pair_positions(relation, subject_dimensions, reference_dimensions):
        gap = MIN_GAP_M + 0.01
        subject = [0.0, 0.0, 0.0]
        reference = [0.0, 0.0, 0.0]
        if relation == SpatialRelationType.LEFT_OF:
            separation = reference_dimensions[0] / 2 + subject_dimensions[0] / 2 + gap
            subject[0], reference[0] = -separation / 2, separation / 2
        elif relation == SpatialRelationType.RIGHT_OF:
            separation = reference_dimensions[0] / 2 + subject_dimensions[0] / 2 + gap
            subject[0], reference[0] = separation / 2, -separation / 2
        elif relation == SpatialRelationType.FRONT_OF:
            separation = reference_dimensions[1] / 2 + subject_dimensions[1] / 2 + gap
            subject[1], reference[1] = separation / 2, -separation / 2
        elif relation == SpatialRelationType.BEHIND:
            separation = reference_dimensions[1] / 2 + subject_dimensions[1] / 2 + gap
            subject[1], reference[1] = -separation / 2, separation / 2
        elif relation == SpatialRelationType.ABOVE:
            subject[2] = reference_dimensions[2] + gap
        elif relation == SpatialRelationType.BELOW:
            reference[2] = subject_dimensions[2] + gap
        return tuple(subject), tuple(reference)

    def _make_object(self, entity, asset, index, existing, rng, intent, preferred=None, allow_overlap_object=None):
        dimensions = asset.dimensions_m or _primitive_dimensions(asset.model_name)
        position = self._position(entity.entity_id, dimensions, existing, rng, intent, preferred, allow_overlap_object)
        base = _slug(entity.semantic_name or asset.model_name)
        if base == "scene_object":
            # Non-Latin semantic names collapse to the generic slug.  Prefer
            # the model-provided stable entity id so two Chinese entities do
            # not both become scene_object_01.
            base = _slug(entity.entity_id or asset.model_name)
        object_id = f"{base}_{index:02d}"
        return SceneObject(object_id=object_id, body_name=object_id, role="task_object",
                           semantic_name=entity.semantic_name, position=position, dimensions_m=dimensions,
                           entity_id=entity.entity_id if index == 1 else None, candidate_for=entity.entity_id,
                           model_id=asset.model_id, model_name=asset.model_name)

    def _position(self, entity_id, dimensions, existing, rng, intent, preferred, allow_overlap_object=None):
        unary_relations = {
            SpatialRelationType.LEFT: Direction.LEFT,
            SpatialRelationType.RIGHT: Direction.RIGHT,
            SpatialRelationType.FRONT: Direction.FRONT,
            SpatialRelationType.BACK: Direction.BACK,
            SpatialRelationType.UP: Direction.UP,
            SpatialRelationType.DOWN: Direction.DOWN,
        }
        unary = [unary_relations[r.relation] for r in intent.spatial_relations if r.scope in {"scene", "selection"} and r.subject == entity_id and r.relation in unary_relations]
        presets = {Direction.LEFT: (-0.22, 0.0, 0.0), Direction.RIGHT: (0.22, 0.0, 0.0),
                   Direction.FRONT: (0.0, 0.38, 0.0), Direction.BACK: (0.0, -0.38, 0.0),
                   Direction.UP: (0.0, 0.0, 0.20), Direction.DOWN: (0.0, 0.0, 0.02)}
        candidates = []
        if preferred is not None: candidates.append((preferred[0], preferred[1], preferred[2] if len(preferred) == 3 else 0.0))
        if unary:
            combined = [0.0, 0.0, 0.0]
            for direction in unary:
                preset = presets[direction]
                for axis, value in enumerate(preset):
                    if value:
                        combined[axis] = value
            candidates.append(tuple(combined))
        candidates.extend((rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y), 0.0) for _ in range(200))
        for x, y, z in candidates:
            x = max(WORKSPACE_X[0] + dimensions[0] / 2, min(WORKSPACE_X[1] - dimensions[0] / 2, x))
            y = max(WORKSPACE_Y[0] + dimensions[1] / 2, min(WORKSPACE_Y[1] - dimensions[1] / 2, y))
            candidate = (round(x, 6), round(y, 6), round(z, 6))
            if all(item.object_id == allow_overlap_object or not _overlap(candidate, dimensions, item.position, item.dimensions_m or (0, 0, 0)) for item in existing):
                return candidate
        raise ValueError("unable to find a valid non-overlapping placement")


def _primitive_dimensions(name):
    return {"cube_basic": (0.06, 0.06, 0.06), "open_box": (0.18, 0.18, 0.10), "button_basic": (0.08, 0.08, 0.04)}[name]


def _overlap(a, ad, b, bd):
    xy_overlap = (
        abs(a[0] - b[0]) < (ad[0] + bd[0]) / 2 + MIN_GAP_M
        and abs(a[1] - b[1]) < (ad[1] + bd[1]) / 2 + MIN_GAP_M
    )
    z_separated = a[2] >= b[2] + bd[2] + MIN_GAP_M or b[2] >= a[2] + ad[2] + MIN_GAP_M
    return xy_overlap and not z_separated


def _distance(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])


def _slug(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "scene_object"
