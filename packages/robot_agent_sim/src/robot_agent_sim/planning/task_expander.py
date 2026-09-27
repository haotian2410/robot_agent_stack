"""Expand grounded semantic sets into the existing single-instance contract."""
from __future__ import annotations
from copy import deepcopy
from ..contracts.grounded_task import GroundedEntity, GroundedTask
from ..contracts.task_intent import QuantityMode, Operation
from .assignment_resolver import resolve_assignments

def expand_grounded_task(
    task: GroundedTask,
    entity_members: dict[str, list[str]],
    registry,
    *,
    pairwise: bool = False,
) -> tuple[GroundedTask, dict]:
    by_entity = {e.entity_id: e for e in task.entities}
    concrete: list[GroundedEntity] = []
    aliases: dict[str, list[str]] = {}
    for entity in task.entities:
        members = list(entity_members.get(entity.entity_id, [entity.object_id]))
        if entity.quantity_mode != QuantityMode.ALL or len(members) <= 1:
            concrete.append(entity); aliases[entity.entity_id] = [entity.entity_id]; continue
        aliases[entity.entity_id] = []
        for index, object_id in enumerate(members, 1):
            item = next((value for value in registry.objects if value.object_id == object_id or value.body_name == object_id), None)
            if item is None:
                raise ValueError(f"grounding_failed: set member is missing from scene registry: {object_id}")
            cid = f"{entity.entity_id}__{index:02d}"
            aliases[entity.entity_id].append(cid)
            concrete.append(entity.model_copy(update={"entity_id": cid, "semantic_entity_id": entity.entity_id, "object_id": object_id, "body_name": item.body_name}))
    expanded: list[Operation] = []
    expansion: dict[str, dict] = {}
    parent_children: dict[str, list[str]] = {}
    parent_dependencies: dict[str, list[str]] = {op.operation_id: list(op.depends_on) for op in task.operations}
    grouped = {}
    for op in task.operations:
        actor = op.source or op.target
        if actor in aliases and len(aliases.get(actor, [])) > 1:
            grouped.setdefault(actor, []).append(op)
    pairwise_index = {}
    for source, ops in grouped.items():
        distinct_destinations = len({op.destination for op in ops}) == len(ops) and all(op.destination for op in ops)
        if len(aliases[source]) > 1 and len(ops) > 1 and distinct_destinations and not pairwise:
            raise ValueError("assignment_cardinality_mismatch: explicit pairwise correspondence is required")
        # Repeated operations on a set (e.g. right, then front) must broadcast
        # to the same members in every phase.  Only explicit pairwise evidence
        # may consume distinct members across distinct destinations.
        if pairwise and len(ops) == len(aliases[source]) and len({op.destination for op in ops}) == len(ops) and all(op.destination for op in ops):
            pairwise_index.update({id(op): aliases[source][index] for index, op in enumerate(ops)})
    for op in task.operations:
        actor = op.source or op.target
        actor_members = [pairwise_index[id(op)]] if id(op) in pairwise_index else (aliases.get(actor, [actor]) if actor else [None])
        sources = actor_members if op.source else [None]
        destinations = aliases.get(op.destination, [op.destination]) if op.destination else [None]
        assignment_plan = None
        if op.source is None and len(actor_members) > 1:
            pairs = [(None, destinations[0]) for _ in actor_members]
            mode = "broadcast"
        else:
            assignment_plan = resolve_assignments(
                op.operation_id,
                [value for value in sources if value],
                [value for value in destinations if value],
                pairwise=pairwise,
            )
            pairs = [(assignment.source_object, assignment.destination_object) for assignment in assignment_plan.assignments]
            mode = assignment_plan.mode.value
        children = []
        for index, (source, destination) in enumerate(pairs, 1):
            target = aliases.get(op.target, [op.target])[0] if op.target else None
            reference = aliases.get(op.reference, [op.reference])[0] if op.reference else None
            if op.source is None and actor_members:
                target = actor_members[index - 1]
            child = op.model_copy(update={"operation_id": f"{op.operation_id}__{index:02d}" if len(pairs) > 1 else op.operation_id,
                                          "source": source, "destination": destination, "target": target, "reference": reference,
                                          "depends_on": []})
            expanded.append(child); children.append({"operation_id": child.operation_id, "source_object": _object_for(concrete, source), "destination_object": _object_for(concrete, destination)})
            parent_children.setdefault(op.operation_id, []).append(child.operation_id)
        expansion[op.operation_id] = {"mode": mode, "children": children}
    # Keep parent operation dependencies serial after expansion.
    normalized = []
    for op in expanded:
        parent = op.operation_id.split("__", 1)[0]
        deps = []
        parent_deps = parent_dependencies.get(parent, [])
        for dependency in parent_deps:
            deps.extend(parent_children.get(dependency, [dependency])[-1:])
        siblings = parent_children.get(parent, [])
        if op.operation_id in siblings and siblings.index(op.operation_id) > 0:
            deps.append(siblings[siblings.index(op.operation_id) - 1])
        normalized.append(op.model_copy(update={"depends_on": list(dict.fromkeys(deps))}))
    return task.model_copy(update={"entities": concrete, "operations": normalized}), expansion

def _object_for(entities, entity_id):
    if not entity_id: return None
    return next((e.object_id for e in entities if e.entity_id == entity_id), entity_id)
