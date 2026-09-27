"""Deterministic semantic-set to operation-member assignment."""
from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, ConfigDict

class AssignmentMode(StrEnum):
    SINGLE = "single"
    BROADCAST = "broadcast"
    PAIRWISE = "pairwise"
    RELATION_MATCHED = "relation_matched"
    REPEAT_ALL = "repeat_all"

class OperationAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str
    source_object: str | None = None
    destination_object: str | None = None

class OperationAssignmentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: AssignmentMode
    assignments: list[OperationAssignment]

def resolve_assignments(
    operation_id: str,
    source_members: list[str],
    destination_members: list[str],
    *,
    pairwise: bool = False,
    relation_pairs: list[tuple[str, str]] | None = None,
) -> OperationAssignmentPlan:
    """Resolve one semantic operation without guessing cardinality.

    A many-to-one operation is broadcast. Equal cardinalities require explicit
    pairwise evidence; otherwise the caller gets a cardinality error instead
    of silently choosing an assignment.
    """
    sources = list(source_members); destinations = list(destination_members)
    if relation_pairs is not None:
        pairs = [(source, destination) for source, destination in relation_pairs]
        if not pairs or any(source not in sources or destination not in destinations for source, destination in pairs):
            raise ValueError("grounding_assignment_ambiguous: authored relation pair is invalid")
        return OperationAssignmentPlan(
            mode=AssignmentMode.RELATION_MATCHED,
            assignments=[
                OperationAssignment(operation_id=f"{operation_id}__{index:02d}", source_object=source, destination_object=destination)
                for index, (source, destination) in enumerate(pairs, 1)
            ],
        )
    if not sources and not destinations:
        return OperationAssignmentPlan(mode=AssignmentMode.SINGLE, assignments=[OperationAssignment(operation_id=operation_id)])
    if len(sources) == 1 and len(destinations) <= 1:
        return OperationAssignmentPlan(mode=AssignmentMode.SINGLE, assignments=[OperationAssignment(operation_id=operation_id, source_object=sources[0] if sources else None, destination_object=destinations[0] if destinations else None)])
    if len(sources) > 1 and len(destinations) == 1:
        return OperationAssignmentPlan(mode=AssignmentMode.BROADCAST, assignments=[OperationAssignment(operation_id=f"{operation_id}__{i:02d}", source_object=source, destination_object=destinations[0]) for i, source in enumerate(sources, 1)])
    if len(sources) == len(destinations) and pairwise:
        return OperationAssignmentPlan(mode=AssignmentMode.PAIRWISE, assignments=[OperationAssignment(operation_id=f"{operation_id}__{i:02d}", source_object=source, destination_object=destination) for i, (source, destination) in enumerate(zip(sources, destinations), 1)])
    raise ValueError("assignment_cardinality_mismatch: explicit pairwise correspondence is required")
