"""Conservative operation deduplication for duplicate entity decomposition."""
from __future__ import annotations


def operation_signature(operation):
    return (
        operation.type,
        operation.source,
        operation.destination,
        operation.target,
        operation.reference,
        operation.motion_direction,
        operation.distance_m,
        operation.motion_scale,
    )


def collapse_duplicate_operations(operations):
    """Collapse only exact duplicates; repeated user actions remain intact."""
    result = []
    seen = set()
    collapsed = []
    for operation in operations:
        signature = operation_signature(operation)
        if signature in seen:
            collapsed.append(operation)
            continue
        seen.add(signature)
        result.append(operation)
    return result, collapsed


__all__ = ["operation_signature", "collapse_duplicate_operations"]
