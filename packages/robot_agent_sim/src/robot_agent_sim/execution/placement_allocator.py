"""Small deterministic interior-slot allocator for repeated placements."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PlacementSlot:
    slot_id: str
    local_position: tuple[float, float, float]

def allocate_interior_slots(container_min, container_max, object_dimensions, occupied=(), gap: float = 0.01) -> list[PlacementSlot]:
    """Return non-overlapping floor-grid slots in stable order.

    Bounds and dimensions are full extents.  The allocator deliberately uses
    a conservative 2-D floor grid; it never invents a slot outside the
    authored interior region.
    """
    min_x, min_y, min_z = map(float, container_min)
    max_x, max_y, max_z = map(float, container_max)
    width, depth, height = map(float, object_dimensions)
    usable_x = max_x - min_x
    usable_y = max_y - min_y
    columns = max(0, math.floor((usable_x + gap) / (width + gap)))
    rows = max(0, math.floor((usable_y + gap) / (depth + gap)))
    if columns == 0 or rows == 0 or height > max_z - min_z + 1e-9:
        return []
    step_x = width + gap
    step_y = depth + gap
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    xs = [center_x + (index - (columns - 1) / 2) * step_x for index in range(columns)]
    ys = [center_y + (index - (rows - 1) / 2) * step_y for index in range(rows)]
    # Put central rows first, then use stable left-to-right order.  This keeps
    # the gripper away from container walls whenever a central row exists.
    candidates = sorted(((x, y) for y in ys for x in xs), key=lambda value: (abs(value[1] - center_y), abs(value[0] - center_x), value[1], value[0]))
    occupied = list(occupied)
    slots = []
    index = 1
    for x, y in candidates:
        candidate = (x, y, min_z + height / 2)
        box = ((x - width / 2, x + width / 2), (y - depth / 2, y + depth / 2))
        if not any(_overlap(box, other) for other in occupied):
            slots.append(PlacementSlot(f"interior_slot_{index}", candidate))
            index += 1
    # Small authored containers sometimes have only one floor cell while the
    # runtime still permits stacking (for example a fruit whose mesh is wider
    # than the nominal basket footprint).  Use a deterministic vertical slot
    # when there is room above the occupied floor cell instead of reusing the
    # same 3-D placement and silently overlapping it.
    if not slots and occupied and len(occupied) < 2 and max_z - min_z >= 2 * height:
        slots.append(PlacementSlot("interior_slot_vertical", (center_x, center_y, min_z + 1.5 * height)))
    return slots

def _overlap(a, b):
    return not (a[0][1] <= b[0][0] or b[0][1] <= a[0][0] or a[1][1] <= b[1][0] or b[1][1] <= a[1][0])
