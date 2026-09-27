"""Small deterministic interior-slot allocator for repeated placements."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class PlacementSlot:
    slot_id: str
    local_position: tuple[float, float, float]

def allocate_interior_slots(container_min, container_max, object_dimensions, occupied=(), gap: float = 0.02, max_overhang: float = 0.0) -> list[PlacementSlot]:
    """Return non-overlapping floor-grid slots in stable order.

    Bounds and dimensions are full extents.  The allocator deliberately uses
    a conservative 2-D floor grid.  ``max_overhang`` is an explicit bounded
    policy for irregular meshes; with its default of zero every slot remains
    inside the authored interior region.
    """
    min_x, min_y, min_z = map(float, container_min)
    max_x, max_y, max_z = map(float, container_max)
    max_overhang = max(0.0, float(max_overhang))
    # Some generated open meshes (notably a curved banana) have an axis
    # aligned bounding box slightly longer than the authored cavity although
    # their contact footprint can rest safely with a small, explicit
    # overhang.  The policy is bounded and opt-in; authored containers keep
    # strict footprint containment by default.
    floor_min_x = min_x - max_overhang
    floor_max_x = max_x + max_overhang
    floor_min_y = min_y - max_overhang
    floor_max_y = max_y + max_overhang
    width, depth, height = map(float, object_dimensions)
    usable_x = floor_max_x - floor_min_x
    usable_y = floor_max_y - floor_min_y
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
    # Always try the geometric centre first when the footprint fits.  An
    # even-by-even floor grid otherwise starts at a corner, which is legal
    # for packing but leaves a gripper's fingers rubbing the container wall
    # during the first grasp.  The centre candidate is especially important
    # for a single object; subsequent objects still use the deterministic
    # grid to remain separated.
    candidates = [(center_x, center_y)]
    # Prefer cardinal offsets before corner/row combinations.  A diagonal
    # grid point can overlap the centre payload in both axes even when its
    # Euclidean distance looks large; axis-aligned neighbours are easier for
    # the gripper to approach and leave a predictable clearance margin.
    candidates.extend([(center_x - step_x, center_y), (center_x + step_x, center_y)])
    candidates.extend([(center_x, center_y - step_y), (center_x, center_y + step_y)])
    candidates.extend((x, y) for y in ys for x in xs)
    # Cardinal/centre candidates are generated independently of the packed
    # grid.  Keep only footprints that fit the authored interior; otherwise a
    # narrow container could receive a slot whose centre is inside but whose
    # gripper/object footprint protrudes through a wall.
    fit_min_x = floor_min_x + width / 2
    fit_max_x = floor_max_x - width / 2
    fit_min_y = floor_min_y + depth / 2
    fit_max_y = floor_max_y - depth / 2
    candidates = [
        (x, y)
        for x, y in candidates
        if fit_min_x - 1e-9 <= x <= fit_max_x + 1e-9
        and fit_min_y - 1e-9 <= y <= fit_max_y + 1e-9
    ]
    candidates = sorted(
        set(candidates),
        key=lambda value: (abs(value[1] - center_y) + abs(value[0] - center_x), abs(value[1] - center_y), abs(value[0] - center_x), value[1], value[0]),
    )
    occupied = list(occupied)
    slots = []
    index = 1
    for x, y in candidates:
        candidate = (x, y, min_z + height / 2)
        box = ((x - width / 2, x + width / 2), (y - depth / 2, y + depth / 2))
        if not any(_overlap(box, other) for other in occupied):
            slots.append(PlacementSlot(f"interior_slot_{index}", candidate))
            index += 1
    # Do not silently stack payloads in v1.  A full floor grid is an explicit
    # capacity failure; a future stacking resolver can add 3-D layers with
    # the required stability and collision checks.
    return slots

def _overlap(a, b):
    return not (a[0][1] <= b[0][0] or b[0][1] <= a[0][0] or a[1][1] <= b[1][0] or b[1][1] <= a[1][0])
