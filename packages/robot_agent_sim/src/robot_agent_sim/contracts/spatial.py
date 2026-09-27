from __future__ import annotations

from enum import StrEnum


class SpatialRelationType(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    UP = "up"
    DOWN = "down"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    ON = "on"
    NEAR = "near"
    NEAREST = "nearest"
    FARTHEST = "farthest"
    LEFTMOST = "leftmost"
    RIGHTMOST = "rightmost"
    FRONTMOST = "frontmost"
    BACKMOST = "backmost"
    HIGHEST = "highest"
    LOWEST = "lowest"
