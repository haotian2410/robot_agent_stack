"""Compatibility names for the protocol-owned interaction registry."""

from robot_agent_protocol import InteractionRegistry as SceneRegistry
from robot_agent_protocol import InteractionRegistryError as SceneRegistryError

__all__ = ["SceneRegistry", "SceneRegistryError"]
