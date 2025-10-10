"""Core package for The Assistant project."""

from __future__ import annotations

from importlib import metadata

from .memory_vessel import MemoryContainmentVessel, MemoryFragment

try:  # pragma: no cover - runtime metadata lookup
    __version__ = metadata.version("the-assistant")
except metadata.PackageNotFoundError:  # pragma: no cover - local development fallback
    __version__ = "0.1.0"

__all__ = ["MemoryContainmentVessel", "MemoryFragment", "__version__"]
