"""Configuration loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config from the repository config directory."""
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

