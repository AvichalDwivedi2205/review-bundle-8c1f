"""Shared template loading helpers for scenario-family splits."""

from __future__ import annotations

from latentgoalops.server.config import load_config


def _matches_split(payload: dict, split: str) -> bool:
    tag = str(payload.get("split", "core"))
    wanted = "heldout" if split == "heldout" else "core"
    return tag == wanted


def load_initiative_effects(split: str = "core") -> dict[str, dict]:
    """Return initiative templates for the requested scenario split."""
    templates = load_config("transitions.yaml")["initiative_effects"]
    return {
        name: payload
        for name, payload in templates.items()
        if _matches_split(payload, split)
    }


def load_events(split: str = "core") -> dict[str, dict]:
    """Return event templates for the requested scenario split."""
    templates = load_config("events.yaml")["events"]
    return {
        name: payload
        for name, payload in templates.items()
        if _matches_split(payload, split)
    }
