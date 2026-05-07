"""Shared experiment configuration for paper-grade evaluations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Configuration knobs used for ablations and paper experiments."""

    enable_hidden_shift: bool = True
    enable_delayed_effects: bool = True
    expose_decision_ledger: bool = True
    reward_mode: Literal["shaped", "sparse"] = "shaped"
    task3_horizon_override: int | None = None
    scenario_split: Literal["core", "heldout"] = "core"

    def slug(self) -> str:
        """Stable short name for outputs and tables."""
        parts = []
        if not self.enable_hidden_shift:
            parts.append("no_shift")
        if not self.enable_delayed_effects:
            parts.append("no_delay")
        if not self.expose_decision_ledger:
            parts.append("no_ledger")
        if self.reward_mode != "shaped":
            parts.append(self.reward_mode)
        if self.task3_horizon_override is not None:
            parts.append(f"h{self.task3_horizon_override}")
        if self.scenario_split != "core":
            parts.append(self.scenario_split)
        return "+".join(parts) if parts else "full"
