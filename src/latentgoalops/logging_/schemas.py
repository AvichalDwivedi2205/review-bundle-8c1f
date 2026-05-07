"""Structured logging schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class StepLog(BaseModel):
    """Per-step execution log."""

    timestamp: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    run_id: str
    task_id: str
    seed: int
    policy: str
    model_name: str
    step_index: int
    action: dict[str, Any]
    reward: float
    done: bool
    observation: dict[str, Any]
    provider_usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeSummary(BaseModel):
    """Terminal episode summary."""

    timestamp: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    run_id: str
    task_id: str
    seed: int
    policy: str
    model_name: str
    score: float
    total_reward: float
    total_steps: int
    provider_usage: dict[str, Any] = Field(default_factory=dict)
    grade: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointState(BaseModel):
    """Resumable batch-run state."""

    run_id: str
    model_name: str
    policy: str = "heuristic"
    tasks: list[str]
    seeds: list[int]
    started_at: str = Field(default_factory=utc_now_iso)
    next_index: int = 0
    cumulative_cost_usd: float = 0.0
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    failure_mode: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
