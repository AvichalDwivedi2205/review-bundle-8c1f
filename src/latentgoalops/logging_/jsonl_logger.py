"""JSONL log writer."""

from __future__ import annotations

import json
from pathlib import Path

from latentgoalops.logging_.schemas import EpisodeSummary, StepLog


class JsonlLogger:
    """Small JSONL writer for reproducible experiments."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_step(self, step_log: StepLog) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step_log.model_dump(mode="json")) + "\n")

    def write_episode(self, episode_summary: EpisodeSummary) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode_summary.model_dump(mode="json")) + "\n")

