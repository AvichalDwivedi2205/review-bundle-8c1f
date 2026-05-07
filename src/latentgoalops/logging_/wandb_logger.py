"""Optional Weights & Biases experiment logging."""

from __future__ import annotations

from typing import Any

from latentgoalops.logging_.schemas import EpisodeSummary, StepLog

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class WandbLogger:
    """Thin wrapper around a W&B run."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        run_name: str,
        group: str | None,
        tags: list[str],
        config: dict[str, Any],
    ) -> None:
        self.enabled = enabled and wandb is not None
        self._run = None
        if not self.enabled:
            return
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            config=config,
            reinit="finish_previous",
        )
        wandb.define_metric("episode_index")
        wandb.define_metric("step_global")
        wandb.define_metric("episode/*", step_metric="episode_index")
        wandb.define_metric("step/*", step_metric="step_global")

    def log_step(self, step_log: StepLog, episode_index: int, step_global: int) -> None:
        """Log a per-step point to W&B."""
        if not self.enabled or self._run is None:
            return
        payload = {
            "episode_index": episode_index,
            "step_global": step_global,
            "step/reward": step_log.reward,
            "step/done": int(step_log.done),
            "step/elapsed_seconds": step_log.elapsed_seconds,
            "step/input_tokens": step_log.provider_usage.get("input_tokens", 0),
            "step/output_tokens": step_log.provider_usage.get("output_tokens", 0),
            "step/cost_usd": step_log.provider_usage.get("cost_usd", 0.0),
            "step/parse_fallback": int(bool(step_log.provider_usage.get("parse_fallback", False))),
            "step/seed": step_log.seed,
            "step/task_id": step_log.task_id,
            "step/policy": step_log.policy,
            "step/model_name": step_log.model_name,
            "step/step_index": step_log.step_index,
        }
        self._run.log(payload)

    def log_episode(self, episode_summary: EpisodeSummary, episode_index: int) -> None:
        """Log an episode summary to W&B."""
        if not self.enabled or self._run is None:
            return
        payload = {
            "episode_index": episode_index,
            "episode/score": episode_summary.score,
            "episode/total_reward": episode_summary.total_reward,
            "episode/total_steps": episode_summary.total_steps,
            "episode/elapsed_seconds": episode_summary.elapsed_seconds,
            "episode/input_tokens": episode_summary.provider_usage.get("input_tokens", 0),
            "episode/output_tokens": episode_summary.provider_usage.get("output_tokens", 0),
            "episode/cost_usd": episode_summary.provider_usage.get("cost_usd", 0.0),
            "episode/seed": episode_summary.seed,
            "episode/task_id": episode_summary.task_id,
            "episode/policy": episode_summary.policy,
            "episode/model_name": episode_summary.model_name,
            "episode/parse_fallback": int(bool(episode_summary.provider_usage.get("parse_fallback", False))),
        }
        if "progress_fraction" in episode_summary.metadata:
            payload["episode/progress_fraction"] = episode_summary.metadata["progress_fraction"]
        if "eta_seconds" in episode_summary.metadata:
            payload["episode/eta_seconds"] = episode_summary.metadata["eta_seconds"]
        for key, value in episode_summary.grade.get("sub_scores", {}).items():
            payload[f"episode/subscore/{key}"] = value
        self._run.log(payload)

    def log_summary(self, summary: dict[str, float], extra: dict[str, Any] | None = None) -> None:
        """Write terminal summary metrics."""
        if not self.enabled or self._run is None:
            return
        for key, value in summary.items():
            self._run.summary[f"summary/{key}"] = value
        for key, value in (extra or {}).items():
            self._run.summary[key] = value

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self._run is not None:
            self._run.finish()
