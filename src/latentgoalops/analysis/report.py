"""CLI to generate a paper-friendly plot bundle from run logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentgoalops.analysis.aggregate import load_run_records
from latentgoalops.analysis.plots import (
    plot_adaptation_distribution,
    plot_coherence_vs_score,
    plot_cost_vs_score,
    plot_heatmap,
    plot_oracle_gap,
    plot_parse_fallbacks,
    plot_score_distributions,
    plot_step_reward_trajectories,
    plot_task_scores,
    plot_token_usage,
)
from latentgoalops.analysis.tables import write_summary_tables


def _runtime_versions(steps, episodes) -> list[str]:
    versions: set[str] = set()
    for frame in (episodes, steps):
        if not frame.empty and "benchmark_runtime_version" in frame.columns:
            versions.update(
                str(value) for value in frame["benchmark_runtime_version"].dropna().tolist() if str(value).strip()
            )
    return sorted(versions)


def _validate_report_inputs(steps, episodes) -> None:
    duplicate_episode_rows = int(episodes.attrs.get("duplicate_episode_rows", 0))
    duplicate_step_rows = int(steps.attrs.get("duplicate_step_rows", 0))
    if duplicate_episode_rows > 0 or duplicate_step_rows > 0:
        raise ValueError(
            "Duplicate step/episode rows detected in the input logs. "
            "Use a clean output directory or deduplicated run tree before generating a paper report."
        )
    if not episodes.empty and "paper_eval" in episodes.columns:
        paper_values = sorted({int(value) for value in episodes["paper_eval"].dropna().tolist()})
        if paper_values == [0, 1]:
            raise ValueError(
                "Mixed paper-eval and non-paper-eval episodes detected in the same report input. "
                "Generate reports from a single evaluation protocol at a time."
            )
    if len(_runtime_versions(steps, episodes)) > 1:
        raise ValueError(
            "Mixed benchmark runtime versions detected in the same report input. "
            "Generate reports from a single benchmark runtime version at a time."
        )
    if not episodes.empty and "episodes_total" in episodes.columns:
        incomplete_runs: list[str] = []
        for (run_id, model_name, policy), group in episodes.groupby(["run_id", "model_name", "policy"], dropna=False):
            totals = [int(value) for value in group["episodes_total"].dropna().tolist() if int(value) > 0]
            if not totals:
                continue
            expected = max(totals)
            observed = int(len(group))
            if observed < expected:
                incomplete_runs.append(f"{run_id}:{model_name}:{policy} ({observed}/{expected})")
        if incomplete_runs:
            raise ValueError(
                "Incomplete runs detected in the input logs. "
                "Resume the interrupted sweeps or remove partial outputs before generating a paper report: "
                + ", ".join(incomplete_runs[:5])
            )


def _primary_episode_slice(episodes):
    if episodes.empty:
        return episodes, "all"
    if "paper_eval" in episodes.columns and int(episodes["paper_eval"].min()) == 1 and int(episodes["paper_eval"].max()) == 1:
        return episodes, "paper_eval_all"
    if "rescued_episode" in episodes.columns and float(episodes["rescued_episode"].sum()) > 0:
        return episodes, "all_with_assistance"
    if "empty_fallback_episode" in episodes.columns and float(episodes["empty_fallback_episode"].sum()) > 0:
        return episodes, "all_with_invalid_actions"
    if "strict_episode" in episodes.columns and float(episodes["strict_episode"].sum()) == float(len(episodes)):
        return episodes, "all_native"
    return episodes, "all"


def _primary_step_slice(steps):
    if steps.empty:
        return steps, "all"
    if "paper_eval" in steps.columns and int(steps["paper_eval"].min()) == 1 and int(steps["paper_eval"].max()) == 1:
        return steps, "paper_eval_all"
    if "assisted_step" in steps.columns and float(steps["assisted_step"].sum()) > 0:
        return steps, "all_with_assistance"
    if "strict_step" in steps.columns and float(steps["strict_step"].sum()) == float(len(steps)):
        return steps, "all_native"
    return steps, "all"


def generate_report(input_path: str, output_dir: str) -> None:
    """Generate a plot bundle from one log file or a directory tree of runs."""
    steps, episodes = load_run_records(input_path)
    _validate_report_inputs(steps, episodes)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    primary_episodes, episode_slice = _primary_episode_slice(episodes)
    primary_steps, step_slice = _primary_step_slice(steps)
    runtime_versions = _runtime_versions(primary_steps, primary_episodes)

    plot_task_scores(primary_episodes, output_root / "task_scores_bar.png")
    plot_score_distributions(primary_episodes, output_root / "score_distributions_box.png")
    plot_heatmap(primary_episodes, output_root / "score_heatmap.png")
    plot_cost_vs_score(primary_episodes, output_root / "cost_vs_score.png")
    plot_step_reward_trajectories(primary_steps, output_root / "reward_trajectories.png")
    plot_token_usage(primary_episodes, output_root / "token_usage.png")
    plot_parse_fallbacks(episodes, output_root / "parse_fallbacks.png")
    plot_adaptation_distribution(primary_episodes, output_root / "adaptation_distribution.png")
    plot_coherence_vs_score(primary_episodes, output_root / "coherence_vs_score.png")
    plot_oracle_gap(primary_episodes, output_root / "oracle_gap.png")
    write_summary_tables(primary_episodes, output_root)
    (output_root / "report_metadata.json").write_text(
        json.dumps(
            {
                "episode_slice": episode_slice,
                "step_slice": step_slice,
                "total_episode_rows": int(len(episodes)),
                "primary_episode_rows": int(len(primary_episodes)),
                "total_step_rows": int(len(steps)),
                "primary_step_rows": int(len(primary_steps)),
                "rescued_episode_rate": float(episodes["rescued_episode"].mean()) if "rescued_episode" in episodes else 0.0,
                "empty_fallback_episode_rate": float(episodes["empty_fallback_episode"].mean())
                if "empty_fallback_episode" in episodes
                else 0.0,
                "response_cap_hit_episode_rate": float(episodes["response_cap_hit"].mean())
                if "response_cap_hit" in episodes
                else 0.0,
                "paper_eval_fraction": float(episodes["paper_eval"].mean()) if "paper_eval" in episodes else 0.0,
                "benchmark_runtime_versions": runtime_versions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs", help="A runs.jsonl file or a directory containing many run outputs.")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()
    generate_report(args.input, args.output_dir)
    print(f"Generated plots under {args.output_dir}")


if __name__ == "__main__":
    main()
