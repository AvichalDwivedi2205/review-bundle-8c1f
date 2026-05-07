"""Aggregate JSONL experiment logs into tabular summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _episode_identity(record: dict) -> tuple[str, str, int, str, str]:
    return (
        str(record.get("run_id", "")),
        str(record.get("task_id", "")),
        int(record.get("seed", 0)),
        str(record.get("policy", "")),
        str(record.get("model_name", "")),
    )


def _step_identity(record: dict) -> tuple[str, str, int, str, str, int]:
    return (
        str(record.get("run_id", "")),
        str(record.get("task_id", "")),
        int(record.get("seed", 0)),
        str(record.get("policy", "")),
        str(record.get("model_name", "")),
        int(record.get("step_index", 0)),
    )


def _decision_only_score(payload: dict) -> float | None:
    task_id = str(payload.get("task_id", ""))
    sub_scores = payload.get("grade", {}).get("sub_scores", {}) or {}
    component_weights = {
        "task3_startup_week": {
            "final_utility": 0.50,
            "adaptation": 0.25,
            "coherence": 0.15,
            "constraints": 0.10,
        },
        "task6_incident_response_week": {
            "final_utility": 0.4634,
            "adaptation": 0.2927,
            "coherence": 0.1220,
            "constraints": 0.1220,
        },
        "task7_quarterly_headcount_plan": {
            "final_utility": 0.6364,
            "adaptation": 0.2614,
            "coherence": 0.0568,
            "constraints": 0.0454,
        },
    }.get(task_id)
    if component_weights is None:
        return None
    present = [
        (name, float(sub_scores[name]), weight)
        for name, weight in component_weights.items()
        if name in sub_scores and sub_scores[name] is not None
    ]
    if not present:
        return None
    total_weight = sum(weight for _, _, weight in present) or 1.0
    return sum(value * weight for _, value, weight in present) / total_weight


def _infer_policy(payload: dict) -> str:
    policy = payload.get("policy") or payload.get("metadata", {}).get("policy")
    if policy:
        return str(policy)
    model_name = str(payload.get("model_name", ""))
    if model_name in {"random", "heuristic", "oracle"}:
        return model_name
    return "model"


def _experiment_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata", {}) or {}
    experiment_config = metadata.get("experiment_config", {}) or {}
    return {
        "benchmark_runtime_version": metadata.get("benchmark_runtime_version"),
        "ablation": (
            "full"
            if not experiment_config
            else "+".join(
                part
                for part, enabled in [
                    ("no_shift", not experiment_config.get("enable_hidden_shift", True)),
                    ("no_delay", not experiment_config.get("enable_delayed_effects", True)),
                    ("no_ledger", not experiment_config.get("expose_decision_ledger", True)),
                    ("sparse", experiment_config.get("reward_mode", "shaped") == "sparse"),
                ]
                if enabled
            )
            or "full"
        ),
        "reward_mode": experiment_config.get("reward_mode", "shaped"),
        "enable_hidden_shift": bool(experiment_config.get("enable_hidden_shift", True)),
        "enable_delayed_effects": bool(experiment_config.get("enable_delayed_effects", True)),
        "expose_decision_ledger": bool(experiment_config.get("expose_decision_ledger", True)),
    }


def _jsonl_paths(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        return sorted(candidate.rglob("runs.jsonl"))
    return []


def load_run_records(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load step and episode records from one file or a directory tree of logs."""
    step_records: dict[tuple[str, str, int, str, str, int], dict] = {}
    episode_records: dict[tuple[str, str, int, str, str], dict] = {}
    duplicate_step_rows = 0
    duplicate_episode_rows = 0
    source_paths = _jsonl_paths(path)
    for jsonl_path in source_paths:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if "total_steps" in payload and "score" in payload:
                    flat = {
                        **payload,
                        "policy": _infer_policy(payload),
                        "cost_usd": payload.get("provider_usage", {}).get("cost_usd", 0.0),
                        "input_tokens": payload.get("provider_usage", {}).get("input_tokens", 0),
                        "output_tokens": payload.get("provider_usage", {}).get("output_tokens", 0),
                        "parse_fallback": int(bool(payload.get("provider_usage", {}).get("parse_fallback", False))),
                        "parse_repaired": int(bool(payload.get("provider_usage", {}).get("parse_repaired", False))),
                        "heuristic_rescue": int(bool(payload.get("provider_usage", {}).get("heuristic_rescue", False))),
                        "empty_fallback": int(bool(payload.get("provider_usage", {}).get("empty_fallback", False))),
                        "response_cap_hit": int(bool(payload.get("provider_usage", {}).get("response_cap_hit", False))),
                        "invalid_response_excerpt": payload.get("provider_usage", {}).get("invalid_response_excerpt"),
                        "finish_reason_counts": payload.get("provider_usage", {}).get("finish_reason_counts", {}) or {},
                        "task2_visible_floor_applied": int(
                            bool(payload.get("provider_usage", {}).get("task2_visible_floor_applied", False))
                        ),
                        "strict_episode": int(bool(payload.get("metadata", {}).get("strict_episode", False))),
                        "rescued_episode": int(bool(payload.get("metadata", {}).get("rescued_episode", False))),
                        "assisted_episode": int(bool(payload.get("metadata", {}).get("assisted_episode", False))),
                        "empty_fallback_episode": int(bool(payload.get("metadata", {}).get("empty_fallback_episode", False))),
                        "heuristic_rescue_episode": int(bool(payload.get("metadata", {}).get("heuristic_rescue_episode", False))),
                        "paper_eval": int(bool(payload.get("metadata", {}).get("paper_eval", False))),
                        "hidden_shift_present": int(bool(payload.get("metadata", {}).get("hidden_shift_present", False))),
                        "hidden_shift_step": payload.get("metadata", {}).get("hidden_shift_step"),
                        "hidden_shift_type": payload.get("metadata", {}).get("hidden_shift_type"),
                        "episodes_completed": payload.get("metadata", {}).get("episodes_completed"),
                        "episodes_total": payload.get("metadata", {}).get("episodes_total"),
                        "progress_fraction": payload.get("metadata", {}).get("progress_fraction"),
                        "adaptation_scored": int(bool(payload.get("grade", {}).get("details", {}).get("adaptation_scored", False))),
                        "decision_only_score": _decision_only_score(payload),
                        "belief_tracking_score": payload.get("grade", {}).get("sub_scores", {}).get("belief_tracking"),
                        **_experiment_metadata(payload),
                    }
                    for key, value in payload.get("grade", {}).get("sub_scores", {}).items():
                        flat[f"subscore_{key}"] = value
                    flat["source_jsonl"] = str(jsonl_path)
                    episode_key = _episode_identity(flat)
                    if episode_key in episode_records:
                        duplicate_episode_rows += 1
                    episode_records[episode_key] = flat
                elif "step_index" in payload and "reward" in payload:
                    flat = {
                        **payload,
                        "policy": _infer_policy(payload),
                        "cost_usd": payload.get("provider_usage", {}).get("cost_usd", 0.0),
                        "input_tokens": payload.get("provider_usage", {}).get("input_tokens", 0),
                        "output_tokens": payload.get("provider_usage", {}).get("output_tokens", 0),
                        "parse_fallback": int(bool(payload.get("provider_usage", {}).get("parse_fallback", False))),
                        "parse_repaired": int(bool(payload.get("provider_usage", {}).get("parse_repaired", False))),
                        "heuristic_rescue": int(bool(payload.get("provider_usage", {}).get("heuristic_rescue", False))),
                        "empty_fallback": int(bool(payload.get("provider_usage", {}).get("empty_fallback", False))),
                        "response_cap_hit": int(bool(payload.get("provider_usage", {}).get("response_cap_hit", False))),
                        "finish_reason": payload.get("provider_usage", {}).get("finish_reason"),
                        "task2_visible_floor_applied": int(
                            bool(payload.get("provider_usage", {}).get("task2_visible_floor_applied", False))
                        ),
                        "strict_step": int(bool(payload.get("metadata", {}).get("strict_step", False))),
                        "assisted_step": int(bool(payload.get("metadata", {}).get("assisted_step", False))),
                        "paper_eval": int(bool(payload.get("metadata", {}).get("paper_eval", False))),
                        **_experiment_metadata(payload),
                    }
                    flat["source_jsonl"] = str(jsonl_path)
                    step_key = _step_identity(flat)
                    if step_key in step_records:
                        duplicate_step_rows += 1
                    step_records[step_key] = flat
    step_df = pd.DataFrame(list(step_records.values()))
    episode_df = pd.DataFrame(list(episode_records.values()))
    step_df.attrs["duplicate_step_rows"] = duplicate_step_rows
    step_df.attrs["duplicate_episode_rows"] = duplicate_episode_rows
    step_df.attrs["source_path_count"] = len(source_paths)
    episode_df.attrs["duplicate_step_rows"] = duplicate_step_rows
    episode_df.attrs["duplicate_episode_rows"] = duplicate_episode_rows
    episode_df.attrs["source_path_count"] = len(source_paths)
    return step_df, episode_df


def load_episode_summaries(path: str | Path) -> pd.DataFrame:
    """Load terminal episode rows from run logs."""
    _, episodes = load_run_records(path)
    return episodes


def load_step_logs(path: str | Path) -> pd.DataFrame:
    """Load step rows from run logs."""
    steps, _ = load_run_records(path)
    return steps


def summarize_scores(path: str | Path) -> pd.DataFrame:
    """Compute mean/std scores by model, policy, and task."""
    df = load_episode_summaries(path)
    if df.empty:
        return df
    return (
        df.groupby(["policy", "model_name", "task_id"], as_index=False)["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def summarize_costs(path: str | Path) -> pd.DataFrame:
    """Compute cost and token usage by model, policy, and task."""
    df = load_episode_summaries(path)
    if df.empty:
        return df
    return (
        df.groupby(["policy", "model_name", "task_id"], as_index=False)[["cost_usd", "input_tokens", "output_tokens"]]
        .mean()
        .reset_index()
    )


def summarize_parse_fallbacks(path: str | Path) -> pd.DataFrame:
    """Compute parse fallback rates by policy/model/task."""
    df = load_episode_summaries(path)
    if df.empty:
        return df
    columns = ["parse_fallback", "parse_repaired", "rescued_episode"]
    if "empty_fallback_episode" in df.columns:
        columns.append("empty_fallback_episode")
    if "response_cap_hit" in df.columns:
        columns.append("response_cap_hit")
    return (
        df.groupby(["policy", "model_name", "task_id"], as_index=False)[columns]
        .mean()
        .reset_index()
    )
