"""Generate paper-oriented tables from the frozen paper bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentgoalops.analysis.aggregate import load_run_records
from latentgoalops.analysis.stats import bootstrap_mean_ci
from latentgoalops.paper.run_paper_suite import generate_paper_suite


SUBSCORE_COLUMNS = [
    "subscore_final_utility",
    "subscore_adaptation",
    "subscore_coherence",
    "subscore_constraints",
]

MIN_BOOTSTRAP_SAMPLE_COUNT = 3

TASK_ID_ALIASES = {
    "task1_feedback_triage": "task1_feedback",
    "task2_roadmap_priority": "task2_prioritization",
}

PRIMARY_BENCHMARK_ROLES = {"primary", "benchmark"}
EXTERNAL_ANCHOR_ROLES = {"external_anchor"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_manifest(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    suite_result = generate_paper_suite(config_path)
    root = _repo_root()
    metadata_dir = Path(suite_result["metadata_dir"])
    config = json.loads((metadata_dir / "paper_protocol_resolved.json").read_text(encoding="utf-8"))
    manifest_path = root / config["paper"]["metadata_dir"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return config, manifest, manifest_path.parent


def _split_dir(model_entry: dict[str, Any], split: str) -> Path:
    return _repo_root() / model_entry["run_dir"] / split


def _task_set_map(config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bucket, task_ids in config.get("task_sets", {}).items():
        for task_id in task_ids or []:
            mapping[str(task_id)] = str(bucket)
    return mapping


def _canonical_task_id(task_id: str) -> str:
    return TASK_ID_ALIASES.get(str(task_id), str(task_id))


def _task_bucket(task_id: str, task_set_map: dict[str, str]) -> str:
    canonical = _canonical_task_id(str(task_id))
    return task_set_map.get(canonical, task_set_map.get(str(task_id), "unassigned"))


def _display_name_for_model_entry(model_entry: dict[str, Any]) -> str:
    return str(model_entry.get("display_name") or model_entry["name"])


def _benchmark_role_for_model_entry(model_entry: dict[str, Any]) -> str:
    return str(model_entry.get("benchmark_role", "primary"))


def _model_names_for_roles(config: dict[str, Any], roles: set[str]) -> set[str]:
    names: set[str] = set()
    for model_entry in config.get("models", []) or []:
        if _benchmark_role_for_model_entry(model_entry) in roles:
            names.add(_display_name_for_model_entry(model_entry))
    return names


def _filter_model_episodes(episodes: pd.DataFrame, display_names: set[str]) -> pd.DataFrame:
    if episodes.empty or not display_names:
        return pd.DataFrame(columns=episodes.columns)
    return episodes[episodes["display_name"].isin(display_names)].copy()


def _episode_rows_for_split(model_entry: dict[str, Any], split: str) -> pd.DataFrame:
    _, episodes = load_run_records(_split_dir(model_entry, split))
    if episodes.empty:
        return episodes
    episodes = episodes.copy()
    episodes["requested_split"] = split
    episodes["display_name"] = model_entry["display_name"]
    episodes["family"] = model_entry["family"]
    episodes["size_b"] = float(model_entry["size_b"])
    episodes["benchmark_role"] = _benchmark_role_for_model_entry(model_entry)
    episodes["provider_mode"] = model_entry["provider_mode"]
    episodes["provenance"] = model_entry["provenance"]
    episodes["paper_model_name"] = model_entry["name"]
    return episodes


def _baseline_rows_for_split(baseline_name: str, baseline_entry: dict[str, Any], split: str) -> pd.DataFrame:
    split_entry = baseline_entry.get("splits", {}).get(split)
    if not split_entry:
        return pd.DataFrame()
    _, episodes = load_run_records(_repo_root() / split_entry["run_dir"])
    if episodes.empty:
        return episodes
    episodes = episodes.copy()
    episodes["requested_split"] = split
    episodes["display_name"] = baseline_entry["display_name"]
    episodes["family"] = baseline_entry["family"]
    episodes["size_b"] = 0.0
    episodes["provider_mode"] = baseline_entry["provider_mode"]
    episodes["provenance"] = baseline_entry["provenance"]
    episodes["paper_model_name"] = baseline_name
    return episodes


def _mean_and_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean_value = float(sum(values) / len(values))
    if len(values) < MIN_BOOTSTRAP_SAMPLE_COUNT:
        return mean_value, None, None
    ci_low, ci_high = bootstrap_mean_ci(values, samples=2000)
    return mean_value, ci_low, ci_high


def _aggregate_episode_group(group: pd.DataFrame) -> dict[str, Any]:
    score_values = [float(value) for value in group["score"].tolist()]
    overall_mean, ci_low, ci_high = _mean_and_ci(score_values)
    strict_count = int(group["strict_episode"].sum()) if "strict_episode" in group else 0
    strict_values = (
        [float(value) for value in group.loc[group["strict_episode"] == 1, "score"].tolist()]
        if "strict_episode" in group
        else []
    )
    strict_mean, strict_ci_low, strict_ci_high = _mean_and_ci(strict_values)
    shifted_values = (
        [float(value) for value in group.loc[group["hidden_shift_present"] == 1, "score"].tolist()]
        if "hidden_shift_present" in group
        else []
    )
    shifted_mean, shifted_ci_low, shifted_ci_high = _mean_and_ci(shifted_values)
    unshifted_values = (
        [float(value) for value in group.loc[group["hidden_shift_present"] == 0, "score"].tolist()]
        if "hidden_shift_present" in group
        else []
    )
    unshifted_mean, unshifted_ci_low, unshifted_ci_high = _mean_and_ci(unshifted_values)
    result: dict[str, Any] = {
        "episode_count": int(len(group)),
        "overall_mean_score": overall_mean,
        "overall_ci_low": ci_low,
        "overall_ci_high": ci_high,
        "strict_native_episode_count": strict_count if "strict_episode" in group else None,
        "strict_native_overall_mean_score": strict_mean if strict_count > 0 else None,
        "strict_native_ci_low": strict_ci_low if strict_count > 0 else None,
        "strict_native_ci_high": strict_ci_high if strict_count > 0 else None,
        "native_action_episode_rate": float(group["strict_episode"].mean()) if "strict_episode" in group else None,
        "empty_fallback_episode_rate": float(group["empty_fallback_episode"].mean())
        if "empty_fallback_episode" in group
        else None,
        "response_cap_hit_episode_rate": float(group["response_cap_hit"].mean()) if "response_cap_hit" in group else None,
        "shifted_episode_count": int(len(shifted_values)) if "hidden_shift_present" in group else None,
        "shifted_overall_mean_score": shifted_mean if shifted_values else None,
        "shifted_ci_low": shifted_ci_low if shifted_values else None,
        "shifted_ci_high": shifted_ci_high if shifted_values else None,
        "unshifted_episode_count": int(len(unshifted_values)) if "hidden_shift_present" in group else None,
        "unshifted_overall_mean_score": unshifted_mean if unshifted_values else None,
        "unshifted_ci_low": unshifted_ci_low if unshifted_values else None,
        "unshifted_ci_high": unshifted_ci_high if unshifted_values else None,
        "decision_only_mean_score": float(group["decision_only_score"].dropna().mean())
        if "decision_only_score" in group and not group["decision_only_score"].dropna().empty
        else None,
        "belief_tracking_mean_score": float(group["belief_tracking_score"].dropna().mean())
        if "belief_tracking_score" in group and not group["belief_tracking_score"].dropna().empty
        else None,
    }
    strict_native = result["strict_native_overall_mean_score"]
    overall = result["overall_mean_score"]
    result["shell_penalty"] = float(strict_native - overall) if strict_native is not None else None
    shifted = result["shifted_overall_mean_score"]
    unshifted = result["unshifted_overall_mean_score"]
    result["shift_adaptation_penalty"] = (
        float(unshifted - shifted) if shifted is not None and unshifted is not None else None
    )
    if "cost_usd" in group:
        result["total_cost_usd"] = float(group["cost_usd"].fillna(0.0).sum())
    if "input_tokens" in group:
        result["total_input_tokens"] = int(group["input_tokens"].fillna(0).sum())
    if "output_tokens" in group:
        result["total_output_tokens"] = int(group["output_tokens"].fillna(0).sum())
    return result


def _main_results_table(episodes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    main_tasks = set(config.get("task_sets", {}).get("main", []) or [])
    main_df = episodes[episodes["task_id"].isin(main_tasks)].copy()
    rows: list[dict[str, Any]] = []
    for (display_name, family, size_b, provider_mode, provenance, requested_split), group in main_df.groupby(
        ["display_name", "family", "size_b", "provider_mode", "provenance", "requested_split"], dropna=False
    ):
        row = {
            "model": display_name,
            "family": family,
            "size_b": float(size_b),
            "provider_mode": provider_mode,
            "provenance": provenance,
            "split": requested_split,
            "task_set": "main",
        }
        row.update(_aggregate_episode_group(group))
        rows.append(row)
    columns = [
        "model",
        "family",
        "size_b",
        "provider_mode",
        "provenance",
        "split",
        "task_set",
        "episode_count",
        "overall_mean_score",
        "overall_ci_low",
        "overall_ci_high",
        "strict_native_episode_count",
        "strict_native_overall_mean_score",
        "strict_native_ci_low",
        "strict_native_ci_high",
        "native_action_episode_rate",
        "empty_fallback_episode_rate",
        "response_cap_hit_episode_rate",
        "shifted_episode_count",
        "shifted_overall_mean_score",
        "shifted_ci_low",
        "shifted_ci_high",
        "unshifted_episode_count",
        "unshifted_overall_mean_score",
        "unshifted_ci_low",
        "unshifted_ci_high",
        "decision_only_mean_score",
        "belief_tracking_mean_score",
        "shell_penalty",
        "shift_adaptation_penalty",
        "total_cost_usd",
        "total_input_tokens",
        "total_output_tokens",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["family", "size_b", "split"]).reset_index(drop=True)


def _task_breakdown_table(episodes: pd.DataFrame, task_set_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (display_name, family, size_b, requested_split, task_id), group in episodes.groupby(
        ["display_name", "family", "size_b", "requested_split", "task_id"], dropna=False
    ):
        row = {
            "model": display_name,
            "family": family,
            "size_b": float(size_b),
            "split": requested_split,
            "task_id": task_id,
            "task_bucket": _task_bucket(str(task_id), task_set_map),
        }
        row.update(_aggregate_episode_group(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_bucket", "task_id", "family", "size_b", "split"]).reset_index(drop=True)


def _subscore_breakdown_table(episodes: pd.DataFrame, task_set_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available_subscores = [column for column in SUBSCORE_COLUMNS if column in episodes.columns]
    for (display_name, family, size_b, requested_split, task_id), group in episodes.groupby(
        ["display_name", "family", "size_b", "requested_split", "task_id"], dropna=False
    ):
        row = {
            "model": display_name,
            "family": family,
            "size_b": float(size_b),
            "split": requested_split,
            "task_id": task_id,
            "task_bucket": _task_bucket(str(task_id), task_set_map),
        }
        for column in available_subscores:
            values = group[column].dropna()
            row[column.replace("subscore_", "")] = float(values.mean()) if not values.empty else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_bucket", "task_id", "family", "size_b", "split"]).reset_index(drop=True)


def _heldout_gap_table(main_results: pd.DataFrame) -> pd.DataFrame:
    pivot = main_results.pivot_table(
        index=["model", "family", "size_b", "provider_mode", "provenance"],
        columns="split",
        values="overall_mean_score",
    ).reset_index()
    if "core" in pivot.columns and "heldout" in pivot.columns:
        pivot["heldout_gap"] = pivot["heldout"] - pivot["core"]
    else:
        pivot["heldout_gap"] = None
    return pivot.sort_values(["family", "size_b"]).reset_index(drop=True)


def _family_scaling_table(main_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "family",
        "size_b",
        "split",
        "episode_count",
        "overall_mean_score",
        "overall_ci_low",
        "overall_ci_high",
        "strict_native_episode_count",
        "strict_native_overall_mean_score",
        "strict_native_ci_low",
        "strict_native_ci_high",
        "native_action_episode_rate",
        "empty_fallback_episode_rate",
        "response_cap_hit_episode_rate",
        "shifted_episode_count",
        "shifted_overall_mean_score",
        "shifted_ci_low",
        "shifted_ci_high",
        "unshifted_episode_count",
        "unshifted_overall_mean_score",
        "unshifted_ci_low",
        "unshifted_ci_high",
        "shell_penalty",
        "shift_adaptation_penalty",
        "total_output_tokens",
    ]
    return main_results[columns].sort_values(["family", "size_b", "split"]).reset_index(drop=True)


def _baseline_results_table(baseline_episodes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if baseline_episodes.empty:
        return pd.DataFrame()
    main_tasks = set(config.get("task_sets", {}).get("main", []) or [])
    main_df = baseline_episodes[baseline_episodes["task_id"].isin(main_tasks)].copy()
    rows: list[dict[str, Any]] = []
    for (display_name, family, provider_mode, provenance, requested_split), group in main_df.groupby(
        ["display_name", "family", "provider_mode", "provenance", "requested_split"], dropna=False
    ):
        row = {
            "baseline": display_name,
            "family": family,
            "provider_mode": provider_mode,
            "provenance": provenance,
            "split": requested_split,
            "task_set": "main",
        }
        row.update(_aggregate_episode_group(group))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["split", "overall_mean_score"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _benchmark_health_table(main_results: pd.DataFrame, baseline_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not baseline_results.empty:
        for split, group in baseline_results.groupby("split"):
            scores = {str(row["baseline"]): float(row["overall_mean_score"]) for _, row in group.iterrows()}
            random_score = scores.get("random")
            heuristic_score = scores.get("visible heuristic")
            oracle_score = scores.get("myopic oracle")
            rows.append(
                {
                    "check": f"{split}_baseline_ordering",
                    "value": json.dumps(scores, sort_keys=True),
                    "passes": bool(
                        random_score is not None
                        and heuristic_score is not None
                        and oracle_score is not None
                        and random_score < heuristic_score < oracle_score
                    ),
                    "threshold": "random < visible heuristic < myopic oracle",
                }
            )
            if random_score is not None and oracle_score is not None:
                rows.append(
                    {
                        "check": f"{split}_oracle_gap",
                        "value": oracle_score - random_score,
                        "passes": bool((oracle_score - random_score) >= 0.20),
                        "threshold": ">= 0.20",
                    }
                )
    if not main_results.empty:
        rows.append(
            {
                "check": "model_split_coverage",
                "value": int(len(main_results)),
                "passes": bool(len(main_results) == int(main_results["model"].nunique()) * 2),
                "threshold": "core and heldout rows for each frozen model",
            }
        )
        rows.append(
            {
                "check": "main_task_episode_count",
                "value": int(main_results["episode_count"].min()),
                "passes": bool(int(main_results["episode_count"].min()) >= 30),
                "threshold": ">= 30 episodes per model/split on the primary task aggregate",
            }
        )
        shifted_penalties = main_results["shift_adaptation_penalty"].dropna()
        rows.append(
            {
                "check": "silent_shift_penalty_present",
                "value": float(shifted_penalties.median()) if not shifted_penalties.empty else None,
                "passes": bool(not shifted_penalties.empty and float(shifted_penalties.median()) >= 0.10),
                "threshold": "median unshifted-minus-shifted gap >= 0.10",
            }
        )
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _leakage_table(config: dict[str, Any], task_set_map: dict[str, str]) -> pd.DataFrame:
    columns = [
        "split",
        "task_id",
        "canonical_task_id",
        "task_bucket",
        "view",
        "centroid_accuracy",
        "explicit_goal_token_rate",
        "n_train",
        "n_test",
        "vocab_size",
        "numeric_feature_count",
    ]
    inputs = config.get("analysis_inputs", {}).get("leakage", {}) or {}
    rows: list[dict[str, Any]] = []
    for split, raw_path in inputs.items():
        payload = _load_json(_repo_root() / str(raw_path))
        for task_id, report in (payload.get("results", {}) or {}).items():
            rows.append(
                {
                    "split": split,
                    "task_id": task_id,
                    "canonical_task_id": _canonical_task_id(task_id),
                    "task_bucket": _task_bucket(task_id, task_set_map),
                    "view": "full_observation",
                    "centroid_accuracy": report.get("centroid_accuracy"),
                    "explicit_goal_token_rate": report.get("explicit_goal_token_rate"),
                    "n_train": report.get("n_train"),
                    "n_test": report.get("n_test"),
                    "vocab_size": report.get("vocab_size"),
                    "numeric_feature_count": report.get("numeric_feature_count"),
                }
            )
            for view_name, view_report in (report.get("views", {}) or {}).items():
                rows.append(
                    {
                        "split": split,
                        "task_id": task_id,
                        "canonical_task_id": _canonical_task_id(task_id),
                        "task_bucket": _task_bucket(task_id, task_set_map),
                        "view": view_name,
                        "centroid_accuracy": view_report.get("centroid_accuracy"),
                        "explicit_goal_token_rate": view_report.get("explicit_goal_token_rate"),
                        "n_train": view_report.get("n_train"),
                        "n_test": view_report.get("n_test"),
                        "vocab_size": view_report.get("vocab_size"),
                        "numeric_feature_count": view_report.get("numeric_feature_count"),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["task_bucket", "canonical_task_id", "split", "view"])
        .reset_index(drop=True)
    )


def _extract_repeat_index(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"rep(?:eat_)?(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _reliability_table(config: dict[str, Any], task_set_map: dict[str, str]) -> pd.DataFrame:
    columns = [
        "split",
        "model_name",
        "repeat_count",
        "mean_repeat_score",
        "std_repeat_score",
        "repeat_ci_low",
        "repeat_ci_high",
        "score_at_1",
        "score_at_3",
        "score_at_5",
        "native_action_episode_rate",
        "empty_fallback_episode_rate",
        "response_cap_hit_episode_rate",
    ]
    inputs = config.get("analysis_inputs", {}).get("reliability", {}) or {}
    main_tasks = set(config.get("task_sets", {}).get("main", []) or [])
    rows: list[dict[str, Any]] = []
    for split, root_key in [("core", "core_root"), ("heldout", "heldout_root")]:
        reliability_root = _repo_root() / str(inputs.get(root_key, ""))
        if not reliability_root.exists():
            continue
        for model_dir in sorted(path for path in reliability_root.iterdir() if path.is_dir()):
            _, episodes = load_run_records(model_dir)
            if episodes.empty:
                continue
            episodes = episodes.copy()
            episodes["canonical_task_id"] = episodes["task_id"].map(_canonical_task_id)
            episodes = episodes[episodes["canonical_task_id"].isin(main_tasks)]
            if episodes.empty:
                continue
            repeat_index = _extract_repeat_index(episodes["run_id"])
            if repeat_index.isna().all() and "source_jsonl" in episodes.columns:
                repeat_index = _extract_repeat_index(episodes["source_jsonl"])
            episodes["repeat_index"] = repeat_index
            repeat_scores = (
                episodes.dropna(subset=["repeat_index"])
                .groupby("repeat_index", as_index=False)["score"]
                .mean()
                .sort_values("repeat_index")
            )
            score_values = [float(value) for value in repeat_scores["score"].tolist()]
            ci_low, ci_high = bootstrap_mean_ci(score_values) if score_values else (0.0, 0.0)
            score_at_1 = float(np.mean(score_values[:1])) if len(score_values) >= 1 else None
            score_at_3 = float(max(score_values[:3])) if len(score_values) >= 3 else None
            score_at_5 = float(max(score_values[:5])) if len(score_values) >= 5 else None
            rows.append(
                {
                    "split": split,
                    "model_name": str(model_dir.name),
                    "repeat_count": int(len(score_values)),
                    "mean_repeat_score": float(np.mean(score_values)) if score_values else None,
                    "std_repeat_score": float(np.std(score_values, ddof=0)) if score_values else None,
                    "repeat_ci_low": ci_low if score_values else None,
                    "repeat_ci_high": ci_high if score_values else None,
                    "score_at_1": score_at_1,
                    "score_at_3": score_at_3,
                    "score_at_5": score_at_5,
                    "native_action_episode_rate": float(episodes["strict_episode"].mean())
                    if "strict_episode" in episodes
                    else None,
                    "empty_fallback_episode_rate": float(episodes["empty_fallback_episode"].mean())
                    if "empty_fallback_episode" in episodes
                    else None,
                    "response_cap_hit_episode_rate": float(episodes["response_cap_hit"].mean())
                    if "response_cap_hit" in episodes
                    else None,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["model_name", "split"]).reset_index(drop=True)


def _shell_controlled_label(row: pd.Series) -> str:
    frozen = float(row.get("frozen_overall_mean_score") or 0.0)
    assisted = row.get("diagnostic_assisted_overall_mean_score")
    native = float(row.get("diagnostic_native_action_episode_rate") or 0.0)
    repaired = float(row.get("diagnostic_parse_repaired_episode_rate") or 0.0)
    if pd.notna(assisted) and float(assisted) >= frozen + 0.05 and native >= 0.20:
        return "partial_shell_recovery"
    if repaired > 0.0 and native < 0.10:
        return "parse_repair_minimal_help"
    if native == 0.0:
        return "no_native_recovery"
    return "mixed_effect"


def _shell_controlled_diagnostic_table(main_results: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "diagnostic_name",
        "description",
        "model",
        "family",
        "size_b",
        "split",
        "frozen_overall_mean_score",
        "frozen_strict_native_overall_mean_score",
        "frozen_native_action_episode_rate",
        "frozen_empty_fallback_episode_rate",
        "diagnostic_overall_mean_score",
        "diagnostic_strict_native_overall_mean_score",
        "diagnostic_assisted_overall_mean_score",
        "diagnostic_native_action_episode_rate",
        "diagnostic_assisted_episode_rate",
        "diagnostic_empty_fallback_episode_rate",
        "diagnostic_response_cap_hit_episode_rate",
        "diagnostic_parse_repaired_episode_count",
        "diagnostic_parse_repaired_episode_rate",
        "delta_overall_mean_score",
        "delta_native_action_episode_rate",
        "delta_empty_fallback_episode_rate",
        "assisted_minus_frozen_overall",
        "diagnostic_label",
    ]
    diagnostic_runs = manifest.get("diagnostic_runs", {}) or {}
    if not diagnostic_runs:
        return pd.DataFrame(columns=columns)
    frozen_lookup = main_results.set_index(["model", "split"])
    rows: list[dict[str, Any]] = []
    for diagnostic_name, diagnostic in diagnostic_runs.items():
        description = str(diagnostic.get("description", ""))
        for model_entry in diagnostic.get("models", []) or []:
            for split in ("core", "heldout"):
                split_payload = (model_entry.get("splits", {}) or {}).get(split)
                if not split_payload:
                    continue
                frozen_row = (
                    frozen_lookup.loc[(model_entry["display_name"], split)]
                    if (model_entry["display_name"], split) in frozen_lookup.index
                    else None
                )
                row = {
                    "diagnostic_name": diagnostic_name,
                    "description": description,
                    "model": model_entry["display_name"],
                    "family": model_entry["family"],
                    "size_b": float(model_entry["size_b"]),
                    "split": split,
                    "frozen_overall_mean_score": frozen_row["overall_mean_score"] if frozen_row is not None else None,
                    "frozen_strict_native_overall_mean_score": frozen_row["strict_native_overall_mean_score"] if frozen_row is not None else None,
                    "frozen_native_action_episode_rate": frozen_row["native_action_episode_rate"] if frozen_row is not None else None,
                    "frozen_empty_fallback_episode_rate": frozen_row["empty_fallback_episode_rate"] if frozen_row is not None else None,
                    "diagnostic_overall_mean_score": split_payload.get("overall_mean_score"),
                    "diagnostic_strict_native_overall_mean_score": split_payload.get("strict_native_overall_mean_score"),
                    "diagnostic_assisted_overall_mean_score": split_payload.get("assisted_overall_mean_score"),
                    "diagnostic_native_action_episode_rate": split_payload.get("native_action_episode_rate"),
                    "diagnostic_assisted_episode_rate": split_payload.get("assisted_episode_rate"),
                    "diagnostic_empty_fallback_episode_rate": split_payload.get("empty_fallback_episode_rate"),
                    "diagnostic_response_cap_hit_episode_rate": split_payload.get("response_cap_hit_episode_rate"),
                    "diagnostic_parse_repaired_episode_count": split_payload.get("parse_repaired_episode_count"),
                    "diagnostic_parse_repaired_episode_rate": split_payload.get("parse_repaired_episode_rate"),
                }
                if frozen_row is not None:
                    row["delta_overall_mean_score"] = (
                        float(row["diagnostic_overall_mean_score"] - row["frozen_overall_mean_score"])
                        if pd.notna(row["diagnostic_overall_mean_score"])
                        else None
                    )
                    row["delta_native_action_episode_rate"] = (
                        float(row["diagnostic_native_action_episode_rate"] - row["frozen_native_action_episode_rate"])
                        if pd.notna(row["diagnostic_native_action_episode_rate"])
                        else None
                    )
                    row["delta_empty_fallback_episode_rate"] = (
                        float(row["diagnostic_empty_fallback_episode_rate"] - row["frozen_empty_fallback_episode_rate"])
                        if pd.notna(row["diagnostic_empty_fallback_episode_rate"])
                        else None
                    )
                    row["assisted_minus_frozen_overall"] = (
                        float(row["diagnostic_assisted_overall_mean_score"] - row["frozen_overall_mean_score"])
                        if pd.notna(row["diagnostic_assisted_overall_mean_score"])
                        else None
                    )
                else:
                    row["delta_overall_mean_score"] = None
                    row["delta_native_action_episode_rate"] = None
                    row["delta_empty_fallback_episode_rate"] = None
                    row["assisted_minus_frozen_overall"] = None
                row["diagnostic_label"] = _shell_controlled_label(pd.Series(row))
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["family", "size_b", "split"]).reset_index(drop=True)


def _mechanism_label(row: pd.Series) -> str:
    native_rate = float(row.get("native_action_episode_rate") or 0.0)
    shell_penalty = float(row.get("shell_penalty") or 0.0)
    shift_penalty = float(row.get("shift_adaptation_penalty") or 0.0)
    if native_rate < 0.8 and shell_penalty > 0.02:
        return "interface_failure_depresses_observed_score"
    if native_rate < 0.8 and abs(shell_penalty) <= 0.02:
        return "interface_failure_without_clear_score_penalty"
    if native_rate >= 0.95 and shift_penalty > 0.10:
        return "native_policy_struggles_after_shift"
    if native_rate >= 0.95:
        return "native_policy_signal"
    return "mixed_interface_and_policy"


def _mechanism_diagnostics_table(family_scaling: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in family_scaling.iterrows():
        native_rate = row.get("native_action_episode_rate")
        rows.append(
            {
                "model": row["model"],
                "family": row["family"],
                "size_b": row["size_b"],
                "split": row["split"],
                "overall_mean_score": row["overall_mean_score"],
                "strict_native_episode_count": row["strict_native_episode_count"],
                "strict_native_overall_mean_score": row["strict_native_overall_mean_score"],
                "native_action_episode_rate": native_rate,
                "interface_failure_rate": float(1.0 - native_rate) if pd.notna(native_rate) else None,
                "empty_fallback_episode_rate": row["empty_fallback_episode_rate"],
                "response_cap_hit_episode_rate": row["response_cap_hit_episode_rate"],
                "shell_penalty": row["shell_penalty"],
                "shift_adaptation_penalty": row["shift_adaptation_penalty"],
                "mechanism_label": _mechanism_label(row),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "size_b", "split"]).reset_index(drop=True)


def _scaling_delta_label(delta_overall: float, delta_native_rate: float, delta_native_score: float | None) -> str:
    if delta_overall > 0.01 and delta_native_rate > -0.05:
        return "larger_model_gain"
    if delta_overall < -0.01 and delta_native_rate < -0.10:
        return "larger_model_interface_regression"
    if delta_overall < -0.01 and delta_native_score is not None and delta_native_score < -0.01:
        return "larger_model_native_policy_regression"
    if abs(delta_overall) <= 0.01:
        return "approximately_flat"
    return "mixed_or_ambiguous"


def _scaling_delta_table(family_scaling: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "split",
        "from_model",
        "to_model",
        "from_size_b",
        "to_size_b",
        "delta_overall_mean_score",
        "delta_strict_native_overall_mean_score",
        "delta_native_action_episode_rate",
        "delta_empty_fallback_episode_rate",
        "delta_shifted_overall_mean_score",
        "delta_unshifted_overall_mean_score",
        "delta_shell_penalty",
        "delta_shift_adaptation_penalty",
        "scaling_label",
    ]
    rows: list[dict[str, Any]] = []
    for (family, split), group in family_scaling.groupby(["family", "split"], dropna=False):
        ordered = group.sort_values("size_b").reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            previous_native = previous.get("strict_native_overall_mean_score")
            current_native = current.get("strict_native_overall_mean_score")
            delta_native = (
                float(current_native - previous_native)
                if pd.notna(current_native) and pd.notna(previous_native)
                else None
            )
            delta_overall = float(current["overall_mean_score"] - previous["overall_mean_score"])
            delta_native_rate = float(current["native_action_episode_rate"] - previous["native_action_episode_rate"])
            rows.append(
                {
                    "family": family,
                    "split": split,
                    "from_model": previous["model"],
                    "to_model": current["model"],
                    "from_size_b": previous["size_b"],
                    "to_size_b": current["size_b"],
                    "delta_overall_mean_score": delta_overall,
                    "delta_strict_native_overall_mean_score": delta_native,
                    "delta_native_action_episode_rate": delta_native_rate,
                    "delta_empty_fallback_episode_rate": float(
                        current["empty_fallback_episode_rate"] - previous["empty_fallback_episode_rate"]
                    ),
                    "delta_shifted_overall_mean_score": float(
                        current["shifted_overall_mean_score"] - previous["shifted_overall_mean_score"]
                    ),
                    "delta_unshifted_overall_mean_score": float(
                        current["unshifted_overall_mean_score"] - previous["unshifted_overall_mean_score"]
                    ),
                    "delta_shell_penalty": float(current["shell_penalty"] - previous["shell_penalty"]),
                    "delta_shift_adaptation_penalty": float(
                        current["shift_adaptation_penalty"] - previous["shift_adaptation_penalty"]
                    ),
                    "scaling_label": _scaling_delta_label(delta_overall, delta_native_rate, delta_native),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["family", "from_size_b", "split"]).reset_index(drop=True)


def _iter_terminal_episode_payloads(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for jsonl_path in sorted(run_dir.rglob("runs.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if "total_steps" in payload and "score" in payload:
                    payloads.append(payload)
    return payloads


def _adaptation_episode_rows(
    manifest: dict[str, Any], task_set_map: dict[str, str], allowed_model_names: set[str] | None = None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_entry in manifest.get("models", []):
        display_name = str(model_entry.get("display_name") or model_entry["name"])
        if allowed_model_names is not None and display_name not in allowed_model_names:
            continue
        for split in ("core", "heldout"):
            for payload in _iter_terminal_episode_payloads(_split_dir(model_entry, split)):
                task_id = str(payload.get("task_id", ""))
                grader = payload.get("grade", {}) or payload.get("observation", {}).get("metadata", {}).get("grader", {}) or {}
                sub_scores = grader.get("sub_scores", {}) or {}
                details = grader.get("details", {}) or {}
                rows.append(
                    {
                        "model": display_name,
                        "family": model_entry["family"],
                        "size_b": float(model_entry["size_b"]),
                        "benchmark_role": str(model_entry.get("benchmark_role", "primary")),
                        "provider_mode": model_entry["provider_mode"],
                        "provenance": model_entry["provenance"],
                        "split": split,
                        "task_id": task_id,
                        "task_bucket": _task_bucket(task_id, task_set_map),
                        "seed": int(payload.get("seed", 0)),
                        "hidden_shift_present": int(bool(payload.get("metadata", {}).get("hidden_shift_present", False))),
                        "adaptation_scored": int(bool(details.get("adaptation_scored", False))),
                        "adaptation_score": sub_scores.get("adaptation"),
                        "behavioral_shift_detection_delay_steps": details.get("behavioral_shift_detection_delay_steps"),
                        "post_shift_recovery": details.get("post_shift_recovery"),
                        "post_shift_regret": details.get("post_shift_regret"),
                        "coherence_score": sub_scores.get("coherence"),
                        "constraint_score": sub_scores.get("constraints"),
                        "final_utility_score": sub_scores.get("final_utility"),
                    }
                )
    return pd.DataFrame(rows)


def _adaptation_audit_table(
    manifest: dict[str, Any], task_set_map: dict[str, str], allowed_model_names: set[str] | None = None
) -> pd.DataFrame:
    columns = [
        "model",
        "family",
        "size_b",
        "provider_mode",
        "provenance",
        "split",
        "task_id",
        "task_bucket",
        "episode_count",
        "shifted_episode_count",
        "no_shift_episode_count",
        "adaptation_scored_episode_count",
        "zero_adaptation_episode_rate",
        "nonzero_adaptation_episode_rate",
        "mean_adaptation_score",
        "mean_detection_delay_steps",
        "mean_post_shift_recovery",
        "mean_post_shift_regret",
        "mean_coherence_score",
        "mean_constraint_score",
        "mean_final_utility_score",
    ]
    episode_rows = _adaptation_episode_rows(manifest, task_set_map, allowed_model_names)
    if episode_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in episode_rows.groupby(
        ["model", "family", "size_b", "provider_mode", "provenance", "split", "task_id", "task_bucket"],
        dropna=False,
    ):
        (
            model,
            family,
            size_b,
            provider_mode,
            provenance,
            split,
            task_id,
            task_bucket,
        ) = keys
        shifted = group[group["hidden_shift_present"] == 1].copy()
        scored = shifted[shifted["adaptation_scored"] == 1].copy()
        adaptation_values = scored["adaptation_score"].dropna()
        rows.append(
            {
                "model": model,
                "family": family,
                "size_b": float(size_b),
                "provider_mode": provider_mode,
                "provenance": provenance,
                "split": split,
                "task_id": task_id,
                "task_bucket": task_bucket,
                "episode_count": int(len(group)),
                "shifted_episode_count": int(len(shifted)),
                "no_shift_episode_count": int(len(group) - len(shifted)),
                "adaptation_scored_episode_count": int(len(scored)),
                "zero_adaptation_episode_rate": (
                    float((adaptation_values == 0.0).mean()) if not adaptation_values.empty else None
                ),
                "nonzero_adaptation_episode_rate": (
                    float((adaptation_values > 0.0).mean()) if not adaptation_values.empty else None
                ),
                "mean_adaptation_score": float(adaptation_values.mean()) if not adaptation_values.empty else None,
                "mean_detection_delay_steps": (
                    float(scored["behavioral_shift_detection_delay_steps"].dropna().mean())
                    if not scored["behavioral_shift_detection_delay_steps"].dropna().empty
                    else None
                ),
                "mean_post_shift_recovery": (
                    float(scored["post_shift_recovery"].dropna().mean())
                    if not scored["post_shift_recovery"].dropna().empty
                    else None
                ),
                "mean_post_shift_regret": (
                    float(scored["post_shift_regret"].dropna().mean())
                    if not scored["post_shift_regret"].dropna().empty
                    else None
                ),
                "mean_coherence_score": float(group["coherence_score"].dropna().mean())
                if not group["coherence_score"].dropna().empty
                else None,
                "mean_constraint_score": float(group["constraint_score"].dropna().mean())
                if not group["constraint_score"].dropna().empty
                else None,
                "mean_final_utility_score": float(group["final_utility_score"].dropna().mean())
                if not group["final_utility_score"].dropna().empty
                else None,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["task_bucket", "task_id", "family", "size_b", "split"]
    ).reset_index(drop=True)


def _write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


def generate_tables(config_path: str | Path) -> dict[str, Any]:
    config, manifest, metadata_dir = _load_manifest(config_path)
    root = _repo_root()
    task_set_map = _task_set_map(config)
    primary_model_names = _model_names_for_roles(config, PRIMARY_BENCHMARK_ROLES)
    external_anchor_names = _model_names_for_roles(config, EXTERNAL_ANCHOR_ROLES)
    frames = []
    for model_entry in manifest["models"]:
        for split in ("core", "heldout"):
            frames.append(_episode_rows_for_split(model_entry, split))
    episodes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    primary_episodes = _filter_model_episodes(episodes, primary_model_names)
    external_anchor_episodes = _filter_model_episodes(episodes, external_anchor_names)
    baseline_frames = []
    for baseline_name, baseline_entry in manifest.get("baseline_runs", {}).items():
        for split in ("core", "heldout"):
            baseline_frames.append(_baseline_rows_for_split(baseline_name, baseline_entry, split))
    baseline_episodes = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()

    tables_dir = root / config["paper"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    main_results = _main_results_table(primary_episodes, config)
    external_anchor_results = _main_results_table(external_anchor_episodes, config)
    task_breakdown = _task_breakdown_table(primary_episodes, task_set_map)
    subscore_breakdown = _subscore_breakdown_table(primary_episodes, task_set_map)
    heldout_gap = _heldout_gap_table(main_results)
    family_scaling = _family_scaling_table(main_results)
    baseline_results = _baseline_results_table(baseline_episodes, config)
    benchmark_health = _benchmark_health_table(main_results, baseline_results)
    leakage_results = _leakage_table(config, task_set_map)
    reliability_results = _reliability_table(config, task_set_map)
    mechanism_diagnostics = _mechanism_diagnostics_table(family_scaling)
    scaling_deltas = _scaling_delta_table(family_scaling)
    adaptation_audit = _adaptation_audit_table(manifest, task_set_map, primary_model_names)
    shell_controlled_diagnostic = _shell_controlled_diagnostic_table(main_results, manifest)

    _write_table(main_results, tables_dir, "main_results")
    _write_table(external_anchor_results, tables_dir, "external_anchor_results")
    _write_table(baseline_results, tables_dir, "baseline_results")
    _write_table(benchmark_health, tables_dir, "benchmark_health")
    _write_table(leakage_results, tables_dir, "leakage_results")
    _write_table(reliability_results, tables_dir, "reliability_results")
    _write_table(task_breakdown, tables_dir, "task_breakdown")
    _write_table(subscore_breakdown, tables_dir, "subscore_breakdown")
    _write_table(heldout_gap, tables_dir, "heldout_gap")
    _write_table(family_scaling, tables_dir, "family_scaling")
    _write_table(mechanism_diagnostics, tables_dir, "mechanism_diagnostics")
    _write_table(scaling_deltas, tables_dir, "scaling_deltas")
    _write_table(adaptation_audit, tables_dir, "adaptation_audit")
    _write_table(shell_controlled_diagnostic, tables_dir, "shell_controlled_diagnostic")

    summary = {
        "table_dir": str(tables_dir.relative_to(root)),
        "episode_rows": int(len(episodes)),
        "primary_episode_rows": int(len(primary_episodes)),
        "external_anchor_episode_rows": int(len(external_anchor_episodes)),
        "baseline_episode_rows": int(len(baseline_episodes)),
        "model_count": int(len(main_results["model"].unique())) if not main_results.empty else 0,
        "external_anchor_count": int(len(external_anchor_results["model"].unique())) if not external_anchor_results.empty else 0,
        "baseline_count": int(len(baseline_results["baseline"].unique())) if not baseline_results.empty else 0,
        "main_rows": int(len(main_results)),
        "external_anchor_rows": int(len(external_anchor_results)),
        "baseline_rows": int(len(baseline_results)),
        "leakage_rows": int(len(leakage_results)),
        "reliability_rows": int(len(reliability_results)),
        "task_rows": int(len(task_breakdown)),
        "subscore_rows": int(len(subscore_breakdown)),
        "benchmark_health_rows": int(len(benchmark_health)),
        "mechanism_rows": int(len(mechanism_diagnostics)),
        "scaling_delta_rows": int(len(scaling_deltas)),
        "adaptation_audit_rows": int(len(adaptation_audit)),
        "shell_controlled_diagnostic_rows": int(len(shell_controlled_diagnostic)),
        "manifest_path": str((metadata_dir / "run_manifest.json").relative_to(root)),
    }
    (tables_dir / "tables_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_protocol.yaml")
    args = parser.parse_args()
    print(json.dumps(generate_tables(args.config), indent=2))


if __name__ == "__main__":
    main()
