"""Freeze-aware paper suite runner for tables, figures, and metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class FrozenModelRun:
    name: str
    display_name: str
    family: str
    size_b: float
    benchmark_role: str
    provider_mode: str
    provenance: str
    run_dir: Path


@dataclass(slots=True)
class FrozenDiagnosticRun:
    diagnostic_name: str
    description: str
    settings: dict[str, Any]
    name: str
    display_name: str
    family: str
    size_b: float
    provider_mode: str
    provenance: str
    run_dir: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _load_protocol(config_path: Path) -> tuple[dict[str, Any], list[FrozenModelRun]]:
    root = _repo_root()
    protocol = _load_yaml(config_path)
    models = [
        FrozenModelRun(
            name=str(item["name"]),
            display_name=str(item.get("display_name") or item["name"]),
            family=str(item["family"]),
            size_b=float(item["size_b"]),
            benchmark_role=str(item.get("benchmark_role", "primary")),
            provider_mode=str(item["provider_mode"]),
            provenance=str(item["provenance"]),
            run_dir=_resolve_path(root, str(item["run_dir"])),
        )
        for item in protocol.get("models", [])
    ]
    return protocol, models


def _summary_payload(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _validate_model_run(model: FrozenModelRun) -> dict[str, Any]:
    split_names: list[str] = ["core", "heldout"]
    if (model.run_dir / "smoke-core" / "summary.json").exists():
        split_names.insert(0, "smoke-core")
    return _validate_named_summaries(model.run_dir, model.name, tuple(split_names))


def _validate_named_summaries(run_dir: Path, run_name: str, split_names: tuple[str, ...]) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    for split in split_names:
        splits[split] = _validate_split_summary(run_dir / split, run_name, split)
    return splits


def _validate_split_summary(run_dir: Path, run_name: str, split: str) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {split} summary for {run_name}: {summary_path}")
    payload = _summary_payload(summary_path)
    if not bool(payload.get("run_complete", False)):
        raise ValueError(f"Incomplete {split} summary for {run_name}: {summary_path}")
    return payload


def _task_membership(protocol: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bucket_name, task_ids in protocol.get("task_sets", {}).items():
        for task_id in task_ids or []:
            mapping[str(task_id)] = str(bucket_name)
    return mapping


def _manifest_split_payload(payload: dict[str, Any]) -> dict[str, Any]:
    parse_repaired_rate = payload.get("parse_repaired_episode_rate")
    if parse_repaired_rate is None:
        repaired_count = payload.get("parse_repaired_episode_count")
        episode_denominator = payload.get("episodes_completed", payload.get("episode_count"))
        if repaired_count is not None and episode_denominator:
            parse_repaired_rate = float(repaired_count) / float(episode_denominator)
    return {
        "overall_mean_score": payload.get("overall_mean_score"),
        "strict_native_overall_mean_score": payload.get("strict_native_overall_mean_score"),
        "assisted_overall_mean_score": payload.get("assisted_overall_mean_score"),
        "native_action_episode_rate": payload.get("native_action_episode_rate"),
        "assisted_episode_rate": payload.get("assisted_episode_rate"),
        "empty_fallback_episode_rate": payload.get("empty_fallback_episode_rate"),
        "response_cap_hit_episode_rate": payload.get("response_cap_hit_episode_rate"),
        "parse_repaired_episode_count": payload.get("parse_repaired_episode_count"),
        "parse_repaired_episode_rate": parse_repaired_rate,
        "shifted_overall_mean_score": payload.get("shifted_overall_mean_score"),
        "unshifted_overall_mean_score": payload.get("unshifted_overall_mean_score"),
        "episodes_expected": payload.get("episodes_expected"),
        "episodes_completed": payload.get("episodes_completed"),
        "completion_status": payload.get("completion_status"),
        "benchmark_runtime_versions": payload.get("benchmark_runtime_versions", []),
    }


def _load_diagnostic_runs(protocol: dict[str, Any]) -> list[FrozenDiagnosticRun]:
    root = _repo_root()
    diagnostics: list[FrozenDiagnosticRun] = []
    for diagnostic_name, diagnostic in (protocol.get("diagnostic_runs", {}) or {}).items():
        description = str(diagnostic.get("description", ""))
        settings = dict(diagnostic.get("settings", {}) or {})
        for item in diagnostic.get("models", []) or []:
            diagnostics.append(
                FrozenDiagnosticRun(
                    diagnostic_name=str(diagnostic_name),
                    description=description,
                    settings=settings,
                    name=str(item["name"]),
                    display_name=str(item.get("display_name") or item["name"]),
                    family=str(item["family"]),
                    size_b=float(item["size_b"]),
                    provider_mode=str(item["provider_mode"]),
                    provenance=str(item["provenance"]),
                    run_dir=_resolve_path(root, str(item["run_dir"])),
                )
            )
    return diagnostics


def build_manifest(protocol: dict[str, Any], models: list[FrozenModelRun]) -> dict[str, Any]:
    root = _repo_root()
    paper = protocol.get("paper", {})
    task_membership = _task_membership(protocol)
    manifest_models: list[dict[str, Any]] = []
    for model in models:
        split_payloads = _validate_model_run(model)
        manifest_models.append(
            {
                "name": model.name,
                "display_name": model.display_name,
                "family": model.family,
                "size_b": model.size_b,
                "benchmark_role": model.benchmark_role,
                "provider_mode": model.provider_mode,
                "provenance": model.provenance,
                "run_dir": str(model.run_dir.relative_to(root)),
                "splits": {split_name: _manifest_split_payload(payload) for split_name, payload in split_payloads.items()},
            }
        )
    manifest_baselines: dict[str, dict[str, Any]] = {}
    for baseline_name, baseline in (protocol.get("baseline_runs", {}) or {}).items():
        split_dirs = baseline.get("splits", {}) or {}
        split_payloads = {
            split: _validate_split_summary(_resolve_path(root, str(path)), str(baseline_name), split)
            for split, path in split_dirs.items()
        }
        manifest_baselines[str(baseline_name)] = {
            "name": str(baseline_name),
            "display_name": str(baseline.get("display_name", baseline_name)),
            "family": str(baseline.get("family", "baseline")),
            "provider_mode": str(baseline.get("provider_mode", "deterministic_policy")),
            "provenance": str(baseline.get("provenance", "unknown")),
            "splits": {
                split: {
                    "run_dir": str(_resolve_path(root, str(split_dirs[split])).relative_to(root)),
                    **_manifest_split_payload(payload),
                }
                for split, payload in split_payloads.items()
            },
        }

    manifest_diagnostics: dict[str, dict[str, Any]] = {}
    for diagnostic_run in _load_diagnostic_runs(protocol):
        split_payloads = _validate_named_summaries(diagnostic_run.run_dir, diagnostic_run.name, ("core", "heldout"))
        group = manifest_diagnostics.setdefault(
            diagnostic_run.diagnostic_name,
            {
                "name": diagnostic_run.diagnostic_name,
                "description": diagnostic_run.description,
                "settings": diagnostic_run.settings,
                "models": [],
            },
        )
        group["models"].append(
            {
                "name": diagnostic_run.name,
                "display_name": diagnostic_run.display_name,
                "family": diagnostic_run.family,
                "size_b": diagnostic_run.size_b,
                "provider_mode": diagnostic_run.provider_mode,
                "provenance": diagnostic_run.provenance,
                "run_dir": str(diagnostic_run.run_dir.relative_to(root)),
                "splits": {split_name: _manifest_split_payload(payload) for split_name, payload in split_payloads.items()},
            }
        )

    return {
        "paper": paper,
        "task_sets": protocol.get("task_sets", {}),
        "task_membership": task_membership,
        "metrics": protocol.get("metrics", {}),
        "analysis_questions": protocol.get("analysis_questions", []),
        "figure_plan": protocol.get("figure_plan", {}),
        "table_plan": protocol.get("table_plan", {}),
        "baseline_runs": manifest_baselines,
        "models": manifest_models,
        "diagnostic_runs": manifest_diagnostics,
    }


def _ensure_output_dirs(protocol: dict[str, Any]) -> dict[str, Path]:
    root = _repo_root()
    paper = protocol.get("paper", {})
    dirs = {
        "freeze_root": _resolve_path(root, str(paper["freeze_root"])),
        "canonical_run_root": _resolve_path(root, str(paper["canonical_run_root"])),
        "tables_dir": _resolve_path(root, str(paper["tables_dir"])),
        "figures_dir": _resolve_path(root, str(paper["figures_dir"])),
        "case_studies_dir": _resolve_path(root, str(paper["case_studies_dir"])),
        "metadata_dir": _resolve_path(root, str(paper["metadata_dir"])),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_paper_suite(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    protocol, models = _load_protocol(config_path)
    dirs = _ensure_output_dirs(protocol)
    manifest = build_manifest(protocol, models)
    metadata_root = dirs["metadata_dir"]
    (metadata_root / "paper_protocol_resolved.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (metadata_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "config_path": str(config_path),
        "freeze_root": str(dirs["freeze_root"]),
        "metadata_dir": str(metadata_root),
        "model_count": len(models),
        "primary_model_count": sum(1 for model in models if model.benchmark_role == "primary"),
        "external_anchor_count": sum(1 for model in models if model.benchmark_role == "external_anchor"),
        "main_task_count": len(protocol.get("task_sets", {}).get("main", []) or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_protocol.yaml")
    args = parser.parse_args()
    result = generate_paper_suite(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
