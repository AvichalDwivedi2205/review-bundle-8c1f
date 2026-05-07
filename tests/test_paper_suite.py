from __future__ import annotations

import json
from pathlib import Path

import yaml

from latentgoalops.analysis.make_paper_tables import generate_tables
from latentgoalops.paper.export_latex_tables import export_latex_tables
from latentgoalops.paper.run_paper_suite import generate_paper_suite


def _write_summary(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_mean_score": score,
        "strict_native_overall_mean_score": score,
        "assisted_overall_mean_score": score,
        "native_action_episode_rate": 1.0,
        "assisted_episode_rate": 1.0,
        "empty_fallback_episode_rate": 0.0,
        "response_cap_hit_episode_rate": 0.0,
        "parse_repaired_episode_count": 0,
        "parse_repaired_episode_rate": 0.0,
        "shifted_overall_mean_score": score - 0.1,
        "unshifted_overall_mean_score": score + 0.1,
        "episodes_expected": 30,
        "episodes_completed": 30,
        "run_complete": True,
        "benchmark_runtime_versions": ["test-runtime-v1"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_runs(path: Path, model_name: str, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task_ids = ["task3_startup_week", "task6_incident_response_week"]
    rows = []
    for index, score in enumerate(scores):
        rows.append(
            {
                "run_id": "test-run",
                "task_id": task_ids[index % len(task_ids)],
                "seed": 100 + index,
                "policy": "model" if model_name != "random" else "random",
                "model_name": model_name,
                "total_steps": 2,
                "score": score,
                "provider_usage": {},
                "metadata": {
                    "strict_episode": True,
                    "empty_fallback_episode": False,
                    "paper_eval": True,
                    "hidden_shift_present": index % 2 == 0,
                },
                "grade": {
                    "sub_scores": {
                        "final_utility": score,
                        "adaptation": score,
                        "coherence": score,
                        "constraints": score,
                    }
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_generate_paper_suite_writes_manifest(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    freeze_root = repo_root / "outputs" / "paper-freeze-test"
    run_root = freeze_root / "runs" / "toy-model"
    for split, score in [("smoke-core", 0.2), ("core", 0.4), ("heldout", 0.3)]:
        _write_summary(run_root / split / "summary.json", score)
    diagnostic_root = freeze_root / "diagnostics" / "shell-controlled" / "toy-model"
    for split, score in [("core", 0.35), ("heldout", 0.25)]:
        _write_summary(diagnostic_root / split / "summary.json", score)

    config_path = repo_root / "configs" / "paper_protocol.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "paper": {
            "paper_id": "test-paper",
            "freeze_root": "outputs/paper-freeze-test",
            "canonical_run_root": "outputs/paper-freeze-test/runs",
            "tables_dir": "outputs/paper-freeze-test/tables",
            "figures_dir": "outputs/paper-freeze-test/figures",
            "case_studies_dir": "outputs/paper-freeze-test/case_studies",
            "metadata_dir": "outputs/paper-freeze-test/metadata",
        },
        "task_sets": {
            "main": ["task3_startup_week", "task6_incident_response_week"],
            "supplemental": [],
            "appendix": [],
        },
        "metrics": {"primary_metric": "overall_mean_score"},
        "diagnostic_runs": {
            "shell_controlled": {
                "description": "unit test diagnostic",
                "settings": {"parse_repair_enabled": True},
                "models": [
                    {
                        "name": "toy-model",
                        "family": "toy",
                        "size_b": 1,
                        "provider_mode": "local",
                        "provenance": "unit-test-diagnostic",
                        "run_dir": "outputs/paper-freeze-test/diagnostics/shell-controlled/toy-model",
                    }
                ],
            }
        },
        "models": [
            {
                "name": "toy-model",
                "family": "toy",
                "size_b": 1,
                "provider_mode": "local",
                "provenance": "unit-test",
                "run_dir": "outputs/paper-freeze-test/runs/toy-model",
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("latentgoalops.paper.run_paper_suite._repo_root", lambda: repo_root)

    result = generate_paper_suite(config_path)

    assert result["model_count"] == 1
    metadata_root = freeze_root / "metadata"
    manifest = json.loads((metadata_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["models"][0]["name"] == "toy-model"
    assert manifest["models"][0]["splits"]["core"]["overall_mean_score"] == 0.4
    assert manifest["diagnostic_runs"]["shell_controlled"]["models"][0]["splits"]["core"]["overall_mean_score"] == 0.35


def test_generate_tables_writes_model_and_baseline_tables(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    freeze_root = repo_root / "outputs" / "paper-freeze-test"
    run_root = freeze_root / "runs" / "toy-model"
    anchor_run_root = freeze_root / "runs" / "anchor-model"
    for split, score in [("smoke-core", 0.2), ("core", 0.4), ("heldout", 0.3)]:
        _write_summary(run_root / split / "summary.json", score)
        _write_runs(run_root / split / "runs.jsonl", "toy-model", [score, score + 0.1])
        _write_summary(anchor_run_root / split / "summary.json", score - 0.05)
        _write_runs(anchor_run_root / split / "runs.jsonl", "anchor-model", [score - 0.05, score])

    baseline_root = freeze_root / "baselines" / "random"
    for split, score in [("core", 0.1), ("heldout", 0.2)]:
        _write_summary(baseline_root / split / "summary.json", score)
        _write_runs(baseline_root / split / "runs.jsonl", "random", [score, score + 0.1])
    diagnostic_root = freeze_root / "diagnostics" / "shell-controlled" / "toy-model"
    for split, score in [("core", 0.35), ("heldout", 0.25)]:
        _write_summary(diagnostic_root / split / "summary.json", score)

    config_path = repo_root / "configs" / "paper_protocol.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "paper": {
            "paper_id": "test-paper",
            "freeze_root": "outputs/paper-freeze-test",
            "canonical_run_root": "outputs/paper-freeze-test/runs",
            "tables_dir": "outputs/paper-freeze-test/tables",
            "figures_dir": "outputs/paper-freeze-test/figures",
            "case_studies_dir": "outputs/paper-freeze-test/case_studies",
            "metadata_dir": "outputs/paper-freeze-test/metadata",
        },
        "task_sets": {
            "main": ["task3_startup_week", "task6_incident_response_week"],
            "supplemental": [],
            "appendix": [],
        },
        "metrics": {"primary_metric": "overall_mean_score"},
        "diagnostic_runs": {
            "shell_controlled": {
                "description": "unit test diagnostic",
                "settings": {"parse_repair_enabled": True},
                "models": [
                    {
                        "name": "toy-model",
                        "family": "toy",
                        "size_b": 1,
                        "provider_mode": "local",
                        "provenance": "unit-test-diagnostic",
                        "run_dir": "outputs/paper-freeze-test/diagnostics/shell-controlled/toy-model",
                    }
                ],
            }
        },
        "baseline_runs": {
            "random": {
                "display_name": "random",
                "family": "baseline",
                "provider_mode": "deterministic_policy",
                "provenance": "unit-test",
                "splits": {
                    "core": "outputs/paper-freeze-test/baselines/random/core",
                    "heldout": "outputs/paper-freeze-test/baselines/random/heldout",
                },
            }
        },
        "models": [
            {
                "name": "toy-model",
                "family": "toy",
                "size_b": 1,
                "benchmark_role": "primary",
                "provider_mode": "local",
                "provenance": "unit-test",
                "run_dir": "outputs/paper-freeze-test/runs/toy-model",
            },
            {
                "name": "anchor-model",
                "display_name": "anchor-model",
                "family": "toy-anchor",
                "size_b": 2,
                "benchmark_role": "external_anchor",
                "provider_mode": "api",
                "provenance": "unit-test-anchor",
                "run_dir": "outputs/paper-freeze-test/runs/anchor-model",
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("latentgoalops.paper.run_paper_suite._repo_root", lambda: repo_root)
    monkeypatch.setattr("latentgoalops.analysis.make_paper_tables._repo_root", lambda: repo_root)
    monkeypatch.setattr("latentgoalops.paper.export_latex_tables._repo_root", lambda: repo_root)

    result = generate_tables(config_path)

    assert result["model_count"] == 1
    assert result["external_anchor_count"] == 1
    assert result["baseline_count"] == 1
    tables_root = freeze_root / "tables"
    assert (tables_root / "main_results.csv").exists()
    assert (tables_root / "external_anchor_results.csv").exists()
    assert (tables_root / "baseline_results.csv").exists()
    assert (tables_root / "mechanism_diagnostics.csv").exists()
    assert (tables_root / "adaptation_audit.csv").exists()
    assert (tables_root / "shell_controlled_diagnostic.csv").exists()

    main_results = json.loads((tables_root / "main_results.json").read_text(encoding="utf-8"))
    anchor_results = json.loads((tables_root / "external_anchor_results.json").read_text(encoding="utf-8"))
    assert {row["model"] for row in main_results} == {"toy-model"}
    assert {row["model"] for row in anchor_results} == {"anchor-model"}

    latex_result = export_latex_tables(config_path)
    generated_root = repo_root / "paper" / "latex" / "generated"
    assert latex_result["file_count"] >= 5
    assert (generated_root / "main_results_table.tex").exists()
    assert (generated_root / "external_anchor_table.tex").exists()
    assert (generated_root / "baseline_table.tex").exists()
    assert (generated_root / "shell_controlled_diagnostic_table.tex").exists()
