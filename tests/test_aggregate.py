"""Aggregation integrity tests."""

from __future__ import annotations

import json
import math

from latentgoalops.analysis.aggregate import load_run_records


def test_load_run_records_dedupes_duplicate_episode_rows(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "runs.jsonl"
    payload = {
        "run_id": "dup-run",
        "task_id": "task1_feedback_triage",
        "seed": 100,
        "policy": "model",
        "model_name": "openai-gpt-oss-20b",
        "score": 0.8,
        "total_steps": 1,
        "elapsed_seconds": 1.0,
        "provider_usage": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0},
        "grade": {"sub_scores": {}},
        "metadata": {"strict_episode": True, "rescued_episode": False, "paper_eval": True},
    }
    path.write_text("\n".join([json.dumps(payload), json.dumps(payload)]), encoding="utf-8")

    steps, episodes = load_run_records(tmp_path)

    assert steps.empty
    assert len(episodes) == 1
    assert episodes.attrs["duplicate_episode_rows"] == 1


def test_load_run_records_flattens_runtime_and_cap_hit_fields(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "runs.jsonl"
    episode_payload = {
        "run_id": "runtime-run",
        "task_id": "task1_feedback_triage",
        "seed": 100,
        "policy": "model",
        "model_name": "openai-gpt-oss-20b",
        "score": 0.5,
        "total_steps": 1,
        "elapsed_seconds": 1.0,
        "provider_usage": {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "response_cap_hit": True,
            "finish_reason_counts": {"length": 1},
            "invalid_response_excerpt": "{\"task_id\":\"task1_feedback_triage\"",
        },
        "grade": {"sub_scores": {}},
        "metadata": {
            "strict_episode": False,
            "rescued_episode": False,
            "paper_eval": True,
            "benchmark_runtime_version": "runtime-v1",
            "episodes_completed": 1,
            "episodes_total": 7,
        },
    }
    step_payload = {
        "run_id": "runtime-run",
        "task_id": "task1_feedback_triage",
        "seed": 100,
        "policy": "model",
        "model_name": "openai-gpt-oss-20b",
        "step_index": 0,
        "reward": 0.1,
        "provider_usage": {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "response_cap_hit": True,
            "finish_reason": "length",
        },
        "metadata": {"paper_eval": True, "benchmark_runtime_version": "runtime-v1"},
    }
    path.write_text("\n".join([json.dumps(step_payload), json.dumps(episode_payload)]), encoding="utf-8")

    steps, episodes = load_run_records(tmp_path)

    assert steps.iloc[0]["response_cap_hit"] == 1
    assert steps.iloc[0]["finish_reason"] == "length"
    assert steps.iloc[0]["benchmark_runtime_version"] == "runtime-v1"
    assert episodes.iloc[0]["response_cap_hit"] == 1
    assert episodes.iloc[0]["invalid_response_excerpt"] == "{\"task_id\":\"task1_feedback_triage\""
    assert episodes.iloc[0]["finish_reason_counts"] == {"length": 1}
    assert episodes.iloc[0]["benchmark_runtime_version"] == "runtime-v1"
    assert episodes.iloc[0]["episodes_completed"] == 1
    assert episodes.iloc[0]["episodes_total"] == 7


def test_load_run_records_renormalizes_decision_only_score_without_adaptation(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "runs.jsonl"
    payload = {
        "run_id": "decision-run",
        "task_id": "task7_quarterly_headcount_plan",
        "seed": 101,
        "policy": "oracle",
        "model_name": "oracle",
        "score": 0.7,
        "total_steps": 4,
        "elapsed_seconds": 1.0,
        "provider_usage": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0},
        "grade": {
            "sub_scores": {
                "final_utility": 0.7,
                "coherence": 0.6,
                "constraints": 0.8,
                "belief_tracking": 0.5,
            },
            "details": {"adaptation_scored": False},
        },
        "metadata": {"paper_eval": True, "benchmark_runtime_version": "runtime-v1"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, episodes = load_run_records(tmp_path)

    expected = (0.6364 * 0.7 + 0.0568 * 0.6 + 0.0454 * 0.8) / (0.6364 + 0.0568 + 0.0454)
    assert episodes.iloc[0]["adaptation_scored"] == 0
    assert math.isclose(float(episodes.iloc[0]["decision_only_score"]), expected)
