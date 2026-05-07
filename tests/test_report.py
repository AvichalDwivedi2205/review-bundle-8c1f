"""Report-generation slice tests."""

from __future__ import annotations

import pandas as pd

from latentgoalops.analysis.report import _primary_episode_slice, _primary_step_slice, _validate_report_inputs


def test_report_prefers_strict_episode_rows_when_available():
    episodes = pd.DataFrame(
        [
            {"task_id": "task2_roadmap_priority", "score": 0.9, "strict_episode": 1},
            {"task_id": "task2_roadmap_priority", "score": 0.4, "strict_episode": 0},
        ]
    )
    primary, mode = _primary_episode_slice(episodes)
    assert mode == "all"
    assert len(primary) == 2


def test_report_keeps_all_rows_when_assistance_is_present():
    episodes = pd.DataFrame(
        [
            {"task_id": "task2_roadmap_priority", "score": 0.9, "strict_episode": 1, "rescued_episode": 0},
            {"task_id": "task2_roadmap_priority", "score": 0.4, "strict_episode": 0, "rescued_episode": 1},
        ]
    )
    primary, mode = _primary_episode_slice(episodes)
    assert mode == "all_with_assistance"
    assert len(primary) == 2


def test_report_prefers_strict_step_rows_when_available():
    steps = pd.DataFrame(
        [
            {"task_id": "task3_startup_week", "step_index": 0, "reward": 0.2, "strict_step": 1},
            {"task_id": "task3_startup_week", "step_index": 0, "reward": 0.8, "strict_step": 0},
        ]
    )
    primary, mode = _primary_step_slice(steps)
    assert mode == "all"
    assert len(primary) == 2


def test_report_validation_rejects_duplicate_rows():
    episodes = pd.DataFrame([{"task_id": "task1_feedback_triage", "paper_eval": 1}])
    steps = pd.DataFrame()
    episodes.attrs["duplicate_episode_rows"] = 1
    steps.attrs["duplicate_step_rows"] = 0
    try:
        _validate_report_inputs(steps, episodes)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate-row validation to raise.")


def test_report_validation_rejects_mixed_paper_eval_inputs():
    episodes = pd.DataFrame(
        [
            {"task_id": "task1_feedback_triage", "paper_eval": 1},
            {"task_id": "task2_roadmap_priority", "paper_eval": 0},
        ]
    )
    steps = pd.DataFrame()
    episodes.attrs["duplicate_episode_rows"] = 0
    steps.attrs["duplicate_step_rows"] = 0
    try:
        _validate_report_inputs(steps, episodes)
    except ValueError as exc:
        assert "Mixed paper-eval" in str(exc)
    else:
        raise AssertionError("Expected mixed protocol validation to raise.")


def test_report_validation_rejects_mixed_runtime_versions():
    episodes = pd.DataFrame(
        [
            {"task_id": "task1_feedback_triage", "paper_eval": 1, "benchmark_runtime_version": "runtime-v1"},
            {"task_id": "task2_roadmap_priority", "paper_eval": 1, "benchmark_runtime_version": "runtime-v2"},
        ]
    )
    steps = pd.DataFrame()
    episodes.attrs["duplicate_episode_rows"] = 0
    steps.attrs["duplicate_step_rows"] = 0
    try:
        _validate_report_inputs(steps, episodes)
    except ValueError as exc:
        assert "Mixed benchmark runtime versions" in str(exc)
    else:
        raise AssertionError("Expected mixed runtime-version validation to raise.")


def test_report_validation_rejects_incomplete_runs():
    episodes = pd.DataFrame(
        [
            {
                "task_id": "task3_startup_week",
                "paper_eval": 1,
                "benchmark_runtime_version": "runtime-v1",
                "run_id": "strict-run",
                "model_name": "openai-gpt-oss-20b",
                "policy": "model",
                "episodes_total": 3,
            },
            {
                "task_id": "task6_incident_response_week",
                "paper_eval": 1,
                "benchmark_runtime_version": "runtime-v1",
                "run_id": "strict-run",
                "model_name": "openai-gpt-oss-20b",
                "policy": "model",
                "episodes_total": 3,
            },
        ]
    )
    steps = pd.DataFrame()
    episodes.attrs["duplicate_episode_rows"] = 0
    steps.attrs["duplicate_step_rows"] = 0
    try:
        _validate_report_inputs(steps, episodes)
    except ValueError as exc:
        assert "Incomplete runs detected" in str(exc)
    else:
        raise AssertionError("Expected incomplete-run validation to raise.")
