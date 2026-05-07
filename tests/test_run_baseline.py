"""Baseline-runner helper tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from latentgoalops.baseline.run_baseline import (
    BENCHMARK_RUNTIME_VERSION,
    DEFAULT_AGENT_MODEL,
    DEFAULT_PERSONA_MODEL,
    _completion_budget,
    _checkpoint_signature,
    _load_checkpoint,
    _logged_episode_payloads,
    _parse_budget_cap_arg,
    _reasoning_effort_for_step,
    _response_excerpt,
    _summarize_logged_episodes,
    run,
)
from latentgoalops.experiment import ExperimentConfig
from latentgoalops.baseline.run_model_ladder import _parse_models
from latentgoalops.models import TaskId


def test_logged_episode_payloads_handles_missing_file(tmp_path):
    assert _logged_episode_payloads(tmp_path / "missing.jsonl") == []


def test_logged_episode_payloads_filters_non_episode_rows(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step_index": 0, "reward": 0.1}),
                json.dumps({"task_id": "task1_feedback_triage", "score": 0.8, "total_steps": 1}),
            ]
        ),
        encoding="utf-8",
    )
    assert _logged_episode_payloads(path) == [
        {"task_id": "task1_feedback_triage", "score": 0.8, "total_steps": 1}
    ]


def test_parse_budget_cap_arg_supports_uncapped_mode():
    assert _parse_budget_cap_arg(None) is None
    assert _parse_budget_cap_arg("none") is None
    assert _parse_budget_cap_arg("uncapped") is None
    assert _parse_budget_cap_arg("2.5") == 2.5


def test_short_structured_tasks_get_extra_budget_and_no_reasoning_effort():
    assert _completion_budget(TaskId.TASK1) == 2200
    assert _completion_budget(TaskId.TASK2) == 2200
    assert _completion_budget(TaskId.TASK3) == 2400
    assert _completion_budget(TaskId.TASK4) == 3000
    assert _completion_budget(TaskId.TASK6) == 2200
    assert _completion_budget(TaskId.TASK7) == 3000
    assert _reasoning_effort_for_step(TaskId.TASK1) is None
    assert _reasoning_effort_for_step(TaskId.TASK2) is None
    assert _reasoning_effort_for_step(TaskId.TASK3) is None
    assert _reasoning_effort_for_step(TaskId.TASK4) is None
    assert _reasoning_effort_for_step(TaskId.TASK6) is None
    assert _reasoning_effort_for_step(TaskId.TASK7) is None
    assert _reasoning_effort_for_step(TaskId.TASK5) == "low"


def test_response_excerpt_compacts_whitespace_and_truncates():
    excerpt = _response_excerpt("  {\n  \"hello\":   \"world\" \n}\n", limit=12)
    assert excerpt is not None
    assert excerpt.startswith("{")
    assert excerpt.endswith("...")
    assert '"hello"' in excerpt


def test_model_ladder_parse_models_supports_explicit_model_lists():
    assert _parse_models("openai-gpt-oss-20b,kimi-k2.5,glm-5", "model", DEFAULT_PERSONA_MODEL, "openai_compat") == [
        "openai-gpt-oss-20b",
        "kimi-k2.5",
        "glm-5",
    ]


def test_model_ladder_defaults_to_local_open_models_for_ollama():
    assert _parse_models("default", "model", DEFAULT_PERSONA_MODEL, "ollama") == [
        "qwen3:8b",
        "qwen3:14b",
        "gemma3:12b",
        "gpt-oss:20b",
        "qwen3:30b",
    ]


def test_run_accepts_uncapped_budget(tmp_path):
    means = run(
        SimpleNamespace(
            policy="random",
            model=DEFAULT_AGENT_MODEL,
            persona_model=DEFAULT_PERSONA_MODEL,
            operator_style="auto",
            tasks="task1_feedback_triage",
            seeds="100:1",
            output_dir=str(tmp_path),
            run_id="random-uncapped-smoke",
            budget_cap=None,
            temperature=0.0,
            provider="openai_compat",
            api_base_url=None,
            ollama_host=None,
            ollama_num_ctx=None,
            ollama_keep_alive="15m",
            ollama_think="auto",
            disable_hidden_shift=False,
            disable_delayed_effects=False,
            hide_decision_ledger=False,
            reward_mode="shaped",
            task3_horizon_override=None,
            scenario_split="core",
            paper_eval=False,
            disable_parse_repair=False,
            disable_heuristic_rescue=False,
            enable_task2_visible_floor=False,
            wandb=False,
            wandb_project="latentgoalops",
            wandb_entity=None,
            wandb_group=None,
            wandb_tags="",
        )
    )

    assert "task1_feedback_triage" in means
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["episode_count"] == 1
    assert summary["run_complete"] is True
    assert summary["overall_mean_score"] == summary["mean_scores"]["task1_feedback_triage"]


def test_summarize_logged_episodes_recomputes_assistance_counts():
    episodes = [
        {
            "run_id": "assist-run",
            "task_id": "task1_feedback_triage",
            "seed": 100,
            "policy": "model",
            "model_name": DEFAULT_AGENT_MODEL,
            "score": 0.8,
            "provider_usage": {
                "response_cap_hit": False,
                "parse_fallback": False,
                "parse_repaired": False,
                "heuristic_rescue": False,
                "empty_fallback": False,
                "task2_visible_floor_applied": False,
                "response_cap_hit_steps": 0,
                "parse_fallback_steps": 0,
                "parse_repaired_steps": 0,
                "heuristic_rescue_steps": 0,
                "empty_fallback_steps": 0,
                "task2_visible_floor_steps": 0,
                "finish_reason_counts": {"stop": 1},
            },
            "metadata": {"strict_episode": True, "rescued_episode": False, "hidden_shift_present": False},
        },
        {
            "run_id": "assist-run",
            "task_id": "task1_feedback_triage",
            "seed": 101,
            "policy": "model",
            "model_name": DEFAULT_AGENT_MODEL,
            "score": 0.1,
            "provider_usage": {
                "response_cap_hit": True,
                "parse_fallback": True,
                "parse_repaired": False,
                "heuristic_rescue": False,
                "empty_fallback": True,
                "task2_visible_floor_applied": False,
                "response_cap_hit_steps": 1,
                "parse_fallback_steps": 2,
                "parse_repaired_steps": 0,
                "heuristic_rescue_steps": 0,
                "empty_fallback_steps": 2,
                "task2_visible_floor_steps": 0,
                "finish_reason_counts": {"length": 1},
            },
            "metadata": {"strict_episode": False, "rescued_episode": False, "hidden_shift_present": False},
        },
        {
            "run_id": "assist-run",
            "task_id": "task1_feedback_triage",
            "seed": 102,
            "policy": "model",
            "model_name": DEFAULT_AGENT_MODEL,
            "score": 0.5,
            "provider_usage": {
                "response_cap_hit": False,
                "parse_fallback": True,
                "parse_repaired": False,
                "heuristic_rescue": True,
                "empty_fallback": False,
                "task2_visible_floor_applied": False,
                "response_cap_hit_steps": 0,
                "parse_fallback_steps": 1,
                "parse_repaired_steps": 0,
                "heuristic_rescue_steps": 1,
                "empty_fallback_steps": 0,
                "task2_visible_floor_steps": 0,
                "finish_reason_counts": {"stop": 1},
            },
            "metadata": {"strict_episode": False, "rescued_episode": True, "hidden_shift_present": False},
        },
    ]
    summary = _summarize_logged_episodes(
        ["task1_feedback_triage"],
        episodes,
        paper_eval=False,
        expected_episode_count=3,
        completion_status="complete",
    )
    assert summary["episode_count"] == 3
    assert summary["episodes_expected"] == 3
    assert summary["run_complete"] is True
    assert summary["strict_episode_count"] == 1
    assert summary["rescued_episode_count"] == 1
    assert summary["empty_fallback_episode_count"] == 1
    assert summary["parse_fallback_episode_count"] == 2
    assert summary["heuristic_rescue_episode_count"] == 1
    assert summary["response_cap_hit_episode_count"] == 1
    assert summary["parse_fallback_step_count"] == 3
    assert summary["empty_fallback_step_count"] == 2
    assert summary["response_cap_hit_step_count"] == 1
    assert summary["finish_reason_counts"] == {"stop": 2, "length": 1}


def test_summarize_logged_episodes_dedupes_duplicate_terminal_rows():
    payload = {
        "run_id": "dup-run",
        "task_id": "task1_feedback_triage",
        "seed": 100,
        "policy": "model",
        "model_name": DEFAULT_AGENT_MODEL,
        "score": 0.8,
        "provider_usage": {
            "response_cap_hit": False,
            "parse_fallback": False,
            "parse_repaired": False,
            "heuristic_rescue": False,
            "empty_fallback": False,
            "task2_visible_floor_applied": False,
            "response_cap_hit_steps": 0,
            "parse_fallback_steps": 0,
            "parse_repaired_steps": 0,
            "heuristic_rescue_steps": 0,
            "empty_fallback_steps": 0,
            "task2_visible_floor_steps": 0,
            "finish_reason_counts": {"stop": 1},
        },
        "metadata": {"strict_episode": True, "rescued_episode": False, "hidden_shift_present": False},
    }
    summary = _summarize_logged_episodes(
        ["task1_feedback_triage"],
        [payload, payload],
        paper_eval=True,
        expected_episode_count=1,
        completion_status="complete",
    )
    assert summary["episode_count"] == 1
    assert summary["duplicate_episode_row_count"] == 1
    assert summary["mean_scores"]["task1_feedback_triage"] == 0.8


def test_summarize_logged_episodes_marks_incomplete_runs():
    payload = {
        "run_id": "partial-run",
        "task_id": "task1_feedback_triage",
        "seed": 100,
        "policy": "model",
        "model_name": DEFAULT_AGENT_MODEL,
        "score": 0.8,
        "provider_usage": {"finish_reason_counts": {"stop": 1}},
        "metadata": {"strict_episode": True, "rescued_episode": False, "hidden_shift_present": False},
    }
    summary = _summarize_logged_episodes(
        ["task1_feedback_triage"],
        [payload],
        paper_eval=True,
        expected_episode_count=3,
        completion_status="interrupted_transient",
    )
    assert summary["episodes_completed"] == 1
    assert summary["episodes_expected"] == 3
    assert summary["run_complete"] is False
    assert summary["completion_status"] == "interrupted_transient"


def test_load_checkpoint_rejects_signature_mismatch(tmp_path):
    signature = _checkpoint_signature(
        run_id="run-a",
        provider_model=DEFAULT_AGENT_MODEL,
        benchmark_model=DEFAULT_AGENT_MODEL,
        persona_model=DEFAULT_PERSONA_MODEL,
        policy="model",
        tasks=["task1_feedback_triage"],
        seeds=[100],
        temperature=0.0,
        provider_name="openai_compat",
        provider_settings={
            "api_base_url": None,
            "ollama_host": None,
            "ollama_num_ctx": None,
            "ollama_keep_alive": "15m",
            "ollama_think": "auto",
        },
        experiment_config=ExperimentConfig(),
        eval_flags={
            "paper_eval": True,
            "allow_parse_repair": False,
            "allow_heuristic_rescue": False,
            "allow_task2_visible_floor": False,
        },
    )
    assert signature["benchmark_runtime_version"] == BENCHMARK_RUNTIME_VERSION
    checkpoint_path = tmp_path / "checkpoint.json"
    _load_checkpoint(
        checkpoint_path,
        "run-a",
        DEFAULT_AGENT_MODEL,
        "model",
        ["task1_feedback_triage"],
        [100],
        signature,
    )
    other_signature = dict(signature)
    other_signature["paper_eval"] = False
    try:
        _load_checkpoint(
            checkpoint_path,
            "run-a",
            DEFAULT_AGENT_MODEL,
            "model",
            ["task1_feedback_triage"],
            [100],
            other_signature,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected checkpoint signature mismatch to raise.")
