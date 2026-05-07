"""Baseline runner for LatentGoalOps."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.baseline.heuristics import choose_action
from latentgoalops.baseline.parsers import parse_action
from latentgoalops.baseline.prompts import output_schema, system_prompt, user_prompt
from latentgoalops.baseline.providers import (
    ChatProvider,
    ProviderCredentialError,
    ProviderExhaustedError,
    ProviderTransientError,
    create_chat_provider,
)
from latentgoalops.baseline.synthetic_operator import (
    apply_operator_guardrails,
    operator_style_choices,
    resolve_operator_persona,
    stabilize_model_action,
)
from latentgoalops.logging_.jsonl_logger import JsonlLogger
from latentgoalops.logging_.schemas import CheckpointState, EpisodeSummary, StepLog
from latentgoalops.logging_.wandb_logger import WandbLogger
from latentgoalops.models import LatentGoalOpsAction, TaskId
from latentgoalops.server.environment import LatentGoalOpsEnvironment

DEFAULT_AGENT_MODEL = "openai-gpt-oss-20b"
DEFAULT_PERSONA_MODEL = "openai-gpt-oss-20b"
BENCHMARK_RUNTIME_VERSION = "2026-04-07-shell-overhaul-v6"
SHORT_STRUCTURED_TASKS = {
    TaskId.TASK1,
    TaskId.TASK2,
    TaskId.TASK3,
    TaskId.TASK4,
    TaskId.TASK6,
    TaskId.TASK7,
}


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    whole = max(0, int(round(value)))
    hours, rem = divmod(whole, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_tasks(task_arg: str) -> list[str]:
    if task_arg == "all":
        return [task.value for task in TaskId]
    return [task.strip() for task in task_arg.split(",") if task.strip()]


def _parse_seeds(seed_arg: str) -> list[int]:
    if ":" in seed_arg:
        start, count = [int(part) for part in seed_arg.split(":", 1)]
        return list(range(start, start + count))
    return [int(part) for part in seed_arg.split(",") if part.strip()]


def _parse_budget_cap_arg(raw: str | float | int | None) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    value = str(raw).strip()
    if not value or value.lower() in {"none", "null", "inf", "infinity", "unlimited", "uncapped"}:
        return None
    return float(value)


def _experiment_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        enable_hidden_shift=not bool(getattr(args, "disable_hidden_shift", False)),
        enable_delayed_effects=not bool(getattr(args, "disable_delayed_effects", False)),
        expose_decision_ledger=not bool(getattr(args, "hide_decision_ledger", False)),
        reward_mode=str(getattr(args, "reward_mode", "shaped")),
        task3_horizon_override=getattr(args, "task3_horizon_override", None),
        scenario_split=str(getattr(args, "scenario_split", "core")),
    )


def _checkpoint_signature(
    *,
    run_id: str,
    provider_model: str,
    benchmark_model: str,
    persona_model: str,
    policy: str,
    tasks: list[str],
    seeds: list[int],
    temperature: float,
    provider_name: str,
    provider_settings: dict[str, object | None],
    experiment_config: ExperimentConfig,
    eval_flags: dict[str, bool],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "provider_model": provider_model,
        "benchmark_model": benchmark_model,
        "persona_model": persona_model if policy == "synthetic_operator" else None,
        "policy": policy,
        "tasks": list(tasks),
        "seeds": list(seeds),
        "temperature": float(temperature),
        "provider": provider_name,
        "provider_settings": provider_settings,
        "experiment_config": experiment_config.model_dump(mode="json"),
        "benchmark_runtime_version": BENCHMARK_RUNTIME_VERSION,
        "paper_eval": bool(eval_flags["paper_eval"]),
        "allow_parse_repair": bool(eval_flags["allow_parse_repair"]),
        "allow_heuristic_rescue": bool(eval_flags["allow_heuristic_rescue"]),
        "allow_task2_visible_floor": bool(eval_flags["allow_task2_visible_floor"]),
    }


def _load_checkpoint(
    path: Path,
    run_id: str,
    model_name: str,
    policy: str,
    tasks: list[str],
    seeds: list[int],
    signature: dict[str, object],
) -> CheckpointState:
    if path.exists():
        checkpoint = CheckpointState.model_validate_json(path.read_text(encoding="utf-8"))
        mismatches: list[str] = []
        if checkpoint.run_id != run_id:
            mismatches.append("run_id")
        if checkpoint.model_name != model_name:
            mismatches.append("model_name")
        if checkpoint.policy != policy:
            mismatches.append("policy")
        if checkpoint.tasks != tasks:
            mismatches.append("tasks")
        if checkpoint.seeds != seeds:
            mismatches.append("seeds")
        if mismatches:
            raise ValueError(
                "Existing checkpoint does not match the requested run configuration "
                f"({', '.join(mismatches)}). Use a fresh output directory or delete the old checkpoint."
            )
        saved_signature = checkpoint.metadata.get("signature")
        if saved_signature is None:
            if checkpoint.next_index > 0:
                raise ValueError(
                    "Existing checkpoint predates run-signature validation and cannot be safely resumed. "
                    "Use a fresh output directory or delete the old checkpoint + logs."
                )
            checkpoint.metadata["signature"] = signature
            _save_checkpoint(path, checkpoint)
        elif saved_signature != signature:
            raise ValueError(
                "Existing checkpoint metadata does not match the requested run configuration. "
                "Use a fresh output directory or delete the old checkpoint + logs."
            )
        return checkpoint
    checkpoint = CheckpointState(
        run_id=run_id,
        model_name=model_name,
        policy=policy,
        tasks=tasks,
        seeds=seeds,
        metadata={"signature": signature},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(path, checkpoint)
    return checkpoint


def _completion_budget(task_id: TaskId, policy_mode: str = "model") -> int:
    if policy_mode == "synthetic_operator":
        if task_id in {TaskId.TASK1, TaskId.TASK2}:
            return 2600
        if task_id == TaskId.TASK3:
            return 2800
        if task_id in {TaskId.TASK4, TaskId.TASK7}:
            return 2800
        if task_id == TaskId.TASK5:
            return 2600
        if task_id == TaskId.TASK6:
            return 2200
        return 2400
    if task_id in {TaskId.TASK1, TaskId.TASK2}:
        return 2200
    if task_id == TaskId.TASK3:
        return 2400
    if task_id == TaskId.TASK5:
        return 1800
    if task_id == TaskId.TASK6:
        return 2200
    if task_id in {TaskId.TASK4, TaskId.TASK7}:
        return 3000
    return 1800


def _reasoning_effort_for_step(task_id: TaskId, policy_mode: str = "model") -> str | None:
    if task_id in SHORT_STRUCTURED_TASKS:
        return None
    if policy_mode in {"model", "synthetic_operator"}:
        return "low"
    return None


def _response_excerpt(text: str, limit: int = 240) -> str | None:
    compact = " ".join(str(text).split())
    if not compact:
        return None
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(limit - 3, 0)]}..."


def _empty_action(task_id: TaskId) -> LatentGoalOpsAction:
    return LatentGoalOpsAction(task_id=task_id)


def _resolve_provider_model(args: argparse.Namespace) -> str:
    """Return the model that should actually make completions for this run."""
    if args.policy == "synthetic_operator":
        return str(getattr(args, "persona_model", DEFAULT_PERSONA_MODEL))
    return str(getattr(args, "model", DEFAULT_AGENT_MODEL))


def _display_model_name(policy: str, model: str, persona_model: str | None = None) -> str:
    if policy == "model":
        return model
    if policy == "synthetic_operator":
        resolved_persona = persona_model or DEFAULT_PERSONA_MODEL
        return f"persona::{resolved_persona}"
    return policy


def _save_checkpoint(path: Path, checkpoint: CheckpointState) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _logged_episode_payloads(path: Path) -> list[dict]:
    """Read episode summaries from the JSONL log if any were written."""
    if not path.exists():
        return []
    episodes = []
    for jsonl_line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(jsonl_line)
        if "total_steps" in payload and "score" in payload:
            episodes.append(payload)
    return episodes


def _episode_identity_fields(run_id: str, task_id: str, seed: int, policy: str, model_name: str) -> tuple[str, str, int, str, str]:
    return run_id, task_id, int(seed), policy, model_name


def _episode_identity(payload: dict) -> tuple[str, str, int, str, str]:
    return _episode_identity_fields(
        str(payload.get("run_id", "")),
        str(payload.get("task_id", "")),
        int(payload.get("seed", 0)),
        str(payload.get("policy", payload.get("metadata", {}).get("policy", ""))),
        str(payload.get("model_name", "")),
    )


def _dedupe_episode_payloads(episodes: list[dict]) -> tuple[list[dict], int]:
    deduped: dict[tuple[str, str, int, str, str], dict] = {}
    duplicate_count = 0
    for payload in episodes:
        key = _episode_identity(payload)
        if key in deduped:
            duplicate_count += 1
        deduped[key] = payload
    return list(deduped.values()), duplicate_count


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _effective_eval_flags(args: argparse.Namespace) -> dict[str, bool]:
    paper_eval = bool(getattr(args, "paper_eval", False))
    allow_parse_repair = not bool(getattr(args, "disable_parse_repair", False))
    allow_heuristic_rescue = not bool(getattr(args, "disable_heuristic_rescue", False))
    allow_task2_visible_floor = bool(getattr(args, "enable_task2_visible_floor", False))
    if paper_eval:
        allow_parse_repair = False
        allow_heuristic_rescue = False
        allow_task2_visible_floor = False
    return {
        "paper_eval": paper_eval,
        "allow_parse_repair": allow_parse_repair,
        "allow_heuristic_rescue": allow_heuristic_rescue,
        "allow_task2_visible_floor": allow_task2_visible_floor,
    }


def _summarize_logged_episodes(
    tasks: list[str],
    episodes: list[dict],
    paper_eval: bool,
    *,
    expected_episode_count: int | None = None,
    completion_status: str = "complete",
) -> dict[str, object]:
    deduped_episodes, duplicate_episode_row_count = _dedupe_episode_payloads(episodes)
    score_table: dict[str, list[float]] = {task: [] for task in tasks}
    native_score_table: dict[str, list[float]] = {task: [] for task in tasks}
    assisted_score_table: dict[str, list[float]] = {task: [] for task in tasks}
    shifted_score_table: dict[str, list[float]] = {task: [] for task in tasks}
    unshifted_score_table: dict[str, list[float]] = {task: [] for task in tasks}

    strict_episode_count = 0
    assisted_episode_count = 0
    heuristic_rescue_episode_count = 0
    empty_fallback_episode_count = 0
    parse_fallback_episode_count = 0
    parse_repaired_episode_count = 0
    task2_visible_floor_episode_count = 0
    response_cap_hit_episode_count = 0
    parse_fallback_step_count = 0
    parse_repaired_step_count = 0
    heuristic_rescue_step_count = 0
    empty_fallback_step_count = 0
    task2_visible_floor_step_count = 0
    response_cap_hit_step_count = 0
    finish_reason_counts: dict[str, int] = {}

    for payload in deduped_episodes:
        task = payload["task_id"]
        score = float(payload["score"])
        metadata = payload.get("metadata", {}) or {}
        usage = payload.get("provider_usage", {}) or {}

        score_table.setdefault(task, []).append(score)
        if metadata.get("strict_episode", False):
            native_score_table.setdefault(task, []).append(score)
            strict_episode_count += 1
        if metadata.get("rescued_episode", False):
            assisted_score_table.setdefault(task, []).append(score)
            assisted_episode_count += 1
        if metadata.get("hidden_shift_present", False):
            shifted_score_table.setdefault(task, []).append(score)
        else:
            unshifted_score_table.setdefault(task, []).append(score)

        parse_fallback_episode_count += int(bool(usage.get("parse_fallback", False)))
        parse_repaired_episode_count += int(bool(usage.get("parse_repaired", False)))
        heuristic_rescue_episode_count += int(bool(usage.get("heuristic_rescue", False)))
        empty_fallback_episode_count += int(bool(usage.get("empty_fallback", False)))
        task2_visible_floor_episode_count += int(bool(usage.get("task2_visible_floor_applied", False)))
        response_cap_hit_episode_count += int(bool(usage.get("response_cap_hit", False)))
        parse_fallback_step_count += int(usage.get("parse_fallback_steps", 0) or 0)
        parse_repaired_step_count += int(usage.get("parse_repaired_steps", 0) or 0)
        heuristic_rescue_step_count += int(usage.get("heuristic_rescue_steps", 0) or 0)
        empty_fallback_step_count += int(usage.get("empty_fallback_steps", 0) or 0)
        task2_visible_floor_step_count += int(usage.get("task2_visible_floor_steps", 0) or 0)
        response_cap_hit_step_count += int(usage.get("response_cap_hit_steps", 0) or 0)
        for finish_reason, count in (usage.get("finish_reason_counts", {}) or {}).items():
            finish_reason_counts[str(finish_reason)] = finish_reason_counts.get(str(finish_reason), 0) + int(count)

    episode_count = len(deduped_episodes)
    all_scores = [float(payload["score"]) for payload in deduped_episodes]
    strict_scores = [score for scores in native_score_table.values() for score in scores]
    assisted_scores = [score for scores in assisted_score_table.values() for score in scores]
    shifted_scores = [score for scores in shifted_score_table.values() for score in scores]
    unshifted_scores = [score for scores in unshifted_score_table.values() for score in scores]
    runtime_versions = sorted(
        {
            str((payload.get("metadata", {}) or {}).get("benchmark_runtime_version"))
            for payload in deduped_episodes
            if (payload.get("metadata", {}) or {}).get("benchmark_runtime_version")
        }
    )
    episodes_expected = expected_episode_count if expected_episode_count is not None else episode_count
    run_complete = completion_status == "complete" and episode_count >= episodes_expected
    return {
        "mean_scores": {task: _mean_or_zero(scores) for task, scores in score_table.items()},
        "strict_mean_scores": {task: _mean_or_zero(scores) for task, scores in native_score_table.items()},
        "rescued_mean_scores": {task: _mean_or_zero(scores) for task, scores in assisted_score_table.items()},
        "shifted_mean_scores": {task: _mean_or_zero(scores) for task, scores in shifted_score_table.items()},
        "unshifted_mean_scores": {task: _mean_or_zero(scores) for task, scores in unshifted_score_table.items()},
        "primary_mean_scores": {task: _mean_or_zero(scores) for task, scores in score_table.items()} if paper_eval else {},
        "primary_metric": "score" if paper_eval else "mixed_score",
        "paper_safe": bool(paper_eval),
        "overall_mean_score": _mean_or_zero(all_scores),
        "strict_native_overall_mean_score": _mean_or_zero(strict_scores),
        "assisted_overall_mean_score": _mean_or_zero(assisted_scores),
        "shifted_overall_mean_score": _mean_or_zero(shifted_scores),
        "unshifted_overall_mean_score": _mean_or_zero(unshifted_scores),
        "episodes_expected": episodes_expected,
        "episode_count": episode_count,
        "episodes_completed": episode_count,
        "run_complete": run_complete,
        "completion_status": completion_status,
        "duplicate_episode_row_count": duplicate_episode_row_count,
        "strict_episode_count": strict_episode_count,
        "rescued_episode_count": assisted_episode_count,
        "heuristic_rescue_episode_count": heuristic_rescue_episode_count,
        "empty_fallback_episode_count": empty_fallback_episode_count,
        "parse_fallback_episode_count": parse_fallback_episode_count,
        "parse_repaired_episode_count": parse_repaired_episode_count,
        "task2_visible_floor_episode_count": task2_visible_floor_episode_count,
        "response_cap_hit_episode_count": response_cap_hit_episode_count,
        "parse_fallback_count": parse_fallback_episode_count,
        "parse_repaired_count": parse_repaired_episode_count,
        "shifted_episode_count": sum(len(scores) for scores in shifted_score_table.values()),
        "unshifted_episode_count": sum(len(scores) for scores in unshifted_score_table.values()),
        "native_action_episode_rate": strict_episode_count / episode_count if episode_count else 0.0,
        "assisted_episode_rate": assisted_episode_count / episode_count if episode_count else 0.0,
        "empty_fallback_episode_rate": empty_fallback_episode_count / episode_count if episode_count else 0.0,
        "response_cap_hit_episode_rate": response_cap_hit_episode_count / episode_count if episode_count else 0.0,
        "parse_fallback_step_count": parse_fallback_step_count,
        "parse_repaired_step_count": parse_repaired_step_count,
        "heuristic_rescue_step_count": heuristic_rescue_step_count,
        "empty_fallback_step_count": empty_fallback_step_count,
        "task2_visible_floor_step_count": task2_visible_floor_step_count,
        "response_cap_hit_step_count": response_cap_hit_step_count,
        "finish_reason_counts": finish_reason_counts,
        "benchmark_runtime_versions": runtime_versions,
    }


def _model_action(
    provider: ChatProvider,
    env: LatentGoalOpsEnvironment,
    observation,
    policy_mode: str = "model",
    operator_profile=None,
    allow_parse_repair: bool = True,
    allow_parse_fallback: bool = True,
    allow_task2_visible_floor: bool = False,
) -> tuple[LatentGoalOpsAction, dict, dict]:
    system = system_prompt(observation.task_id, policy_mode=policy_mode, operator_profile=operator_profile)
    user = user_prompt(
        observation.model_dump(mode="json"),
        policy_mode=policy_mode,
        operator_profile=operator_profile,
    )
    completion_budget = _completion_budget(observation.task_id, policy_mode=policy_mode)
    reasoning_effort = _reasoning_effort_for_step(observation.task_id, policy_mode=policy_mode)
    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]
    response = provider.complete(
        messages,
        max_tokens=completion_budget,
        reasoning_effort=reasoning_effort,
        json_schema=output_schema(observation.task_id),
    )
    parse_fallback = False
    parse_repaired = False
    heuristic_rescue = False
    empty_fallback = False
    guardrail_adjusted = False
    stabilized_with_fallback = False
    invalid_response_excerpt = None
    try:
        action = parse_action(response.content, observation.task_id)
    except Exception:
        parse_fallback = True
        invalid_response_excerpt = _response_excerpt(response.content)
        if allow_parse_repair:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You repair invalid benchmark action outputs. "
                        "Return exactly one valid JSON object for the active task and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task_prompt": user,
                            "previous_invalid_response": response.content,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                },
            ]
            try:
                repair_response = provider.complete(
                    repair_messages,
                    max_tokens=min(900, completion_budget),
                    reasoning_effort=reasoning_effort,
                    json_schema=output_schema(observation.task_id),
                )
                response = type(response)(
                    content=repair_response.content,
                    input_tokens=response.input_tokens + repair_response.input_tokens,
                    output_tokens=response.output_tokens + repair_response.output_tokens,
                    cost_usd=response.cost_usd + repair_response.cost_usd,
                    finish_reason=repair_response.finish_reason or response.finish_reason,
                    raw={"initial": response.raw, "repair": repair_response.raw},
                )
                action = parse_action(repair_response.content, observation.task_id)
                parse_repaired = True
                parse_fallback = False
                invalid_response_excerpt = None
            except Exception:
                if allow_parse_fallback:
                    heuristic_rescue = True
                    action = choose_action(env, "heuristic")
                else:
                    empty_fallback = True
                    action = _empty_action(observation.task_id)
        else:
            if allow_parse_fallback:
                heuristic_rescue = True
                action = choose_action(env, "heuristic")
            else:
                empty_fallback = True
                action = _empty_action(observation.task_id)
    if policy_mode == "synthetic_operator" and operator_profile is not None:
        action, guardrail_adjusted = apply_operator_guardrails(
            action,
            observation.model_dump(mode="json"),
            operator_profile,
        )
    elif policy_mode == "model" and action.task_id == TaskId.TASK2 and not parse_fallback and allow_task2_visible_floor:
        heuristic_floor = env.sample_heuristic_action()
        action, stabilized_with_fallback = stabilize_model_action(
            action,
            observation.model_dump(mode="json"),
            heuristic_floor,
        )
    native_action = not any([parse_fallback, parse_repaired, heuristic_rescue, empty_fallback, stabilized_with_fallback])
    assisted_action = bool(parse_repaired or heuristic_rescue or stabilized_with_fallback)
    response_cap_hit = bool(response.finish_reason == "length" or response.output_tokens == completion_budget)
    return action, {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "completion_budget": completion_budget,
        "reasoning_effort": reasoning_effort,
        "finish_reason": response.finish_reason,
        "response_cap_hit": response_cap_hit,
        "parse_fallback": parse_fallback,
        "parse_repaired": parse_repaired,
        "heuristic_rescue": heuristic_rescue,
        "empty_fallback": empty_fallback,
        "invalid_response_excerpt": invalid_response_excerpt,
        "task2_visible_floor_applied": stabilized_with_fallback,
        "native_action": native_action,
        "assisted_action": assisted_action,
    }, {
        "operator_profile": operator_profile.as_dict() if operator_profile is not None else None,
        "operator_guardrail_adjusted": guardrail_adjusted,
        "task2_visible_floor_applied": stabilized_with_fallback,
        "policy_mode": policy_mode,
    }


def run(args: argparse.Namespace) -> dict[str, float]:
    """Execute a baseline batch and return mean score by task."""
    load_dotenv(".env")
    tasks = _parse_tasks(args.tasks)
    seeds = _parse_seeds(args.seeds)
    provider_model = _resolve_provider_model(args)
    benchmark_model = str(getattr(args, "model", DEFAULT_AGENT_MODEL))
    persona_model = str(getattr(args, "persona_model", DEFAULT_PERSONA_MODEL))
    run_id = args.run_id or (
        f"{args.policy}_{provider_model}_{getattr(args, 'operator_style', 'na')}"
        if args.policy == "synthetic_operator"
        else f"{args.policy}_{benchmark_model}"
    )
    output_dir = Path(args.output_dir)
    logger = JsonlLogger(output_dir / "runs.jsonl")
    checkpoint_path = output_dir / "checkpoint.json"
    eval_flags = _effective_eval_flags(args)
    experiment_config = _experiment_config_from_args(args)
    checkpoint_signature = _checkpoint_signature(
        run_id=run_id,
        provider_model=provider_model,
        benchmark_model=benchmark_model,
        persona_model=persona_model,
        policy=args.policy,
        tasks=tasks,
        seeds=seeds,
        temperature=float(args.temperature),
        provider_name=str(getattr(args, "provider", "openai_compat")),
        provider_settings={
            "api_base_url": getattr(args, "api_base_url", None),
            "ollama_host": getattr(args, "ollama_host", None),
            "ollama_num_ctx": getattr(args, "ollama_num_ctx", None),
            "ollama_keep_alive": getattr(args, "ollama_keep_alive", None),
            "ollama_think": getattr(args, "ollama_think", None),
        },
        experiment_config=experiment_config,
        eval_flags=eval_flags,
    )
    checkpoint = _load_checkpoint(checkpoint_path, run_id, provider_model, args.policy, tasks, seeds, checkpoint_signature)
    run_started_perf = time.perf_counter()
    provider = None if args.policy not in {"model", "synthetic_operator"} else create_chat_provider(
        provider_model,
        provider=str(getattr(args, "provider", "openai_compat")),
        temperature=float(args.temperature),
        base_url=getattr(args, "api_base_url", None),
        ollama_host=getattr(args, "ollama_host", None),
        ollama_num_ctx=getattr(args, "ollama_num_ctx", None),
        ollama_keep_alive=getattr(args, "ollama_keep_alive", None),
        ollama_think=str(getattr(args, "ollama_think", "auto")),
    )
    budget_cap = _parse_budget_cap_arg(getattr(args, "budget_cap", None))
    score_table: dict[str, list[float]] = {task: [] for task in tasks}
    completion_status = "complete"
    interrupted_error: Exception | None = None
    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_id,
        group=args.wandb_group or f"{args.policy}-{','.join(tasks)}",
        tags=[tag for tag in (args.wandb_tags.split(",") if args.wandb_tags else []) if tag]
        + [args.policy, provider_model, getattr(args, "operator_style", "na")],
        config={
            "policy": args.policy,
            "model": benchmark_model,
            "provider_model": provider_model,
            "provider": str(getattr(args, "provider", "openai_compat")),
            "persona_model": persona_model if args.policy == "synthetic_operator" else None,
            "benchmark_runtime_version": BENCHMARK_RUNTIME_VERSION,
            "operator_style": getattr(args, "operator_style", "na"),
            "tasks": tasks,
            "seeds": seeds,
            "budget_cap": budget_cap,
            "output_dir": str(output_dir),
            "temperature": float(args.temperature),
            "api_base_url": getattr(args, "api_base_url", None),
            "ollama_host": getattr(args, "ollama_host", None),
            "ollama_num_ctx": getattr(args, "ollama_num_ctx", None),
            "ollama_keep_alive": getattr(args, "ollama_keep_alive", None),
            "ollama_think": getattr(args, "ollama_think", None),
            "experiment_config": experiment_config.model_dump(mode="json"),
            "paper_eval": eval_flags["paper_eval"],
        },
    )

    jobs = [(seed, task) for seed in seeds for task in tasks]
    jobs_total = len(jobs)
    step_global = 0
    display_model_name = _display_model_name(args.policy, benchmark_model, persona_model=persona_model)
    logged_episode_payloads = _logged_episode_payloads(logger.path)
    deduped_logged_episodes, _ = _dedupe_episode_payloads(logged_episode_payloads)
    completed_episode_keys = {_episode_identity(payload) for payload in deduped_logged_episodes}

    try:
        for index, (seed, task_id) in enumerate(jobs[checkpoint.next_index :], start=checkpoint.next_index):
            episode_key = _episode_identity_fields(run_id, task_id, seed, args.policy, display_model_name)
            if episode_key in completed_episode_keys:
                checkpoint.next_index = index + 1
                checkpoint.failure_mode = None
                checkpoint.metadata.pop("last_error", None)
                _save_checkpoint(checkpoint_path, checkpoint)
                print(
                    f"[resume-skip] {index + 1}/{jobs_total} task={task_id} seed={seed} already logged; advancing checkpoint",
                    flush=True,
                )
                continue
            episode_started_at = _utc_now_iso()
            episode_started_perf = time.perf_counter()
            env = LatentGoalOpsEnvironment(experiment_config=experiment_config)
            observation = env.reset(seed=seed, task_id=task_id)
            provider_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "response_cap_hit": False,
                "parse_fallback": False,
                "parse_repaired": False,
                "response_cap_hit_steps": 0,
                "parse_fallback_steps": 0,
                "parse_repaired_steps": 0,
                "finish_reason_counts": {},
                "invalid_response_excerpt": None,
            }
            step_index = 0
            operator_profile = None
            if args.policy == "synthetic_operator":
                operator_profile = resolve_operator_persona(getattr(args, "operator_style", "auto"), seed, TaskId(task_id))

            while not observation.done:
                step_started_at = _utc_now_iso()
                step_started_perf = time.perf_counter()
                if args.policy in {"model", "synthetic_operator"}:
                    try:
                        action, usage, model_metadata = _model_action(
                            provider,
                            env,
                            observation,
                            policy_mode=args.policy,
                            operator_profile=operator_profile,
                            allow_parse_repair=eval_flags["allow_parse_repair"],
                            allow_parse_fallback=eval_flags["allow_heuristic_rescue"],
                            allow_task2_visible_floor=eval_flags["allow_task2_visible_floor"],
                        )
                    except ProviderTransientError as exc:
                        completion_status = "interrupted_transient"
                        interrupted_error = exc
                        checkpoint.failure_mode = type(exc).__name__
                        checkpoint.metadata["last_error"] = str(exc)
                        checkpoint.next_index = index
                        _save_checkpoint(checkpoint_path, checkpoint)
                        print(
                            f"[interrupt] task={task_id} seed={seed} failure_mode={type(exc).__name__} message={exc}",
                            flush=True,
                        )
                        break
                    except (ProviderExhaustedError, ProviderCredentialError) as exc:
                        completion_status = f"interrupted_{type(exc).__name__}"
                        interrupted_error = exc
                        checkpoint.failure_mode = type(exc).__name__
                        checkpoint.metadata["last_error"] = str(exc)
                        checkpoint.next_index = index
                        _save_checkpoint(checkpoint_path, checkpoint)
                        raise
                else:
                    action = choose_action(env, args.policy)
                    usage = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "completion_budget": 0,
                        "reasoning_effort": None,
                        "finish_reason": None,
                        "response_cap_hit": False,
                        "parse_fallback": False,
                        "parse_repaired": False,
                        "heuristic_rescue": False,
                        "empty_fallback": False,
                        "invalid_response_excerpt": None,
                        "task2_visible_floor_applied": False,
                        "native_action": True,
                        "assisted_action": False,
                    }
                    model_metadata = {"operator_profile": None, "operator_guardrail_adjusted": False, "policy_mode": args.policy}

                provider_usage["input_tokens"] += usage["input_tokens"]
                provider_usage["output_tokens"] += usage["output_tokens"]
                provider_usage["cost_usd"] += usage["cost_usd"]
                provider_usage["response_cap_hit"] = bool(
                    provider_usage.get("response_cap_hit", False) or usage.get("response_cap_hit", False)
                )
                provider_usage["parse_fallback"] = bool(provider_usage["parse_fallback"] or usage.get("parse_fallback", False))
                provider_usage["parse_repaired"] = bool(provider_usage["parse_repaired"] or usage.get("parse_repaired", False))
                provider_usage["heuristic_rescue"] = bool(provider_usage.get("heuristic_rescue", False) or usage.get("heuristic_rescue", False))
                provider_usage["empty_fallback"] = bool(provider_usage.get("empty_fallback", False) or usage.get("empty_fallback", False))
                provider_usage["task2_visible_floor_applied"] = bool(
                    provider_usage.get("task2_visible_floor_applied", False) or usage.get("task2_visible_floor_applied", False)
                )
                provider_usage["native_action"] = bool(provider_usage.get("native_action", True) and usage.get("native_action", True))
                provider_usage["assisted_action"] = bool(
                    provider_usage.get("assisted_action", False) or usage.get("assisted_action", False)
                )
                provider_usage["parse_fallback_steps"] += int(bool(usage.get("parse_fallback", False)))
                provider_usage["parse_repaired_steps"] += int(bool(usage.get("parse_repaired", False)))
                provider_usage["response_cap_hit_steps"] += int(bool(usage.get("response_cap_hit", False)))
                provider_usage["heuristic_rescue_steps"] = int(provider_usage.get("heuristic_rescue_steps", 0)) + int(
                    bool(usage.get("heuristic_rescue", False))
                )
                provider_usage["empty_fallback_steps"] = int(provider_usage.get("empty_fallback_steps", 0)) + int(
                    bool(usage.get("empty_fallback", False))
                )
                provider_usage["task2_visible_floor_steps"] = int(provider_usage.get("task2_visible_floor_steps", 0)) + int(
                    bool(usage.get("task2_visible_floor_applied", False))
                )
                finish_reason = usage.get("finish_reason")
                if finish_reason:
                    finish_reason_key = str(finish_reason)
                    provider_usage["finish_reason_counts"][finish_reason_key] = int(
                        provider_usage["finish_reason_counts"].get(finish_reason_key, 0)
                    ) + 1
                if provider_usage.get("invalid_response_excerpt") is None and usage.get("invalid_response_excerpt"):
                    provider_usage["invalid_response_excerpt"] = usage["invalid_response_excerpt"]
                checkpoint.cumulative_cost_usd += usage["cost_usd"]
                checkpoint.cumulative_input_tokens += usage["input_tokens"]
                checkpoint.cumulative_output_tokens += usage["output_tokens"]

                if budget_cap is not None and checkpoint.cumulative_cost_usd > budget_cap:
                    completion_status = "interrupted_budget_cap_exceeded"
                    checkpoint.failure_mode = "budget_cap_exceeded"
                    checkpoint.metadata["last_error"] = "Configured budget cap exceeded."
                    checkpoint.next_index = index
                    _save_checkpoint(checkpoint_path, checkpoint)
                    raise ProviderExhaustedError("Configured budget cap exceeded.")

                next_observation = env.step(action)
                step_log = StepLog(
                    started_at=step_started_at,
                    finished_at=_utc_now_iso(),
                    elapsed_seconds=time.perf_counter() - step_started_perf,
                    run_id=run_id,
                    task_id=task_id,
                    seed=seed,
                    policy=args.policy,
                    model_name=display_model_name,
                    step_index=step_index,
                    action=action.model_dump(mode="json", exclude_none=True),
                    reward=float(next_observation.reward or 0.0),
                    done=bool(next_observation.done),
                    observation=next_observation.model_dump(mode="json"),
                    provider_usage=usage,
                    metadata={
                        "policy": args.policy,
                        "benchmark_runtime_version": BENCHMARK_RUNTIME_VERSION,
                        "experiment_config": experiment_config.model_dump(mode="json"),
                        "paper_eval": eval_flags["paper_eval"],
                        "strict_step": bool(usage.get("native_action", False)),
                        "native_action_step": bool(usage.get("native_action", False)),
                        "assisted_step": bool(usage.get("assisted_action", False)),
                        "rescued_step": bool(usage.get("assisted_action", False)),
                        "heuristic_rescue_step": bool(usage.get("heuristic_rescue", False)),
                        "empty_fallback_step": bool(usage.get("empty_fallback", False)),
                        "parse_repair_enabled": eval_flags["allow_parse_repair"],
                        "heuristic_rescue_enabled": eval_flags["allow_heuristic_rescue"],
                        "task2_visible_floor_enabled": eval_flags["allow_task2_visible_floor"],
                        "benchmark_model": benchmark_model if args.policy == "model" else None,
                        "persona_source_model": persona_model if args.policy == "synthetic_operator" else None,
                        **model_metadata,
                    },
                )
                logger.write_step(step_log)
                wandb_logger.log_step(step_log, episode_index=index, step_global=step_global)
                observation = next_observation
                step_index += 1
                step_global += 1

            if interrupted_error is not None:
                break

            grade = env.last_grade.model_dump(mode="json") if env.last_grade else {}
            score = float(grade.get("score", observation.reward or 0.0))
            company_profile = observation.company_profile.model_dump(mode="json") if getattr(observation, "company_profile", None) else None
            hidden_goal = getattr(env, "_hidden_goal", None)
            hidden_shift_present = bool(hidden_goal is not None and getattr(hidden_goal, "shift_step", None) is not None)
            score_table[task_id].append(score)
            elapsed_run_seconds = time.perf_counter() - run_started_perf
            episodes_completed = index + 1
            progress_fraction = episodes_completed / jobs_total if jobs_total else 1.0
            average_episode_seconds = elapsed_run_seconds / episodes_completed if episodes_completed else 0.0
            eta_seconds = average_episode_seconds * max(0, jobs_total - episodes_completed)
            episode_summary = EpisodeSummary(
                started_at=episode_started_at,
                finished_at=_utc_now_iso(),
                elapsed_seconds=time.perf_counter() - episode_started_perf,
                run_id=run_id,
                task_id=task_id,
                seed=seed,
                policy=args.policy,
                model_name=display_model_name,
                score=score,
                total_reward=float(env.state.cumulative_reward),
                total_steps=env.state.step_count,
                provider_usage=provider_usage,
                grade=grade,
                metadata={
                    "policy": args.policy,
                    "benchmark_runtime_version": BENCHMARK_RUNTIME_VERSION,
                    "experiment_config": experiment_config.model_dump(mode="json"),
                    "operator_profile": operator_profile.as_dict() if operator_profile is not None else None,
                    "benchmark_model": benchmark_model if args.policy == "model" else None,
                    "persona_source_model": persona_model if args.policy == "synthetic_operator" else None,
                    "paper_eval": eval_flags["paper_eval"],
                    "strict_episode": bool(provider_usage.get("native_action", False)),
                    "native_action_episode": bool(provider_usage.get("native_action", False)),
                    "assisted_episode": bool(provider_usage.get("assisted_action", False)),
                    "rescued_episode": bool(provider_usage.get("assisted_action", False)),
                    "heuristic_rescue_episode": bool(provider_usage.get("heuristic_rescue", False)),
                    "empty_fallback_episode": bool(provider_usage.get("empty_fallback", False)),
                    "parse_repair_enabled": eval_flags["allow_parse_repair"],
                    "heuristic_rescue_enabled": eval_flags["allow_heuristic_rescue"],
                    "task2_visible_floor_enabled": eval_flags["allow_task2_visible_floor"],
                    "progress_fraction": progress_fraction,
                    "episodes_completed": episodes_completed,
                    "episodes_total": jobs_total,
                    "eta_seconds": eta_seconds,
                    "run_elapsed_seconds": elapsed_run_seconds,
                    "company_profile": company_profile,
                    "company_family": company_profile.get("seed_family") if company_profile else None,
                    "company_archetype_id": company_profile.get("archetype_id") if company_profile else None,
                    "hidden_shift_present": hidden_shift_present,
                    "hidden_shift_step": getattr(hidden_goal, "shift_step", None),
                    "hidden_shift_type": getattr(hidden_goal, "shift_type", None),
                    "hidden_shift_changed_fields": list(getattr(hidden_goal, "changed_fields", ()) or ()),
                },
            )
            logger.write_episode(episode_summary)
            completed_episode_keys.add(episode_key)
            wandb_logger.log_episode(episode_summary, episode_index=index)
            print(
                "[progress] "
                f"{episodes_completed}/{jobs_total} "
                f"task={task_id} seed={seed} "
                f"score={score:.4f} "
                f"episode_time={_format_seconds(episode_summary.elapsed_seconds)} "
                f"run_time={_format_seconds(elapsed_run_seconds)} "
                f"eta={_format_seconds(eta_seconds)}",
                flush=True,
            )
            checkpoint.next_index = index + 1
            checkpoint.failure_mode = None
            checkpoint.metadata.pop("last_error", None)
            _save_checkpoint(checkpoint_path, checkpoint)
        if interrupted_error is not None:
            checkpoint.next_index = index
            _save_checkpoint(checkpoint_path, checkpoint)
    finally:
        episodes_df = _logged_episode_payloads(logger.path)
        summary_payload = _summarize_logged_episodes(
            tasks,
            episodes_df,
            eval_flags["paper_eval"],
            expected_episode_count=jobs_total,
            completion_status=completion_status,
        )
        means = dict(summary_payload["mean_scores"])
        (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        wandb_logger.log_summary(
            means,
            extra={
                "summary/run_id": run_id,
                "summary/ablation": experiment_config.slug(),
                "summary/run_started_at": checkpoint.started_at,
                "summary/run_elapsed_seconds": time.perf_counter() - run_started_perf,
                "summary/paper_eval": eval_flags["paper_eval"],
                "summary/strict_episode_count": summary_payload["strict_episode_count"],
                "summary/rescued_episode_count": summary_payload["rescued_episode_count"],
                "summary/parse_repaired_episode_count": summary_payload["parse_repaired_episode_count"],
                "summary/parse_fallback_episode_count": summary_payload["parse_fallback_episode_count"],
                "summary/cumulative_cost_usd": checkpoint.cumulative_cost_usd,
                "summary/cumulative_input_tokens": checkpoint.cumulative_input_tokens,
                "summary/cumulative_output_tokens": checkpoint.cumulative_output_tokens,
            },
        )
        wandb_logger.finish()
    if interrupted_error is not None:
        raise interrupted_error
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "synthetic_operator", "random", "heuristic", "oracle"], default="heuristic")
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--persona-model", default=DEFAULT_PERSONA_MODEL)
    parser.add_argument("--operator-style", choices=operator_style_choices(), default="auto")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:3")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--budget-cap", type=_parse_budget_cap_arg, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--provider", choices=["auto", "openai_compat", "ollama"], default="openai_compat")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--ollama-keep-alive", default="15m")
    parser.add_argument("--ollama-think", default="auto")
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--task3-horizon-override", type=int, default=None)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    parser.add_argument("--paper-eval", action="store_true")
    parser.add_argument("--disable-parse-repair", action="store_true")
    parser.add_argument("--disable-heuristic-rescue", action="store_true")
    parser.add_argument("--enable-task2-visible-floor", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="latentgoalops")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    args = parser.parse_args()
    means = run(args)
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
