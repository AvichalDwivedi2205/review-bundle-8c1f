"""Repeated-rollout reliability evaluation for paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from latentgoalops.analysis.aggregate import load_episode_summaries
from latentgoalops.analysis.stats import bootstrap_mean_ci
from latentgoalops.baseline.run_baseline import (
    BENCHMARK_RUNTIME_VERSION,
    DEFAULT_AGENT_MODEL,
    DEFAULT_PERSONA_MODEL,
    _parse_budget_cap_arg,
    run,
)


DEFAULT_BENCHMARK_MODELS = [
    DEFAULT_AGENT_MODEL,
]
DEFAULT_OLLAMA_BENCHMARK_MODELS = [
    "qwen3:8b",
    "qwen3:14b",
    "gemma3:12b",
    "gpt-oss:20b",
    "qwen3:30b",
]


def _parse_models(raw: str, policy: str, persona_model: str, provider: str) -> list[str]:
    if raw == "default":
        if policy == "synthetic_operator":
            return [persona_model]
        if provider == "ollama":
            return DEFAULT_OLLAMA_BENCHMARK_MODELS
        return DEFAULT_BENCHMARK_MODELS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _k_values(raw: str, repeats: int) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    return [value for value in values if 1 <= value <= repeats]


def _aggregate_reliability(df, threshold: float, ks: list[int]) -> dict:
    results: dict[str, dict] = {}
    if df.empty:
        return results
    for (model_name, task_id), task_df in df.groupby(["model_name", "task_id"]):
        episode_groups = task_df.groupby(["seed", "repeat_index"])
        seed_scores: dict[int, list[float]] = {}
        for (seed, repeat_index), episode_df in episode_groups:
            seed_scores.setdefault(int(seed), [])
            seed_scores[int(seed)].append(float(episode_df["score"].iloc[0]))
        for seed in seed_scores:
            seed_scores[seed] = seed_scores[seed][: max(ks)]
        task_result = {
            "mean_score": sum(sum(scores) / len(scores) for scores in seed_scores.values()) / max(len(seed_scores), 1),
            "mean_score_ci": [0.0, 0.0],
            "pass_at_k": {},
            "score_at_k": {},
        }
        per_seed_mean_scores = [sum(scores) / len(scores) for scores in seed_scores.values() if scores]
        task_result["mean_score_ci"] = list(bootstrap_mean_ci(per_seed_mean_scores))
        for k in ks:
            pass_values = []
            score_values = []
            for scores in seed_scores.values():
                prefix = scores[:k]
                if not prefix:
                    continue
                pass_values.append(float(any(score >= threshold for score in prefix)))
                score_values.append(max(prefix))
            task_result["pass_at_k"][str(k)] = sum(pass_values) / max(len(pass_values), 1)
            task_result["score_at_k"][str(k)] = sum(score_values) / max(len(score_values), 1)
            task_result.setdefault("pass_at_k_ci", {})[str(k)] = list(bootstrap_mean_ci(pass_values))
            task_result.setdefault("score_at_k_ci", {})[str(k)] = list(bootstrap_mean_ci(score_values))
        results.setdefault(model_name, {})[task_id] = task_result
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "synthetic_operator"], default="model")
    parser.add_argument("--models", default="default")
    parser.add_argument("--persona-model", default=DEFAULT_PERSONA_MODEL)
    parser.add_argument("--operator-style", default="auto")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:10")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/reliability")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--k-values", default="1,3,5")
    parser.add_argument("--budget-cap", type=_parse_budget_cap_arg, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--provider", choices=["auto", "openai_compat", "ollama"], default="openai_compat")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--ollama-keep-alive", default="15m")
    parser.add_argument("--ollama-think", default="auto")
    parser.add_argument("--paper-eval", action="store_true")
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--task3-horizon-override", type=int, default=None)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="latentgoalops")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="reliability")
    parser.add_argument("--wandb-tags", default="reliability")
    args = parser.parse_args()

    load_dotenv(".env")
    models = _parse_models(args.models, args.policy, args.persona_model, args.provider)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_root = output_root / model.replace("/", "-")
        for repeat_index in range(args.repeats):
            repeat_dir = model_root / f"repeat_{repeat_index:02d}"
            run(
                SimpleNamespace(
                    policy=args.policy,
                    model=model,
                    persona_model=model if args.policy == "synthetic_operator" else args.persona_model,
                    operator_style=args.operator_style,
                    tasks=args.tasks,
                    seeds=args.seeds,
                    output_dir=str(repeat_dir),
                    run_id=f"{model.replace('/', '-')}-rep{repeat_index:02d}",
                    budget_cap=args.budget_cap,
                    temperature=args.temperature,
                    provider=args.provider,
                    api_base_url=args.api_base_url,
                    ollama_host=args.ollama_host,
                    ollama_num_ctx=args.ollama_num_ctx,
                    ollama_keep_alive=args.ollama_keep_alive,
                    ollama_think=args.ollama_think,
                    paper_eval=args.paper_eval,
                    disable_hidden_shift=args.disable_hidden_shift,
                    disable_delayed_effects=args.disable_delayed_effects,
                    hide_decision_ledger=args.hide_decision_ledger,
                    reward_mode=args.reward_mode,
                    task3_horizon_override=args.task3_horizon_override,
                    scenario_split=args.scenario_split,
                    wandb=args.wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_group=args.wandb_group,
                    wandb_tags=f"{args.wandb_tags},repeat_{repeat_index:02d}",
                )
            )

    episodes = load_episode_summaries(output_root)
    if not episodes.empty:
        episodes["repeat_index"] = episodes["run_id"].str.extract(r"rep(\d+)").astype(int)
    ks = _k_values(args.k_values, args.repeats)
    report = {
        "models": models,
        "persona_models": models if args.policy == "synthetic_operator" else None,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "provider": args.provider,
        "api_base_url": args.api_base_url,
        "ollama_host": args.ollama_host,
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_keep_alive": args.ollama_keep_alive,
        "ollama_think": args.ollama_think,
        "scenario_split": args.scenario_split,
        "paper_eval": bool(args.paper_eval),
        "benchmark_runtime_version": BENCHMARK_RUNTIME_VERSION,
        "k_values": ks,
        "results": _aggregate_reliability(episodes, args.threshold, ks),
    }
    (output_root / "reliability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
