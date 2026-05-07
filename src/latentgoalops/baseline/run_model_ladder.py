"""Launch a sequential model sweep with consistent logging and grouping."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from latentgoalops.baseline.providers import ProviderTransientError
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_models(raw: str, policy: str, persona_model: str, provider: str) -> list[str]:
    if raw == "default":
        if policy == "synthetic_operator":
            return [persona_model]
        if provider == "ollama":
            return DEFAULT_OLLAMA_BENCHMARK_MODELS
        return DEFAULT_BENCHMARK_MODELS
    return [model.strip() for model in raw.split(",") if model.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "synthetic_operator"], default="model")
    parser.add_argument("--models", default="default")
    parser.add_argument("--persona-model", default=DEFAULT_PERSONA_MODEL)
    parser.add_argument("--operator-style", default="auto")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:10")
    parser.add_argument("--output-root", default="outputs/model-ladder")
    parser.add_argument("--budget-cap", type=_parse_budget_cap_arg, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    parser.add_argument("--wandb-group", default="model-ladder")
    parser.add_argument("--wandb-tags", default="model-ladder,validation")
    args = parser.parse_args()

    load_dotenv(".env")
    models = _parse_models(args.models, args.policy, args.persona_model, args.provider)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()
    started_perf = time.perf_counter()
    aggregate: dict[str, dict[str, float] | str | list[str]] = {
        "started_at": started_at,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "scenario_split": args.scenario_split,
        "paper_eval": bool(args.paper_eval),
        "provider": args.provider,
        "api_base_url": args.api_base_url,
        "ollama_host": args.ollama_host,
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_keep_alive": args.ollama_keep_alive,
        "ollama_think": args.ollama_think,
        "benchmark_runtime_version": BENCHMARK_RUNTIME_VERSION,
        "models": models,
        "persona_models": models if args.policy == "synthetic_operator" else None,
    }
    overall_mean_scores: dict[str, float] = {}
    strict_native_overall_mean_scores: dict[str, float] = {}
    aggregate["completion_status"] = "complete"

    for model in models:
        safe_name = model.replace("/", "-")
        output_dir = output_root / safe_name
        run_id = f"{safe_name}-{args.seeds.replace(':', '_')}"
        print(f"[model-ladder] starting model={model} output_dir={output_dir}", flush=True)
        model_started = time.perf_counter()
        try:
            summary = run(
                SimpleNamespace(
                    policy=args.policy,
                    model=model,
                    persona_model=model if args.policy == "synthetic_operator" else args.persona_model,
                    operator_style=args.operator_style,
                    tasks=args.tasks,
                    seeds=args.seeds,
                    output_dir=str(output_dir),
                    run_id=f"{args.policy}-{run_id}" if args.policy != "model" else run_id,
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
                    wandb_tags=f"{args.wandb_tags},{safe_name}",
                )
            )
        except ProviderTransientError as exc:
            summary_path = output_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            aggregate["completion_status"] = "interrupted_transient"
            aggregate[model] = {
                "elapsed_seconds": time.perf_counter() - model_started,
                "scores": summary,
                "error": str(exc),
            }
            if summary.get("overall_mean_score") is not None:
                overall_mean_scores[model] = float(summary["overall_mean_score"])
            if summary.get("strict_native_overall_mean_score") is not None:
                strict_native_overall_mean_scores[model] = float(summary["strict_native_overall_mean_score"])
            aggregate["overall_mean_scores"] = overall_mean_scores
            aggregate["strict_native_overall_mean_scores"] = strict_native_overall_mean_scores
            aggregate_path = output_root / "aggregate_summary.json"
            aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
            print(f"[model-ladder] interrupted model={model} error={exc}", flush=True)
            raise SystemExit(2) from exc
        aggregate[model] = {
            "elapsed_seconds": time.perf_counter() - model_started,
            "scores": summary,
        }
        if summary.get("overall_mean_score") is not None:
            overall_mean_scores[model] = float(summary["overall_mean_score"])
        if summary.get("strict_native_overall_mean_score") is not None:
            strict_native_overall_mean_scores[model] = float(summary["strict_native_overall_mean_score"])
        aggregate["overall_mean_scores"] = overall_mean_scores
        aggregate["strict_native_overall_mean_scores"] = strict_native_overall_mean_scores
        aggregate_path = output_root / "aggregate_summary.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(
            f"[model-ladder] finished model={model} "
            f"elapsed_seconds={aggregate[model]['elapsed_seconds']:.2f}",
            flush=True,
        )

    aggregate["finished_at"] = _utc_now_iso()
    aggregate["elapsed_seconds"] = time.perf_counter() - started_perf
    aggregate["overall_mean_scores"] = overall_mean_scores
    aggregate["strict_native_overall_mean_scores"] = strict_native_overall_mean_scores
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
