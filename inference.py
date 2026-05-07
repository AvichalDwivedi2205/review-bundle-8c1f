"""Submission baseline runner for the OpenEnv hackathon."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from latentgoalops.baseline.providers import OPENROUTER_BASE_URL
from latentgoalops.baseline.run_baseline import run


DEFAULT_TASKS = ",".join(
    [
        "task1_feedback_triage",
        "task2_roadmap_priority",
        "task3_startup_week",
    ]
)
DEFAULT_API_BASE_URL = "https://inference.do-ai.run/v1"


def _resolve_submission_env(cli_model: str | None, cli_api_base_url: str | None, cli_token: str | None) -> tuple[str, str]:
    """Resolve the OpenAI-compatible endpoint configuration required by the submission."""
    load_dotenv(".env")

    api_base_url = (
        cli_api_base_url
        or os.getenv("API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or (OPENROUTER_BASE_URL if os.getenv("OPENROUTER_API_KEY") else DEFAULT_API_BASE_URL)
    )
    model_name = cli_model or os.getenv("MODEL_NAME") or os.getenv("BASELINE_MODEL")
    token = (
        cli_token
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("DIGITALOCEAN_API_TOKEN")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("MODEL_ACCESS_KEY")
    )

    if not model_name:
        raise ValueError(
            "MODEL_NAME is required. Set MODEL_NAME in the environment or pass --model."
        )
    if not token:
        raise ValueError(
            "HF_TOKEN is required for the submission baseline. "
            "For local runs, you can also set OPENROUTER_API_KEY, DIGITALOCEAN_API_TOKEN, or pass --token."
        )

    # Normalize onto the submission contract so downstream helpers only see one interface.
    os.environ["API_BASE_URL"] = api_base_url
    os.environ.setdefault("OPENAI_BASE_URL", api_base_url)
    os.environ["MODEL_NAME"] = model_name
    os.environ["HF_TOKEN"] = token

    return api_base_url, model_name


def _build_args(parsed: argparse.Namespace, model_name: str) -> SimpleNamespace:
    """Translate submission CLI args into the existing benchmark runner contract."""
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        policy="model",
        model=model_name,
        persona_model=model_name,
        operator_style="auto",
        tasks=parsed.tasks,
        seeds=parsed.seeds,
        output_dir=str(output_dir),
        run_id=parsed.run_id,
        budget_cap=parsed.budget_cap,
        temperature=parsed.temperature,
        disable_hidden_shift=parsed.disable_hidden_shift,
        disable_delayed_effects=parsed.disable_delayed_effects,
        hide_decision_ledger=parsed.hide_decision_ledger,
        reward_mode=parsed.reward_mode,
        task3_horizon_override=parsed.task3_horizon_override,
        scenario_split=parsed.scenario_split,
        wandb=False,
        wandb_project="latentgoalops",
        wandb_entity=None,
        wandb_group=None,
        wandb_tags="submission",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LatentGoalOps submission baseline with the required OpenAI-compatible client contract."
        )
    )
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--seeds", default="100:1")
    parser.add_argument("--output-dir", default="outputs/submission-baseline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--budget-cap", type=float, default=20.0)
    parser.add_argument("--scenario-split", choices=["core", "heldout"], default="core")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    parser.add_argument("--task3-horizon-override", type=int, default=None)
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--model", default=None, help="Optional override for MODEL_NAME.")
    parser.add_argument("--api-base-url", default=None, help="Optional override for API_BASE_URL.")
    parser.add_argument("--token", default=None, help="Optional override for HF_TOKEN.")
    args = parser.parse_args()

    api_base_url, model_name = _resolve_submission_env(
        cli_model=args.model,
        cli_api_base_url=args.api_base_url,
        cli_token=args.token,
    )
    benchmark_args = _build_args(args, model_name=model_name)
    mean_scores = run(benchmark_args)

    summary = {
        "api_base_url": api_base_url,
        "model_name": model_name,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "seeds": args.seeds,
        "mean_scores": mean_scores,
        "summary_path": str(Path(args.output_dir) / "summary.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
