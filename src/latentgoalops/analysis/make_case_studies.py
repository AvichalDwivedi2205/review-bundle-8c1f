"""Export representative trajectories for qualitative paper analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from latentgoalops.analysis.aggregate import load_run_records
from latentgoalops.analysis.make_paper_tables import generate_tables


CASE_DEFINITIONS = [
    {
        "case_id": "deepseek14_success",
        "title": "DeepSeek-R1 14B high-scoring native-policy trajectory",
        "model": "deepseek-r1:14b",
        "split": "core",
        "prefer": "highest_score",
        "requires_empty_fallback": False,
    },
    {
        "case_id": "qwen27_interface_failure",
        "title": "Qwen 27B low-scoring trajectory with action-shell collapse",
        "model": "qwen3.5:27b",
        "split": "core",
        "prefer": "lowest_score",
        "requires_empty_fallback": True,
    },
    {
        "case_id": "gemma26_interface_failure",
        "title": "Gemma 26B mixed policy/interface failure trajectory",
        "model": "gemma4:26b",
        "split": "heldout",
        "prefer": "lowest_score",
        "requires_empty_fallback": True,
    },
    {
        "case_id": "qwen9_native_policy_regression",
        "title": "Qwen 9B lower-scoring native-policy trajectory without shell collapse",
        "model": "qwen3.5:9b",
        "split": "core",
        "prefer": "lowest_score",
        "requires_empty_fallback": False,
        "requires_strict_episode": True,
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_protocol(config_path: str | Path) -> dict[str, Any]:
    from latentgoalops.paper.run_paper_suite import generate_paper_suite

    suite = generate_paper_suite(config_path)
    path = Path(suite["metadata_dir"]) / "paper_protocol_resolved.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(config_path: str | Path) -> dict[str, Any]:
    from latentgoalops.paper.run_paper_suite import generate_paper_suite

    suite = generate_paper_suite(config_path)
    path = Path(suite["metadata_dir"]) / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir_for_model(manifest: dict[str, Any], display_name: str) -> Path:
    for model in manifest["models"]:
        if model["display_name"] == display_name or model["name"] == display_name:
            return _repo_root() / model["run_dir"]
    raise KeyError(f"Unknown frozen model: {display_name}")


def _main_tasks(protocol: dict[str, Any]) -> set[str]:
    return set(protocol.get("task_sets", {}).get("main", []) or [])


def _select_episode(episodes: pd.DataFrame, case: dict[str, Any], main_tasks: set[str]) -> pd.Series:
    candidates = episodes[episodes["task_id"].isin(main_tasks)].copy()
    if case.get("requires_strict_episode"):
        strict_candidates = candidates[candidates["strict_episode"] == 1]
        if not strict_candidates.empty:
            candidates = strict_candidates
    if case.get("requires_empty_fallback"):
        fallback_candidates = candidates[candidates["empty_fallback_episode"] == 1]
        if not fallback_candidates.empty:
            candidates = fallback_candidates
    if candidates.empty:
        raise ValueError(f"No candidate episodes for case {case['case_id']}")
    if case["prefer"] == "highest_score":
        return candidates.sort_values("score", ascending=False).iloc[0]
    if case["prefer"] == "lowest_score":
        return candidates.sort_values("score", ascending=True).iloc[0]
    raise ValueError(f"Unsupported case preference: {case['prefer']}")


def _action_summary(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"raw_action": action}
    keys = [
        "selected_item_ids",
        "chosen_initiatives",
        "priorities",
        "escalate_ids",
        "messaging_action",
        "support_policy",
        "pricing_change_pct",
        "budget_allocations",
        "memory_focus",
        "memory_writes",
    ]
    return {key: action.get(key) for key in keys if key in action and action.get(key) not in (None, [], {})}


def _observation_summary(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    inbox = observation.get("inbox") or []
    backlog = observation.get("backlog") or []
    pending = observation.get("pending_effects") or []
    realized = observation.get("realized_effects") or []
    narrative = str(observation.get("narrative") or "")
    return {
        "sim_date": observation.get("sim_date"),
        "sim_day_label": observation.get("sim_day_label"),
        "step_index": observation.get("step_index"),
        "narrative_excerpt": narrative[:360],
        "inbox_count": len(inbox) if isinstance(inbox, list) else None,
        "backlog_count": len(backlog) if isinstance(backlog, list) else None,
        "pending_effect_count": len(pending) if isinstance(pending, list) else None,
        "realized_effect_count": len(realized) if isinstance(realized, list) else None,
    }


def _episode_key(row: pd.Series) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "task_id": row.get("task_id"),
        "seed": int(row.get("seed")),
        "model_name": row.get("model_name"),
        "policy": row.get("policy"),
    }


def _matching_steps(steps: pd.DataFrame, episode: pd.Series) -> pd.DataFrame:
    mask = (
        (steps["run_id"] == episode["run_id"])
        & (steps["task_id"] == episode["task_id"])
        & (steps["seed"] == episode["seed"])
        & (steps["model_name"] == episode["model_name"])
    )
    return steps[mask].sort_values("step_index")


def _case_payload(case: dict[str, Any], run_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    steps, episodes = load_run_records(run_dir / case["split"])
    episode = _select_episode(episodes, case, _main_tasks(protocol))
    episode_steps = _matching_steps(steps, episode)
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "model": case["model"],
        "split": case["split"],
        "selection_rule": case["prefer"],
        "episode": {
            **_episode_key(episode),
            "score": float(episode["score"]),
            "strict_episode": bool(episode.get("strict_episode", False)),
            "empty_fallback_episode": bool(episode.get("empty_fallback_episode", False)),
            "response_cap_hit": bool(episode.get("response_cap_hit", False)),
            "hidden_shift_present": bool(episode.get("hidden_shift_present", False)),
            "hidden_shift_step": episode.get("hidden_shift_step"),
            "sub_scores": episode.get("grade", {}).get("sub_scores", {}) if isinstance(episode.get("grade"), dict) else {},
            "invalid_response_excerpt": episode.get("invalid_response_excerpt"),
        },
        "timeline": [
            {
                "step_index": int(row["step_index"]),
                "reward": float(row["reward"]),
                "done": bool(row.get("done", False)),
                "strict_step": bool(row.get("strict_step", False)),
                "empty_fallback": bool(row.get("empty_fallback", False)),
                "response_cap_hit": bool(row.get("response_cap_hit", False)),
                "output_tokens": int(row.get("output_tokens", 0) or 0),
                "finish_reason": row.get("finish_reason"),
                "observation": _observation_summary(row.get("observation")),
                "action": _action_summary(row.get("action")),
            }
            for _, row in episode_steps.iterrows()
        ],
    }


def _case_markdown(case_payload: dict[str, Any]) -> str:
    episode = case_payload["episode"]
    lines = [
        f"# {case_payload['title']}",
        "",
        f"- Model: `{case_payload['model']}`",
        f"- Split: `{case_payload['split']}`",
        f"- Task: `{episode['task_id']}`",
        f"- Seed: `{episode['seed']}`",
        f"- Score: `{episode['score']:.4f}`",
        f"- Strict episode: `{episode['strict_episode']}`",
        f"- Empty fallback episode: `{episode['empty_fallback_episode']}`",
        f"- Response cap hit: `{episode['response_cap_hit']}`",
        f"- Hidden shift present: `{episode['hidden_shift_present']}`",
        "",
        "## Timeline",
        "",
    ]
    for step in case_payload["timeline"]:
        action = json.dumps(step["action"], ensure_ascii=True, sort_keys=True)
        obs = step["observation"]
        lines.extend(
            [
                f"### Step {step['step_index']}",
                "",
                f"- Reward: `{step['reward']:.4f}`",
                f"- Strict step: `{step['strict_step']}`",
                f"- Empty fallback: `{step['empty_fallback']}`",
                f"- Cap hit: `{step['response_cap_hit']}`",
                f"- Observation: `{obs.get('sim_day_label')}` with `{obs.get('pending_effect_count')}` pending effects",
                f"- Action summary: `{action}`",
                "",
            ]
        )
    return "\n".join(lines)


def generate_case_studies(config_path: str | Path) -> dict[str, Any]:
    generate_tables(config_path)
    protocol = _load_protocol(config_path)
    manifest = _load_manifest(config_path)
    repo_root = _repo_root()
    output_dir = repo_root / protocol["paper"]["case_studies_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, str]] = {}
    for case in CASE_DEFINITIONS:
        payload = _case_payload(case, _run_dir_for_model(manifest, case["model"]), protocol)
        json_path = output_dir / f"{case['case_id']}.json"
        md_path = output_dir / f"{case['case_id']}.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(_case_markdown(payload), encoding="utf-8")
        generated[case["case_id"]] = {
            "json": str(json_path.relative_to(repo_root)),
            "markdown": str(md_path.relative_to(repo_root)),
        }
    summary = {
        "case_studies_dir": str(output_dir.relative_to(repo_root)),
        "case_count": len(generated),
        "cases": generated,
    }
    (output_dir / "case_studies_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_protocol.yaml")
    args = parser.parse_args()
    print(json.dumps(generate_case_studies(args.config), indent=2))


if __name__ == "__main__":
    main()
