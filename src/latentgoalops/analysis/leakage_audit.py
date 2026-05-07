"""Leakage audit for initial-observation latent-goal predictability."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.models import TaskId
from latentgoalops.server.environment import LatentGoalOpsEnvironment


TOKEN_RE = re.compile(r"[a-z][a-z_]+")
GOAL_TOKENS = {"growth", "retention", "revenue", "efficiency"}


def _task_ids(raw: str) -> list[str]:
    if raw == "all":
        return [task.value for task in TaskId]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _seed_list(raw: str) -> list[int]:
    if ":" in raw:
        start, count = [int(part) for part in raw.split(":", 1)]
        return list(range(start, start + count))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _observation_text(observation: dict) -> str:
    parts = [str(observation.get("task_summary", "")), str(observation.get("narrative", ""))]
    parts.extend(item.get("text", "") for item in observation.get("inbox", []))
    parts.extend(observation.get("stakeholder_notes", []))
    parts.extend(observation.get("alerts", []))
    for stakeholder in observation.get("stakeholders", []):
        parts.append(str(stakeholder.get("role", "")))
        parts.extend(stakeholder.get("favorite_metrics", []))
    for item in observation.get("backlog", []):
        parts.append(str(item.get("title", "")))
        parts.append(str(item.get("description", "")))
        parts.extend(item.get("policy_tags", []))
        parts.extend(item.get("risk_notes", []))
    return " ".join(parts).lower()


def _sender_only_text(observation: dict) -> str:
    parts: list[str] = []
    parts.extend(str(item.get("sender", "")) for item in observation.get("inbox", []))
    parts.extend(str(stakeholder.get("role", "")) for stakeholder in observation.get("stakeholders", []))
    return " ".join(parts).lower()


def _stakeholder_order_text(observation: dict) -> str:
    ordered_roles = [str(stakeholder.get("role", "")) for stakeholder in observation.get("stakeholders", [])]
    return " > ".join(ordered_roles).lower()


def _impact_summary_text(observation: dict) -> str:
    parts: list[str] = []
    for item in observation.get("backlog", []):
        parts.append(str(item.get("impact_summary", "")))
        parts.extend(item.get("risk_notes", []))
        parts.extend(item.get("beneficiary_segments", []))
    return " ".join(parts).lower()


def _metadata_only_text(observation: dict) -> str:
    parts: list[str] = []
    for stakeholder in observation.get("stakeholders", []):
        parts.append(str(stakeholder.get("role", "")))
        parts.extend(stakeholder.get("favorite_metrics", []))
        parts.extend(stakeholder.get("hard_constraints", []))
    for account in observation.get("accounts", []):
        parts.append(str(account.get("segment", "")))
        parts.append(str(account.get("support_tier", "")))
        parts.append(str(account.get("sla_level", "")))
    return " ".join(parts).lower()


def _numeric_features(observation: dict) -> dict[str, float]:
    dashboard = observation.get("dashboard", {}) or {}
    market_context = observation.get("market_context", {}) or {}
    accounts = observation.get("accounts", []) or []
    backlog = observation.get("backlog", []) or []
    stakeholders = observation.get("stakeholders", []) or []
    teams = observation.get("teams", []) or []

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    renewal_windows = [float(account.get("renewal_window_days", 0.0) or 0.0) for account in accounts]
    acvs = [float(account.get("annual_contract_value", 0.0) or 0.0) for account in accounts]
    relationship_health = [float(account.get("relationship_health", 0.0) or 0.0) for account in accounts]
    churn_propensity = [float(account.get("churn_propensity", 0.0) or 0.0) for account in accounts]
    item_costs = [float(item.get("cost", 0.0) or 0.0) for item in backlog]
    implementation_risk = [float(item.get("implementation_risk", 0.0) or 0.0) for item in backlog]

    return {
        "dashboard_dau": float(dashboard.get("dau", 0.0) or 0.0),
        "dashboard_d30_retention": float(dashboard.get("d30_retention", 0.0) or 0.0),
        "dashboard_mrr": float(dashboard.get("mrr", 0.0) or 0.0),
        "dashboard_ops_margin": float(dashboard.get("ops_margin", 0.0) or 0.0),
        "dashboard_support_tickets": float(dashboard.get("support_ticket_volume", 0.0) or 0.0),
        "market_board_pressure": float(market_context.get("board_pressure_level", 0.0) or 0.0),
        "market_runway_months": float(market_context.get("cash_runway_months", 0.0) or 0.0),
        "market_pipeline_health": float(market_context.get("sales_pipeline_health", 0.0) or 0.0),
        "account_count": float(len(accounts)),
        "stakeholder_count": float(len(stakeholders)),
        "team_count": float(len(teams)),
        "backlog_count": float(len(backlog)),
        "mean_renewal_window": _mean(renewal_windows),
        "mean_acv": _mean(acvs),
        "mean_relationship_health": _mean(relationship_health),
        "mean_churn_propensity": _mean(churn_propensity),
        "mean_item_cost": _mean(item_costs),
        "mean_item_risk": _mean(implementation_risk),
    }


def _token_counts(text: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(text))


def _explicit_goal_leak(text: str) -> int:
    lowered = text.lower()
    patterns = [
        r"aligned with (growth|retention|revenue|efficiency)",
        r"around (growth|retention|revenue|efficiency)",
        r"(growth|retention|revenue|efficiency) objective",
        r"goal is (growth|retention|revenue|efficiency)",
    ]
    return int(any(re.search(pattern, lowered) for pattern in patterns))


def _build_vocab(train_rows: list[dict], max_size: int, text_key: str | None) -> list[str]:
    if text_key is None:
        return []
    counts: Counter[str] = Counter()
    for row in train_rows:
        counts.update(_token_counts(row[text_key]))
    return [token for token, _ in counts.most_common(max_size)]


def _vectorize(rows: list[dict], vocab: list[str], numeric_keys: list[str], text_key: str | None) -> np.ndarray:
    matrix = []
    for row in rows:
        token_counts = _token_counts(row[text_key]) if text_key is not None else Counter()
        text_vector = [float(token_counts.get(token, 0.0)) for token in vocab]
        numeric_vector = [float(row["numeric"][key]) for key in numeric_keys]
        matrix.append(text_vector + numeric_vector)
    return np.array(matrix, dtype=float)


def _train_centroids(features: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(set(labels)):
        indices = [idx for idx, value in enumerate(labels) if value == label]
        centroids[label] = features[indices].mean(axis=0)
    return centroids


def _predict_centroid(centroids: dict[str, np.ndarray], vector: np.ndarray) -> str:
    best_label = ""
    best_score = -math.inf
    for label, centroid in centroids.items():
        score = float(np.dot(vector, centroid))
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _collect_examples(tasks: list[str], seeds: list[int], config: ExperimentConfig) -> list[dict]:
    rows: list[dict] = []
    for task_id in tasks:
        for seed in seeds:
            env = LatentGoalOpsEnvironment(experiment_config=config)
            observation = env.reset(seed=seed, task_id=task_id)
            hidden_goal = env._hidden_goal  # type: ignore[attr-defined]
            rows.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "label": hidden_goal.archetype.value if hidden_goal is not None else "unknown",
                    "full_text": _observation_text(observation.model_dump(mode="json")),
                    "sender_only_text": _sender_only_text(observation.model_dump(mode="json")),
                    "stakeholder_order_text": _stakeholder_order_text(observation.model_dump(mode="json")),
                    "impact_summary_text": _impact_summary_text(observation.model_dump(mode="json")),
                    "metadata_only_text": _metadata_only_text(observation.model_dump(mode="json")),
                    "numeric": _numeric_features(observation.model_dump(mode="json")),
                }
            )
    return rows


def _task_report(
    rows: list[dict],
    vocab_size: int,
    *,
    text_key: str | None,
    include_numeric: bool,
) -> dict:
    if not rows:
        return {}
    midpoint = max(1, len(rows) // 2)
    train_rows = rows[:midpoint]
    test_rows = rows[midpoint:]
    numeric_keys = sorted(train_rows[0]["numeric"].keys()) if include_numeric else []
    vocab = _build_vocab(train_rows, vocab_size, text_key)
    train_features = _vectorize(train_rows, vocab, numeric_keys, text_key)
    test_features = _vectorize(test_rows, vocab, numeric_keys, text_key)

    mean = train_features.mean(axis=0)
    std = np.where(train_features.std(axis=0) < 1e-6, 1.0, train_features.std(axis=0))
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    train_labels = [row["label"] for row in train_rows]
    test_labels = [row["label"] for row in test_rows]
    centroids = _train_centroids(train_features, train_labels)

    predictions = [_predict_centroid(centroids, vector) for vector in test_features]
    accuracy = (
        sum(int(pred == label) for pred, label in zip(predictions, test_labels)) / len(test_labels)
        if test_labels
        else 0.0
    )
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, prediction in zip(test_labels, predictions):
        confusion[label][prediction] += 1

    explicit_leak_rate = (
        sum(_explicit_goal_leak(row[text_key]) for row in rows) / len(rows)
        if text_key is not None
        else 0.0
    )
    return {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "centroid_accuracy": round(float(accuracy), 4),
        "explicit_goal_token_rate": round(float(explicit_leak_rate), 4),
        "confusion": {label: dict(values) for label, values in confusion.items()},
        "vocab_size": len(vocab),
        "numeric_feature_count": len(numeric_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="100:40")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--output", default="outputs/leakage_audit.json")
    parser.add_argument("--disable-hidden-shift", action="store_true")
    parser.add_argument("--disable-delayed-effects", action="store_true")
    parser.add_argument("--hide-decision-ledger", action="store_true")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"], default="shaped")
    args = parser.parse_args()

    config = ExperimentConfig(
        enable_hidden_shift=not args.disable_hidden_shift,
        enable_delayed_effects=not args.disable_delayed_effects,
        expose_decision_ledger=not args.hide_decision_ledger,
        reward_mode=args.reward_mode,
    )
    tasks = _task_ids(args.tasks)
    seeds = _seed_list(args.seeds)
    rows = _collect_examples(tasks, seeds, config)

    report = {
        "tasks": tasks,
        "seeds": seeds,
        "results": {
            task_id: (
                lambda task_rows: {
                    **_task_report(task_rows, args.vocab_size, text_key="full_text", include_numeric=True),
                    "views": {
                        "sender_only": _task_report(
                            task_rows,
                            args.vocab_size,
                            text_key="sender_only_text",
                            include_numeric=False,
                        ),
                        "stakeholder_order_only": _task_report(
                            task_rows,
                            args.vocab_size,
                            text_key="stakeholder_order_text",
                            include_numeric=False,
                        ),
                        "impact_summary_only": _task_report(
                            task_rows,
                            args.vocab_size,
                            text_key="impact_summary_text",
                            include_numeric=False,
                        ),
                        "metadata_only": _task_report(
                            task_rows,
                            args.vocab_size,
                            text_key="metadata_only_text",
                            include_numeric=False,
                        ),
                        "numeric_only": _task_report(
                            task_rows,
                            args.vocab_size,
                            text_key=None,
                            include_numeric=True,
                        ),
                    },
                }
            )([row for row in rows if row["task_id"] == task_id])
            for task_id in tasks
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
