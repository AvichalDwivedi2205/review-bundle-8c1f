"""Task graders for LatentGoalOps."""

from __future__ import annotations

from typing import Any

from latentgoalops.models import FeedbackLabel, GraderResult, TaskId
from latentgoalops.server.hidden_goals import compute_utility, feedback_category_weight
from latentgoalops.server.rewards import CHANNELS, cosine_similarity, strategy_embedding


def _weighted_score(components: list[tuple[str, float | None, float]]) -> tuple[float, dict[str, float], bool]:
    """Average weighted components, renormalizing when some metrics are not applicable."""
    active = [(name, float(value), float(weight)) for name, value, weight in components if value is not None]
    total_weight = sum(weight for _, _, weight in active) or 1.0
    score = sum(value * weight for _, value, weight in active) / total_weight
    return score, {name: round(value, 4) for name, value, _ in active}, any(name == "adaptation" for name, _, _ in active)


def grade_task1(
    true_labels: dict[str, FeedbackLabel],
    true_priorities: dict[str, int],
    oracle_escalations: list[str],
    action_payload: dict[str, Any],
    hidden_goal,
) -> GraderResult:
    """Grade the weighted feedback triage task."""
    predicted_labels = {item["item_id"]: FeedbackLabel(item["label"]) for item in action_payload.get("labels", [])}
    predicted_priorities = {item["item_id"]: int(item["priority"]) for item in action_payload.get("priorities", [])}
    escalated = set(action_payload.get("escalate_ids", []))

    total_weight = sum(feedback_category_weight(hidden_goal, label) for label in true_labels.values()) or 1.0
    label_score = sum(
        feedback_category_weight(hidden_goal, label)
        for item_id, label in true_labels.items()
        if predicted_labels.get(item_id) == label
    ) / total_weight

    priority_score = 0.0
    if true_priorities:
        priority_score = sum(
            max(0.0, 1.0 - abs(predicted_priorities.get(item_id, 1) - priority) / 4.0)
            for item_id, priority in true_priorities.items()
        ) / len(true_priorities)

    escalation_score = len(escalated.intersection(oracle_escalations)) / max(len(oracle_escalations), 1)
    score = 0.50 * label_score + 0.30 * priority_score + 0.20 * escalation_score
    return GraderResult(
        task_id=TaskId.TASK1,
        score=round(float(max(0.0, min(1.0, score))), 4),
        sub_scores={
            "label": round(float(label_score), 4),
            "priority": round(float(priority_score), 4),
            "escalation": round(float(escalation_score), 4),
        },
    )


def grade_task2(
    agent_value: float,
    random_baseline: float,
    oracle_value: float,
    unused_budget_ratio: float,
) -> GraderResult:
    """Grade the roadmap prioritization task via normalized utility gain."""
    denom = max(oracle_value - random_baseline, 1e-6)
    normalized = max(0.0, min(1.0, (agent_value - random_baseline) / denom))
    waste_penalty = min(0.25, unused_budget_ratio * 0.2)
    score = max(0.0, normalized - waste_penalty)
    return GraderResult(
        task_id=TaskId.TASK2,
        score=round(float(score), 4),
        sub_scores={
            "normalized_value": round(float(normalized), 4),
            "budget_use": round(float(1.0 - unused_budget_ratio), 4),
        },
    )


def grade_task3(
    latent_utility: float,
    adaptation_score: float | None,
    coherence_score: float,
    constraint_score: float,
    belief_score: float | None = None,
) -> GraderResult:
    """Grade the startup-week task."""
    score, sub_scores, adaptation_scored = _weighted_score(
        [
            ("final_utility", latent_utility, 0.50),
            ("adaptation", adaptation_score, 0.25),
            ("coherence", coherence_score, 0.15),
            ("constraints", constraint_score, 0.10),
        ]
    )
    if belief_score is not None:
        sub_scores["belief_tracking"] = round(float(belief_score), 4)
    return GraderResult(
        task_id=TaskId.TASK3,
        score=round(float(max(0.0, min(1.0, score))), 4),
        sub_scores=sub_scores,
        details={"adaptation_scored": adaptation_scored},
    )


def grade_task6(
    latent_utility: float,
    adaptation_score: float | None,
    coherence_score: float,
    constraint_score: float,
    belief_score: float | None = None,
) -> GraderResult:
    """Grade the incident-response week task."""
    score, sub_scores, adaptation_scored = _weighted_score(
        [
            ("final_utility", latent_utility, 0.4634),
            ("adaptation", adaptation_score, 0.2927),
            ("coherence", coherence_score, 0.1220),
            ("constraints", constraint_score, 0.1220),
        ]
    )
    if belief_score is not None:
        sub_scores["belief_tracking"] = round(float(belief_score), 4)
    return GraderResult(
        task_id=TaskId.TASK6,
        score=round(float(max(0.0, min(1.0, score))), 4),
        sub_scores=sub_scores,
        details={"adaptation_scored": adaptation_scored},
    )


def grade_task7(
    latent_utility: float,
    adaptation_score: float | None,
    coherence_score: float,
    constraint_score: float,
    belief_score: float | None = None,
) -> GraderResult:
    """Grade the quarterly headcount planning task."""
    score, sub_scores, adaptation_scored = _weighted_score(
        [
            ("final_utility", latent_utility, 0.6364),
            ("adaptation", adaptation_score, 0.2614),
            ("coherence", coherence_score, 0.0568),
            ("constraints", constraint_score, 0.0454),
        ]
    )
    if belief_score is not None:
        sub_scores["belief_tracking"] = round(float(belief_score), 4)
    return GraderResult(
        task_id=TaskId.TASK7,
        score=round(float(max(0.0, min(1.0, score))), 4),
        sub_scores=sub_scores,
        details={"adaptation_scored": adaptation_scored},
    )


def grade_task4(
    agent_value: float,
    random_baseline: float,
    oracle_value: float,
    budget_use_ratio: float,
) -> GraderResult:
    """Grade the capital-allocation task."""
    denom = max(oracle_value - random_baseline, 1e-6)
    normalized = max(0.0, min(1.0, (agent_value - random_baseline) / denom))
    budget_use = max(0.0, min(1.0, budget_use_ratio))
    score = 0.85 * normalized + 0.15 * budget_use
    return GraderResult(
        task_id=TaskId.TASK4,
        score=round(float(max(0.0, min(1.0, score))), 4),
        sub_scores={
            "normalized_value": round(float(normalized), 4),
            "budget_use": round(float(budget_use), 4),
        },
    )


def grade_task5(
    agent_value: float,
    random_baseline: float,
    oracle_value: float,
    constraint_score: float,
) -> GraderResult:
    """Grade the crisis-response package task."""
    denom = max(oracle_value - random_baseline, 1e-6)
    normalized = max(0.0, min(1.0, (agent_value - random_baseline) / denom))
    bounded_constraints = max(0.0, min(1.0, constraint_score))
    score = 0.80 * normalized + 0.20 * bounded_constraints
    return GraderResult(
        task_id=TaskId.TASK5,
        score=round(float(max(0.0, min(1.0, score))), 4),
        sub_scores={
            "normalized_value": round(float(normalized), 4),
            "constraints": round(float(bounded_constraints), 4),
        },
    )


def _strategy_vector(value: Any) -> dict[str, float]:
    if isinstance(value, dict) and all(channel in value for channel in CHANNELS):
        return {channel: float(value[channel]) for channel in CHANNELS}
    return strategy_embedding(value)


def trajectory_coherence(actions: list[Any], split_index: int | None = None) -> float:
    """Compute mean within-regime strategic coherence, ignoring one necessary pivot."""
    if len(actions) < 2:
        return 1.0
    similarities = []
    if split_index is None or split_index <= 0 or split_index >= len(actions):
        pairs = zip(actions, actions[1:])
        for previous, current in pairs:
            similarities.append(max(0.0, cosine_similarity(_strategy_vector(previous), _strategy_vector(current))))
    else:
        for previous, current in zip(actions[:split_index], actions[1:split_index]):
            similarities.append(max(0.0, cosine_similarity(_strategy_vector(previous), _strategy_vector(current))))
        for previous, current in zip(actions[split_index:], actions[split_index + 1 :]):
            similarities.append(max(0.0, cosine_similarity(_strategy_vector(previous), _strategy_vector(current))))
    if not similarities:
        return 1.0
    return sum(similarities) / len(similarities)


def latent_score_from_dashboard(dashboard, hidden_goal) -> float:
    """Expose a shared utility helper for evaluation code."""
    return compute_utility(dashboard.to_metric_vector(), hidden_goal)
