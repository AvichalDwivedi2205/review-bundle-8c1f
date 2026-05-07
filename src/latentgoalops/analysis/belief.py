"""Belief-tracking evaluation helpers."""

from __future__ import annotations

import math

from latentgoalops.models import BeliefReport


def _normalize_probs(probs: dict[str, float], universe: list[str]) -> dict[str, float]:
    if not universe:
        return {}
    filtered = {key: max(0.0, float(probs.get(key, 0.0))) for key in universe}
    total = sum(filtered.values())
    if total <= 0.0:
        uniform = 1.0 / len(universe)
        return {key: uniform for key in universe}
    return {key: value / total for key, value in filtered.items()}


def brier_score(probs: dict[str, float], target: str, universe: list[str]) -> float:
    """Multi-class Brier score."""
    normalized = _normalize_probs(probs, universe)
    return sum((normalized[key] - (1.0 if key == target else 0.0)) ** 2 for key in universe) / max(len(universe), 1)


def negative_log_likelihood(probs: dict[str, float], target: str, universe: list[str]) -> float:
    """Bounded NLL for a categorical belief."""
    normalized = _normalize_probs(probs, universe)
    return -math.log(max(1e-8, normalized.get(target, 0.0)))


def expected_calibration_error(rows: list[tuple[float, bool]], buckets: int = 10) -> float:
    """Simple ECE over confidence/correctness pairs."""
    if not rows:
        return 0.0
    bucket_values: list[list[tuple[float, bool]]] = [[] for _ in range(buckets)]
    for confidence, correct in rows:
        index = min(buckets - 1, max(0, int(confidence * buckets)))
        bucket_values[index].append((confidence, correct))
    total = len(rows)
    error = 0.0
    for bucket in bucket_values:
        if not bucket:
            continue
        mean_conf = sum(conf for conf, _ in bucket) / len(bucket)
        mean_acc = sum(1.0 if correct else 0.0 for _, correct in bucket) / len(bucket)
        error += abs(mean_conf - mean_acc) * len(bucket) / total
    return error


def score_belief_report(report: BeliefReport | None, target: dict[str, str]) -> dict[str, float]:
    """Score one factorized belief report against the active latent state."""
    universes = {
        "archetype": ["growth", "retention", "revenue", "efficiency"],
        "risk_posture": ["aggressive", "balanced", "conservative"],
        "planning_horizon": ["immediate", "quarterly", "strategic"],
        "segment_focus": ["self_serve", "smb", "mid_market", "enterprise", "strategic"],
        "governance_strictness": ["flexible", "moderate", "strict"],
    }
    if report is None:
        scores: dict[str, float] = {}
        for field, universe in universes.items():
            scores[f"{field}_brier"] = 1.0
            scores[f"{field}_nll"] = math.log(len(universe))
            scores[f"{field}_confidence"] = 0.0
            scores[f"{field}_correct"] = 0.0
        scores["belief_report_missing"] = 1.0
        return scores

    fields = {
        "archetype": report.archetype_probs,
        "risk_posture": report.risk_posture_probs,
        "planning_horizon": report.planning_horizon_probs,
        "segment_focus": report.segment_focus_probs,
        "governance_strictness": report.governance_strictness_probs,
    }
    scores: dict[str, float] = {}
    for field, probs in fields.items():
        target_value = target[field]
        universe = universes[field]
        normalized = _normalize_probs(probs, universe)
        top_label = max(normalized, key=normalized.get)
        confidence = float(normalized[top_label])
        scores[f"{field}_brier"] = brier_score(probs, target_value, universe)
        scores[f"{field}_nll"] = negative_log_likelihood(probs, target_value, universe)
        scores[f"{field}_confidence"] = confidence
        scores[f"{field}_correct"] = 1.0 if top_label == target_value else 0.0
    scores["belief_report_missing"] = 0.0
    return scores
