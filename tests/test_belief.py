"""Belief-scoring and long-horizon grading tests."""

from __future__ import annotations

import math

from latentgoalops.analysis.belief import score_belief_report
from latentgoalops.server.grader import grade_task3, grade_task7


def test_missing_belief_report_scores_as_missing_not_uniform():
    target = {
        "archetype": "growth",
        "risk_posture": "balanced",
        "planning_horizon": "quarterly",
        "segment_focus": "mid_market",
        "governance_strictness": "moderate",
    }

    row = score_belief_report(None, target)

    assert row["belief_report_missing"] == 1.0
    assert row["archetype_brier"] == 1.0
    assert row["archetype_confidence"] == 0.0
    assert row["archetype_correct"] == 0.0
    assert math.isclose(row["archetype_nll"], math.log(4))


def test_grade_task3_renormalizes_when_adaptation_is_not_applicable():
    grade = grade_task3(
        latent_utility=0.8,
        adaptation_score=None,
        coherence_score=0.5,
        constraint_score=0.9,
        belief_score=0.4,
    )

    expected = (0.50 * 0.8 + 0.15 * 0.5 + 0.10 * 0.9) / 0.75
    assert math.isclose(grade.score, round(expected, 4))
    assert "adaptation" not in grade.sub_scores
    assert grade.sub_scores["belief_tracking"] == 0.4
    assert grade.details["adaptation_scored"] is False


def test_grade_task7_renormalizes_when_adaptation_is_not_applicable():
    grade = grade_task7(
        latent_utility=0.7,
        adaptation_score=None,
        coherence_score=0.6,
        constraint_score=0.8,
        belief_score=0.5,
    )

    expected = (0.6364 * 0.7 + 0.0568 * 0.6 + 0.0454 * 0.8) / (0.6364 + 0.0568 + 0.0454)
    assert math.isclose(grade.score, round(expected, 4))
    assert "adaptation" not in grade.sub_scores
    assert grade.sub_scores["belief_tracking"] == 0.5
    assert grade.details["adaptation_scored"] is False
