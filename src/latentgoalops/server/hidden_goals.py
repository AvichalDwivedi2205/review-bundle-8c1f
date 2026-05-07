"""Latent goal sampling and utility helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from latentgoalops.models import (
    FeedbackLabel,
    GoalArchetype,
    GovernanceStrictness,
    PlanningHorizon,
    RiskPosture,
)
from latentgoalops.server.config import load_config


CHANNELS = ("growth", "retention", "revenue", "efficiency")

FEEDBACK_TO_CHANNEL = {
    FeedbackLabel.BUG: "efficiency",
    FeedbackLabel.FEATURE_REQUEST: "growth",
    FeedbackLabel.CHURN_RISK: "retention",
    FeedbackLabel.PRAISE: "growth",
    FeedbackLabel.BILLING_ISSUE: "revenue",
    FeedbackLabel.LATENCY_COMPLAINT: "efficiency",
}


@dataclass(slots=True)
class LatentObjectiveState:
    """Structured latent objective active at a particular point in time."""

    archetype: GoalArchetype
    weights: dict[str, float]
    primary_kpi: str
    risk_posture: RiskPosture
    planning_horizon: PlanningHorizon
    segment_focus: str
    governance_strictness: GovernanceStrictness


@dataclass(slots=True)
class HiddenGoal:
    """Sampled hidden objective for an episode."""

    archetype: GoalArchetype
    weights: dict[str, float]
    alpha: float
    primary_kpi: str
    risk_posture: RiskPosture
    planning_horizon: PlanningHorizon
    segment_focus: str
    governance_strictness: GovernanceStrictness
    shift_goal: GoalArchetype | None = None
    shift_step: int | None = None
    shift_weights: dict[str, float] | None = None
    shift_primary_kpi: str | None = None
    shift_risk_posture: RiskPosture | None = None
    shift_planning_horizon: PlanningHorizon | None = None
    shift_segment_focus: str | None = None
    shift_governance_strictness: GovernanceStrictness | None = None
    shift_type: str = "abrupt"
    shift_duration_steps: int = 1
    changed_fields: tuple[str, ...] = ()
    shift_trigger: str | None = None


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _enum_choice(enum_type, value: str):
    return enum_type(value)


def _sample_field(options: list[str], rng: random.Random, fallback: str) -> str:
    if not options:
        return fallback
    return str(rng.choice(options))


def _sample_structured_goal(
    archetype: GoalArchetype,
    alpha: float,
    rng: random.Random,
) -> LatentObjectiveState:
    config = load_config("hidden_goals.yaml")
    raw = config["archetypes"][archetype.value]
    weights = sample_weights_for_archetype(archetype, alpha, rng)
    risk_posture = _enum_choice(
        RiskPosture,
        _sample_field(list(raw.get("risk_posture_choices", [])), rng, str(raw.get("risk_posture", "balanced"))),
    )
    planning_horizon = _enum_choice(
        PlanningHorizon,
        _sample_field(
            list(raw.get("planning_horizon_choices", [])),
            rng,
            str(raw.get("planning_horizon", "quarterly")),
        ),
    )
    governance_strictness = _enum_choice(
        GovernanceStrictness,
        _sample_field(
            list(raw.get("governance_strictness_choices", [])),
            rng,
            str(raw.get("governance_strictness", "moderate")),
        ),
    )
    segment_focus = _sample_field(list(raw.get("segment_focus_choices", [])), rng, str(raw.get("segment_focus", "mid_market")))
    return LatentObjectiveState(
        archetype=archetype,
        weights=weights,
        primary_kpi=str(raw.get("primary_kpi", "mrr")),
        risk_posture=risk_posture,
        planning_horizon=planning_horizon,
        segment_focus=segment_focus,
        governance_strictness=governance_strictness,
    )


def sample_hidden_goal(
    rng: random.Random,
    allow_shift: bool = False,
    *,
    max_shift_step: int | None = None,
) -> HiddenGoal:
    """Sample a goal archetype and richer latent objective."""
    config = load_config("hidden_goals.yaml")
    archetype = GoalArchetype(rng.choice(list(config["archetypes"].keys())))
    alpha = float(rng.choice(config["alpha_choices"]))
    base = _sample_structured_goal(archetype, alpha, rng)

    shift_goal: GoalArchetype | None = None
    shift_step: int | None = None
    shift_state: LatentObjectiveState | None = None
    shift_type = "abrupt"
    shift_duration_steps = 1
    changed_fields: tuple[str, ...] = ()
    shift_trigger: str | None = None
    valid_shift_steps = [
        int(step)
        for step in config.get("shift_steps", [3, 4, 5])
        if max_shift_step is None or int(step) <= max_shift_step
    ]
    if max_shift_step is not None and not valid_shift_steps and max_shift_step >= 2:
        valid_shift_steps = list(range(2, max_shift_step + 1))
    if allow_shift and valid_shift_steps and rng.random() < float(config.get("shift_probability", 0.0)):
        choices = [GoalArchetype(value) for value in config["archetypes"] if value != archetype.value]
        shift_goal = rng.choice(choices)
        shift_step = int(rng.choice(valid_shift_steps))
        shift_type = str(rng.choice(config.get("shift_types", ["abrupt"])))
        shift_duration_steps = int(rng.choice(config.get("shift_duration_choices", [1, 2])))
        shift_trigger = str(rng.choice(config.get("shift_triggers", ["board_reprioritization"])))
        target = _sample_structured_goal(shift_goal, alpha, rng)
        candidate_fields = [
            "weights",
            "primary_kpi",
            "risk_posture",
            "planning_horizon",
            "segment_focus",
            "governance_strictness",
        ]
        count = min(len(candidate_fields), max(2, int(rng.choice(config.get("changed_field_count_choices", [2, 3, 4])))))
        changed_fields = tuple(sorted(rng.sample(candidate_fields, k=count)))
        shift_state = LatentObjectiveState(
            archetype=target.archetype if "weights" in changed_fields else base.archetype,
            weights=target.weights if "weights" in changed_fields else dict(base.weights),
            primary_kpi=target.primary_kpi if "primary_kpi" in changed_fields else base.primary_kpi,
            risk_posture=target.risk_posture if "risk_posture" in changed_fields else base.risk_posture,
            planning_horizon=target.planning_horizon if "planning_horizon" in changed_fields else base.planning_horizon,
            segment_focus=target.segment_focus if "segment_focus" in changed_fields else base.segment_focus,
            governance_strictness=(
                target.governance_strictness
                if "governance_strictness" in changed_fields
                else base.governance_strictness
            ),
        )

    return HiddenGoal(
        archetype=base.archetype,
        weights=base.weights,
        alpha=alpha,
        primary_kpi=base.primary_kpi,
        risk_posture=base.risk_posture,
        planning_horizon=base.planning_horizon,
        segment_focus=base.segment_focus,
        governance_strictness=base.governance_strictness,
        shift_goal=shift_state.archetype if shift_state is not None else shift_goal,
        shift_step=shift_step,
        shift_weights=shift_state.weights if shift_state is not None else None,
        shift_primary_kpi=shift_state.primary_kpi if shift_state is not None else None,
        shift_risk_posture=shift_state.risk_posture if shift_state is not None else None,
        shift_planning_horizon=shift_state.planning_horizon if shift_state is not None else None,
        shift_segment_focus=shift_state.segment_focus if shift_state is not None else None,
        shift_governance_strictness=shift_state.governance_strictness if shift_state is not None else None,
        shift_type=shift_type,
        shift_duration_steps=max(1, shift_duration_steps),
        changed_fields=changed_fields,
        shift_trigger=shift_trigger,
    )


def sample_weights_for_archetype(archetype: GoalArchetype, alpha: float, rng: random.Random) -> dict[str, float]:
    """Sample a perturbed weight vector around a goal archetype."""
    config = load_config("hidden_goals.yaml")
    base = config["archetypes"][archetype.value]["weights"]
    dirichlet = np.random.default_rng(rng.randint(0, 10_000_000)).dirichlet([alpha] * 4)
    blended = {
        channel: 0.75 * float(base[channel]) + 0.25 * float(dirichlet[index])
        for index, channel in enumerate(CHANNELS)
    }
    return _normalize(blended)


def _blend_weights(left: dict[str, float], right: dict[str, float], progress: float) -> dict[str, float]:
    return _normalize(
        {
            channel: float(left.get(channel, 0.0)) * (1.0 - progress) + float(right.get(channel, 0.0)) * progress
            for channel in CHANNELS
        }
    )


def _shift_progress(hidden_goal: HiddenGoal, step_index: int) -> float:
    if hidden_goal.shift_step is None or hidden_goal.shift_weights is None or step_index < hidden_goal.shift_step:
        return 0.0
    if hidden_goal.shift_type == "abrupt":
        return 1.0
    duration = max(hidden_goal.shift_duration_steps, 1)
    offset = step_index - hidden_goal.shift_step + 1
    return max(0.0, min(1.0, offset / duration))


def active_state(hidden_goal: HiddenGoal, step_index: int) -> LatentObjectiveState:
    """Return the active structured latent objective at a step."""
    progress = _shift_progress(hidden_goal, step_index)
    if progress <= 0.0 or hidden_goal.shift_weights is None:
        return LatentObjectiveState(
            archetype=hidden_goal.archetype,
            weights=dict(hidden_goal.weights),
            primary_kpi=hidden_goal.primary_kpi,
            risk_posture=hidden_goal.risk_posture,
            planning_horizon=hidden_goal.planning_horizon,
            segment_focus=hidden_goal.segment_focus,
            governance_strictness=hidden_goal.governance_strictness,
        )
    if progress >= 1.0:
        return LatentObjectiveState(
            archetype=hidden_goal.shift_goal or hidden_goal.archetype,
            weights=dict(hidden_goal.shift_weights),
            primary_kpi=hidden_goal.shift_primary_kpi or hidden_goal.primary_kpi,
            risk_posture=hidden_goal.shift_risk_posture or hidden_goal.risk_posture,
            planning_horizon=hidden_goal.shift_planning_horizon or hidden_goal.planning_horizon,
            segment_focus=hidden_goal.shift_segment_focus or hidden_goal.segment_focus,
            governance_strictness=hidden_goal.shift_governance_strictness or hidden_goal.governance_strictness,
        )
    shifted = LatentObjectiveState(
        archetype=hidden_goal.shift_goal or hidden_goal.archetype,
        weights=_blend_weights(hidden_goal.weights, hidden_goal.shift_weights, progress),
        primary_kpi=hidden_goal.shift_primary_kpi or hidden_goal.primary_kpi,
        risk_posture=hidden_goal.shift_risk_posture or hidden_goal.risk_posture,
        planning_horizon=hidden_goal.shift_planning_horizon or hidden_goal.planning_horizon,
        segment_focus=hidden_goal.shift_segment_focus or hidden_goal.segment_focus,
        governance_strictness=hidden_goal.shift_governance_strictness or hidden_goal.governance_strictness,
    )
    if hidden_goal.shift_type == "drift":
        shifted.primary_kpi = hidden_goal.primary_kpi if progress < 0.5 else shifted.primary_kpi
        shifted.risk_posture = hidden_goal.risk_posture if progress < 0.5 else shifted.risk_posture
        shifted.planning_horizon = hidden_goal.planning_horizon if progress < 0.5 else shifted.planning_horizon
        shifted.segment_focus = hidden_goal.segment_focus if progress < 0.5 else shifted.segment_focus
        shifted.governance_strictness = (
            hidden_goal.governance_strictness if progress < 0.5 else shifted.governance_strictness
        )
    return shifted


def snapshot_hidden_goal(hidden_goal: HiddenGoal, step_index: int) -> HiddenGoal:
    """Freeze the active latent state at one step into a shift-free HiddenGoal."""
    state = active_state(hidden_goal, step_index)
    return HiddenGoal(
        archetype=state.archetype,
        weights=dict(state.weights),
        alpha=hidden_goal.alpha,
        primary_kpi=state.primary_kpi,
        risk_posture=state.risk_posture,
        planning_horizon=state.planning_horizon,
        segment_focus=state.segment_focus,
        governance_strictness=state.governance_strictness,
    )


def channel_weight(hidden_goal: HiddenGoal, channel: str, step_index: int = 0) -> float:
    """Return channel weight for a hidden goal."""
    return float(active_state(hidden_goal, step_index).weights.get(channel, 0.0))


def compute_utility(metric_vector: dict[str, float], hidden_goal: HiddenGoal, step_index: int = 0) -> float:
    """Compute latent utility for a normalized four-channel metric vector."""
    state = active_state(hidden_goal, step_index)
    return max(
        0.0,
        min(
            1.0,
            sum(float(metric_vector.get(channel, 0.0)) * state.weights[channel] for channel in CHANNELS),
        ),
    )


def active_weights(hidden_goal: HiddenGoal, step_index: int) -> dict[str, float]:
    """Return the active weight vector at a given step index."""
    return active_state(hidden_goal, step_index).weights


def active_goal_name(hidden_goal: HiddenGoal, step_index: int) -> str:
    """Return the active goal archetype name at a step."""
    return active_state(hidden_goal, step_index).archetype.value


def active_segment_focus(hidden_goal: HiddenGoal, step_index: int) -> str:
    """Return the active segment focus."""
    return active_state(hidden_goal, step_index).segment_focus


def active_governance_strictness(hidden_goal: HiddenGoal, step_index: int) -> GovernanceStrictness:
    """Return the active governance strictness."""
    return active_state(hidden_goal, step_index).governance_strictness


def active_risk_posture(hidden_goal: HiddenGoal, step_index: int) -> RiskPosture:
    """Return the active risk posture."""
    return active_state(hidden_goal, step_index).risk_posture


def active_planning_horizon(hidden_goal: HiddenGoal, step_index: int) -> PlanningHorizon:
    """Return the active planning horizon."""
    return active_state(hidden_goal, step_index).planning_horizon


def belief_target(hidden_goal: HiddenGoal, step_index: int) -> dict[str, str]:
    """Return the structured active latent state for evaluation."""
    state = active_state(hidden_goal, step_index)
    return {
        "archetype": state.archetype.value,
        "risk_posture": state.risk_posture.value,
        "planning_horizon": state.planning_horizon.value,
        "segment_focus": state.segment_focus,
        "governance_strictness": state.governance_strictness.value,
    }


def feedback_category_weight(hidden_goal: HiddenGoal, label: FeedbackLabel, step_index: int = 0) -> float:
    """Map a feedback category to its hidden utility weight."""
    return channel_weight(hidden_goal, FEEDBACK_TO_CHANNEL[label], step_index=step_index)


def initiative_alignment_score(hidden_goal: HiddenGoal, initiative_kind: str, step_index: int = 0) -> float:
    """How aligned an initiative kind is with the latent goal."""
    return channel_weight(hidden_goal, initiative_kind if initiative_kind in CHANNELS else "growth", step_index=step_index)
