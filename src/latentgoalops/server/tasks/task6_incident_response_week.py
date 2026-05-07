"""Task 6: multi-day incident response week."""

from __future__ import annotations

import random

from latentgoalops.models import DashboardState, TaskId
from latentgoalops.server.hidden_goals import HiddenGoal
from latentgoalops.server.tasks.task3_startup_week import (
    apply_task3_action,
    build_task3_episode,
    build_task3_view,
    evaluate_task3_action_value,
    solve_task3_oracle_action,
)


def _incident_dashboard() -> DashboardState:
    return DashboardState(
        dau=14900.0,
        mau=70200.0,
        d7_retention=0.37,
        d30_retention=0.21,
        mrr=35600.0,
        arpu=88.0,
        cac=73.0,
        churn_rate=0.058,
        ops_margin=0.27,
        infra_cost_per_unit=2.36,
        support_ticket_volume=228,
    )


def build_task6_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    horizon: int,
    budget: float,
    capacity: float,
    world: dict,
    split: str = "core",
) -> dict:
    """Create a crisis-heavy incident-response week."""
    episode = build_task3_episode(rng, hidden_goal, horizon, budget, capacity, world, split=split)
    episode["task_id"] = TaskId.TASK6
    episode["dashboard"] = _incident_dashboard()
    episode["budget_remaining"] = float(budget)
    episode["capacity_remaining"] = float(capacity)
    episode["task_summary"] = (
        "Contain and recover from an active incident week. Rebuild trust with at-risk accounts, "
        "sequence follow-up actions across days, manage noisy stakeholder pressure, and adapt if the latent objective shifts."
    )
    episode["alerts"] = [
        "Primary API latency remains unstable after yesterday's incident.",
        "High-touch accounts are asking for daily recovery updates.",
    ]
    episode["market_context"].board_pressure_level = min(1.0, episode["market_context"].board_pressure_level + 0.12)
    episode["market_context"].sales_pipeline_health = max(0.2, episode["market_context"].sales_pipeline_health - 0.08)
    episode["market_context"].cash_runway_months = max(6, episode["market_context"].cash_runway_months - 1)
    episode["backlog"] = [
        item
        for item in episode["backlog"]
        if item.kind in {"retention", "efficiency", "revenue"}
    ][:7]
    episode["events"].setdefault(0, []).append(
        {
            "event_id": "incident_root_cause",
            "name": "incident_root_cause",
            "summary": "A still-unconfirmed root cause is keeping leadership from calling the incident closed.",
            "clue_goal": "retention",
            "alerts": ["CTO says the root cause is not fully contained yet."],
            "effects": {"support_ticket_volume": 16.0, "d30_retention": -0.01},
            "lag_steps": 1,
            "sim_date": episode["start_date"],
        }
    )
    episode["events"].setdefault(2, []).append(
        {
            "event_id": "press_escalation",
            "name": "press_escalation",
            "summary": "A public escalation thread is increasing pressure to demonstrate both reliability and accountability.",
            "clue_goal": "retention",
            "alerts": ["Communications lead flagged a visible customer escalation thread."],
            "effects": {"support_ticket_volume": 10.0, "dau": -90.0},
            "lag_steps": 1,
            "sim_date": episode["start_date"],
        }
    )
    episode["events"].setdefault(4, []).append(
        {
            "event_id": "recovery_board_checkin",
            "name": "recovery_board_checkin",
            "summary": "The board wants evidence that the recovery plan is improving both trust and execution resilience.",
            "clue_goal": "efficiency",
            "alerts": ["Board requested a recovery update ahead of the next check-in."],
            "effects": {},
            "lag_steps": 1,
            "sim_date": episode["start_date"],
        }
    )
    return episode


def build_task6_view(rng: random.Random, hidden_goal: HiddenGoal, episode: dict) -> dict:
    """Create the current observable bundle."""
    view = build_task3_view(rng, hidden_goal, episode)
    view["task_summary"] = episode["task_summary"]
    view["alerts"] = list(dict.fromkeys([*episode.get("alerts", []), *view["alerts"]]))
    if view.get("narrative"):
        view["narrative"] = (
            "Incident response week. "
            + view["narrative"]
            + " False leads and delayed consequences are common, so update your plan cautiously."
        )
    return view


__all__ = [
    "apply_task3_action",
    "build_task6_episode",
    "build_task6_view",
    "evaluate_task3_action_value",
    "solve_task3_oracle_action",
]
