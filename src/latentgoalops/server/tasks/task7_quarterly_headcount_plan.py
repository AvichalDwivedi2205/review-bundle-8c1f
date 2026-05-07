"""Task 7: quarterly headcount planning with delayed staffing effects."""

from __future__ import annotations

import random

from latentgoalops.models import (
    CustomerAccount,
    DashboardState,
    DecisionLedgerEntry,
    InitiativeItem,
    InternalTeamState,
    LatentGoalOpsAction,
    SimCalendarEvent,
    TaskId,
    TemporalEffectRecord,
)
from latentgoalops.server.config import load_config
from latentgoalops.server.hidden_goals import HiddenGoal, active_goal_name, active_state, compute_utility
from latentgoalops.server.memory import (
    apply_agent_writes,
    auto_visible_entities,
    build_memory_workspace,
    initialize_memory_bank,
    record_decision,
    record_effect,
    record_team_update,
    reset_step_budgets,
)
from latentgoalops.server.objective_utils import structured_item_multiplier
from latentgoalops.server.tasks.task3_startup_week import (
    _apply_channel_deltas,
    _effect_record,
    sim_date_for_step,
)
from latentgoalops.server.tasks.task4_capital_allocation import allocation_value, solve_oracle_allocations


def _task7_config() -> dict:
    return load_config("tasks.yaml")["task7"]


def _initial_dashboard() -> DashboardState:
    return DashboardState(
        dau=19800.0,
        mau=85400.0,
        d7_retention=0.42,
        d30_retention=0.25,
        mrr=46800.0,
        arpu=97.0,
        cac=68.0,
        churn_rate=0.046,
        ops_margin=0.33,
        infra_cost_per_unit=2.04,
        support_ticket_volume=148,
    )


def _quarter_label(step_index: int) -> str:
    return f"Quarter {step_index + 1}"


def _curve_units(amount: float, saturation_point: float) -> float:
    if amount <= 0.0:
        return 0.0
    if amount <= saturation_point:
        return amount
    return saturation_point + 0.50 * (amount - saturation_point)


def _headcount_state_signals(episode: dict) -> dict[str, float]:
    dashboard = episode["dashboard"]
    market_context = episode["market_context"]
    accounts = list(episode["accounts"])
    teams_by_id = {team.team_id: team for team in episode["teams"]}
    commercial_accounts = [
        account for account in accounts if account.segment in {"mid_market", "enterprise", "strategic"}
    ] or accounts
    near_renewal_accounts = [account for account in commercial_accounts if account.renewal_window_days <= 30]
    renewal_pressure = len(near_renewal_accounts) / max(len(commercial_accounts), 1)
    trust_gap = sum(max(0.0, 0.68 - account.relationship_health) for account in commercial_accounts) / max(
        len(commercial_accounts),
        1,
    )
    expansion_pressure = sum(account.expansion_potential for account in commercial_accounts) / max(
        len(commercial_accounts),
        1,
    )
    support_stress = max(0.0, min(1.0, (dashboard.support_ticket_volume - 140.0) / 120.0))
    margin_gap = max(0.0, min(1.0, (market_context.gross_margin_target - dashboard.ops_margin) / 0.25))
    runway_tightness = max(0.0, min(1.0, (10.0 - float(market_context.cash_runway_months)) / 7.0))
    pipeline_slack = max(0.0, min(1.0, (0.60 - market_context.sales_pipeline_health) / 0.40))

    tracked_teams = [team for team in teams_by_id.values() if team.function in {"growth", "support", "product", "infra"}]
    reliability_gap = (
        sum(
            max(0.0, 0.72 - team.execution_reliability) + max(0.0, team.burnout_risk - 0.45)
            for team in tracked_teams
        )
        / max(len(tracked_teams), 1)
    )
    growth_team = teams_by_id.get("team_growth")
    support_team = teams_by_id.get("team_support")
    product_team = teams_by_id.get("team_product")
    infra_team = teams_by_id.get("team_infra")
    return {
        "renewal_pressure": renewal_pressure,
        "trust_gap": trust_gap,
        "expansion_pressure": expansion_pressure,
        "support_stress": support_stress,
        "margin_gap": margin_gap,
        "runway_tightness": runway_tightness,
        "pipeline_slack": pipeline_slack,
        "reliability_gap": max(0.0, min(1.0, reliability_gap)),
        "board_pressure": max(0.0, min(1.0, market_context.board_pressure_level)),
        "growth_capacity": max(0.0, min(1.0, (growth_team.capacity if growth_team else 0.5) - 0.45)),
        "support_capacity": max(0.0, min(1.0, (support_team.capacity if support_team else 0.5) - 0.45)),
        "product_reliability": max(0.0, min(1.0, product_team.execution_reliability if product_team else 0.5)),
        "infra_burnout": max(0.0, min(1.0, infra_team.burnout_risk if infra_team else 0.5)),
    }


def _headcount_program_multiplier(
    episode: dict,
    hidden_goal: HiddenGoal,
    item: InitiativeItem,
    step_index: int,
) -> float:
    signals = _headcount_state_signals(episode)
    multiplier = structured_item_multiplier(
        item,
        hidden_goal,
        step_index=step_index,
        company_profile=episode["company_profile"],
    )
    if item.kind == "growth":
        multiplier += 0.24 * signals["pipeline_slack"]
        multiplier += 0.04 * signals["growth_capacity"]
        multiplier -= 0.18 * signals["support_stress"]
        multiplier -= 0.16 * signals["margin_gap"]
        multiplier -= 0.12 * signals["runway_tightness"]
        multiplier -= 0.10 * signals["reliability_gap"]
    elif item.kind == "retention":
        multiplier += 0.22 * signals["renewal_pressure"]
        multiplier += 0.16 * signals["trust_gap"]
        multiplier += 0.10 * signals["support_stress"]
        multiplier += 0.05 * signals["support_capacity"]
        if item.item_id == "hire_enterprise_csms":
            multiplier += 0.08 * signals["expansion_pressure"]
        if item.item_id == "hire_support_ops":
            multiplier += 0.08 * signals["support_stress"] + 0.05 * signals["reliability_gap"]
    elif item.kind == "revenue":
        multiplier += 0.20 * signals["expansion_pressure"]
        multiplier += 0.10 * signals["board_pressure"]
        multiplier -= 0.18 * signals["trust_gap"]
        multiplier -= 0.10 * signals["support_stress"]
        if item.item_id == "hire_billing_ops":
            multiplier += 0.06 * signals["product_reliability"]
    elif item.kind == "efficiency":
        multiplier += 0.24 * signals["margin_gap"]
        multiplier += 0.18 * signals["support_stress"]
        multiplier += 0.16 * signals["reliability_gap"]
        multiplier += 0.12 * signals["runway_tightness"]
        multiplier += 0.05 * signals["infra_burnout"]
    state = active_state(hidden_goal, step_index)
    if state.planning_horizon.value == "strategic":
        multiplier += 0.05
    elif state.planning_horizon.value == "immediate":
        multiplier -= 0.03
    return max(0.35, min(1.90, multiplier))


def _headcount_programs(rng: random.Random, company_profile, accounts: list[CustomerAccount]) -> list[InitiativeItem]:
    top_accounts = [account.account_id for account in accounts[:3]]
    programs = [
        InitiativeItem(
            item_id="hire_growth_pod",
            title="Hire Growth Pod",
            description="Add PMM + growth engineering capacity to improve activation and acquisition throughput.",
            cost=1.0,
            kind="growth",
            kpi_deltas={"growth": 0.10, "retention": 0.00, "revenue": 0.01, "efficiency": -0.05},
            allocation_unit=1.0,
            allocation_max=3.0,
            saturation_point=2.0,
            beneficiary_segments=["self_serve", "smb", "mid_market"],
            beneficiary_account_ids=top_accounts,
            implementation_risk=0.18,
            synergy_item_ids=["hire_support_ops"],
            conflicts_with_ids=["hire_sre_cluster"],
            risk_notes=[
                "Hiring compounding GTM work pays off slowly and can spread product capacity thin.",
                "Backfires if queue health and delivery reliability are already wobbling.",
            ],
        ),
        InitiativeItem(
            item_id="hire_enterprise_csms",
            title="Hire Enterprise CSMs",
            description="Expand high-touch success coverage for renewal-sensitive and strategic accounts.",
            cost=1.0,
            kind="retention",
            kpi_deltas={"growth": 0.00, "retention": 0.11, "revenue": 0.03, "efficiency": -0.03},
            allocation_unit=1.0,
            allocation_max=3.0,
            saturation_point=2.0,
            beneficiary_segments=["mid_market", "enterprise", "strategic"],
            beneficiary_account_ids=top_accounts,
            implementation_risk=0.14,
            synergy_item_ids=["hire_support_ops", "hire_billing_ops"],
            risk_notes=[
                "Coverage helps trust, but payback is slow unless renewals are genuinely at risk.",
                "Works best when support quality stays high enough for CSM follow-through.",
            ],
        ),
        InitiativeItem(
            item_id="hire_billing_ops",
            title="Hire Billing Ops",
            description="Increase monetization and packaging throughput through commercial operations support.",
            cost=1.0,
            kind="revenue",
            kpi_deltas={"growth": -0.01, "retention": -0.02, "revenue": 0.11, "efficiency": 0.01},
            allocation_unit=1.0,
            allocation_max=3.0,
            saturation_point=2.0,
            beneficiary_segments=["mid_market", "enterprise", "strategic"],
            beneficiary_account_ids=top_accounts,
            implementation_risk=0.17,
            requires_item_ids=["hire_enterprise_csms"],
            synergy_item_ids=["hire_enterprise_csms"],
            risk_notes=[
                "Commercial ops hires help expansion, but can backfire if trust is already fragile.",
                "Visible upside is highest after renewal coverage is already credible.",
            ],
        ),
        InitiativeItem(
            item_id="hire_sre_cluster",
            title="Hire SRE Cluster",
            description="Expand reliability and platform operations capacity for cost and incident reduction.",
            cost=1.0,
            kind="efficiency",
            kpi_deltas={"growth": -0.01, "retention": 0.03, "revenue": 0.01, "efficiency": 0.12},
            allocation_unit=1.0,
            allocation_max=3.0,
            saturation_point=2.0,
            beneficiary_segments=["enterprise", "strategic"],
            beneficiary_account_ids=top_accounts,
            implementation_risk=0.12,
            policy_tags=["margin_guardrail", "sla_guardrail"],
            synergy_item_ids=["hire_support_ops"],
            conflicts_with_ids=["hire_growth_pod"],
            risk_notes=[
                "Reliability hires compound over time and help most when burnout is already visible.",
                "Best when leadership is willing to trade near-term growth optics for execution leverage.",
            ],
        ),
        InitiativeItem(
            item_id="hire_support_ops",
            title="Hire Support Ops",
            description="Add support operations and QA capacity to stabilize queue quality and automation coverage.",
            cost=1.0,
            kind="retention",
            kpi_deltas={"growth": 0.01, "retention": 0.09, "revenue": 0.01, "efficiency": 0.06},
            allocation_unit=1.0,
            allocation_max=3.0,
            saturation_point=2.0,
            beneficiary_segments=["smb", "mid_market", "enterprise"],
            beneficiary_account_ids=top_accounts,
            implementation_risk=0.13,
            policy_tags=["sla_guardrail"],
            synergy_item_ids=["hire_enterprise_csms", "hire_growth_pod", "hire_sre_cluster"],
            risk_notes=[
                "Support ops hires reduce fire drills but can be underutilized without workflow discipline.",
                "Acts as a force multiplier when renewals or onboarding quality are already noisy.",
            ],
        ),
    ]
    if company_profile.seed_family in {"developer_api", "plg_analytics"}:
        programs.append(
            InitiativeItem(
                item_id="hire_developer_advocacy",
                title="Hire Developer Advocacy",
                description="Strengthen developer-led adoption and technical education.",
                cost=1.0,
                kind="growth",
                kpi_deltas={"growth": 0.09, "retention": 0.01, "revenue": 0.03, "efficiency": -0.03},
                allocation_unit=1.0,
                allocation_max=2.0,
                saturation_point=1.0,
                beneficiary_segments=["self_serve", "smb", "mid_market"],
                beneficiary_account_ids=top_accounts,
                implementation_risk=0.22,
                requires_item_ids=["hire_growth_pod"],
                synergy_item_ids=["hire_growth_pod"],
                conflicts_with_ids=["hire_sre_cluster"],
                risk_notes=["This is a strong long-horizon bet, but immediate payback is ambiguous."],
            )
        )
    rng.shuffle(programs)
    return programs


def _calendar_events(episode: dict) -> list[SimCalendarEvent]:
    events = []
    for step, event in sorted(episode["events"].items()):
        if step < episode["step_index"] or step > episode["step_index"] + 1:
            continue
        events.append(
            SimCalendarEvent(
                event_id=event["event_id"],
                name=event["name"],
                sim_date=event["sim_date"],
                status="today" if step == episode["step_index"] else "upcoming",
                summary=event["summary"],
                alerts=list(event.get("alerts", [])),
            )
        )
    return events


def _narrative(episode: dict) -> str:
    dashboard = episode["dashboard"]
    market_context = episode["market_context"]
    profile = episode["company_profile"]
    return (
        f"{_quarter_label(episode['step_index'])} opens for {profile.company_name}. "
        f"MRR is {dashboard.mrr:.0f}, ops margin {dashboard.ops_margin:.2f}, support volume {dashboard.support_ticket_volume}, "
        f"and runway {market_context.cash_runway_months} months. Leadership wants a hiring plan that fits the company narrative "
        f"({profile.board_narrative}) and compounds over time rather than just patching the loudest fire."
    )


def build_task7_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    horizon: int,
    world: dict,
) -> dict:
    """Create a multi-quarter headcount planning episode."""
    config = _task7_config()
    start_date = str(config.get("start_date", "2026-04-01"))
    company_profile = world["company_profile"].model_copy(deep=True)
    accounts = [account.model_copy(deep=True) for account in world["accounts"]]
    stakeholders = [stakeholder.model_copy(deep=True) for stakeholder in world["stakeholders"]]
    teams = [team.model_copy(deep=True) for team in world["teams"]]
    market_context = world["market_context"].model_copy(deep=True)
    governance_constraints = [constraint.model_copy(deep=True) for constraint in world["governance_constraints"]]
    dashboard = _initial_dashboard()
    backlog = _headcount_programs(rng, company_profile, accounts)
    per_step_budget = float(config.get("hiring_budget_per_step", 4))
    total_budget = float(config.get("total_hiring_budget", 12))
    episode = {
        "task_id": TaskId.TASK7,
        "company_profile": company_profile,
        "hidden_goal": hidden_goal,
        "dashboard": dashboard,
        "backlog": backlog,
        "accounts": accounts[:6],
        "stakeholders": stakeholders[:4],
        "teams": teams,
        "market_context": market_context,
        "governance_constraints": governance_constraints,
        "budget_remaining": min(per_step_budget, total_budget),
        "quarterly_budget": per_step_budget,
        "total_budget_remaining": total_budget,
        "step_index": 0,
        "horizon": horizon,
        "start_date": start_date,
        "decision_ledger": [],
        "pending_effects": [],
        "realized_effects": [],
        "action_history": [],
        "strategy_signature_history": [],
        "decision_quality_history": [],
        "latent_utility_history": [compute_utility(dashboard.to_metric_vector(), hidden_goal, step_index=0)],
        "goal_history": [hidden_goal.archetype.value],
        "invalid_actions": 0,
        "policy_violations": 0,
        "events": {
            1: {
                "event_id": "planning_review_q2",
                "name": "planning_review_q2",
                "sim_date": sim_date_for_step(start_date, 1),
                "summary": "Leadership review asks whether the first hiring wave is improving execution leverage.",
                "alerts": ["CEO wants evidence that staffing is improving leverage, not just adding cost."],
            },
            2: {
                "event_id": "board_q3_checkin",
                "name": "board_q3_checkin",
                "sim_date": sim_date_for_step(start_date, 2),
                "summary": "The board is reassessing whether headcount growth fits the operating narrative.",
                "alerts": ["Board asked for a tighter headcount story ahead of next quarter."],
            },
        },
        "alerts": [],
        "task_summary": (
            "Allocate quarterly headcount across hiring programs with delayed capacity effects, execution risk, and changing leadership priorities. "
            "The best plan compounds through team health and future operating leverage, not just immediate KPI optics."
        ),
    }
    episode["memory_bank"] = initialize_memory_bank(
        {
            "company_profile": company_profile,
            "accounts": accounts,
            "stakeholders": stakeholders,
            "teams": teams,
            "governance_constraints": governance_constraints,
        },
        TaskId.TASK7.value,
        start_date,
    )
    reset_step_budgets(episode["memory_bank"])
    visible_entity_ids, visible_tags = auto_visible_entities(
        company_profile=company_profile,
        accounts=accounts,
        stakeholders=stakeholders,
        teams=teams,
        alerts=[],
    )
    episode["memory_workspace"] = build_memory_workspace(
        episode["memory_bank"],
        current_step=0,
        visible_entity_ids=visible_entity_ids,
        visible_tags=visible_tags,
    )
    oracle_allocations, oracle_value = solve_oracle_allocations(
        backlog,
        int(episode["budget_remaining"]),
        hidden_goal.weights,
        hidden_goal,
        company_profile,
        step_index=episode["step_index"],
    )
    episode["oracle_value"] = oracle_value
    episode["oracle_allocations"] = oracle_allocations
    return episode


def build_task7_view(episode: dict) -> dict:
    """Create the current observable bundle."""
    signals = _headcount_state_signals(episode)
    stakeholder_notes = [
        f"{stakeholder.name} ({stakeholder.role}) is watching whether hiring matches {episode['company_profile'].board_narrative}."
        for stakeholder in episode["stakeholders"][:2]
    ]
    if signals["support_stress"] >= 0.20 or signals["reliability_gap"] >= 0.20:
        stakeholder_notes.append(
            "Support load and delivery reliability are visibly strained, so front-line growth hiring without enablement could backfire."
        )
    if signals["expansion_pressure"] >= 0.40:
        stakeholder_notes.append(
            "There is visible commercial upside in expansion-ready accounts if execution and coverage stay credible."
        )
    if signals["margin_gap"] >= 0.18 or signals["runway_tightness"] >= 0.18:
        stakeholder_notes.append(
            "Finance is openly scrutinizing hires that add cost before operating leverage shows up."
        )
    return {
        "step_index": episode["step_index"],
        "horizon": episode["horizon"],
        "sim_date": sim_date_for_step(episode["start_date"], episode["step_index"]),
        "sim_day_label": _quarter_label(episode["step_index"]),
        "task_summary": episode["task_summary"],
        "narrative": _narrative(episode),
        "dashboard": episode["dashboard"].model_copy(deep=True),
        "backlog": [item.model_copy(deep=True) for item in episode["backlog"]],
        "accounts": [account.model_copy(deep=True) for account in episode["accounts"]],
        "stakeholders": [stakeholder.model_copy(deep=True) for stakeholder in episode["stakeholders"]],
        "teams": [team.model_copy(deep=True) for team in episode["teams"]],
        "company_profile": episode["company_profile"].model_copy(deep=True),
        "market_context": episode["market_context"].model_copy(deep=True),
        "governance_constraints": [constraint.model_copy(deep=True) for constraint in episode["governance_constraints"]],
        "alerts": list(episode["alerts"]),
        "calendar_events": _calendar_events(episode),
        "decision_ledger": [entry.model_copy(deep=True) for entry in episode["decision_ledger"]],
        "pending_effects": [effect.model_copy(deep=True) for effect in episode["pending_effects"]],
        "realized_effects": [effect.model_copy(deep=True) for effect in episode["realized_effects"]],
        "budget_remaining": episode["budget_remaining"],
        "memory_summary": None,
        "memory_workspace": episode["memory_workspace"],
        "memory_budget_remaining": episode["memory_bank"]["retrieval_budget_remaining"],
        "memory_write_budget_remaining": episode["memory_bank"]["write_budget_remaining"],
        "sprint_budget": episode["quarterly_budget"],
        "stakeholder_notes": stakeholder_notes,
    }


def evaluate_task7_action_value(
    episode: dict,
    action: LatentGoalOpsAction,
    weights: dict[str, float],
    objective: HiddenGoal | None = None,
) -> float:
    """Approximate the strategic value of a task-7 hiring allocation."""
    objective = objective or episode["hidden_goal"]
    budget_limit = float(episode["budget_remaining"])
    allocations = {
        item.item_id: max(
            0.0,
            min(float(action.budget_allocations.get(item.item_id, 0.0)), float(item.allocation_max or 0.0)),
        )
        for item in episode["backlog"]
    }
    spent = sum(allocations.values())
    if spent > budget_limit + 1e-9:
        return 0.0

    score = allocation_value(
        allocations,
        episode["backlog"],
        weights,
        objective,
        episode["company_profile"],
        episode["step_index"],
    )
    chosen = {
        item.item_id: (item, allocations[item.item_id])
        for item in episode["backlog"]
        if allocations[item.item_id] > 0.0
    }
    if not chosen:
        return 0.0

    kind_totals: dict[str, float] = {}
    for item_id, (item, amount) in chosen.items():
        effective_units = _curve_units(amount, float(item.saturation_point or amount))
        multiplier = _headcount_program_multiplier(episode, objective, item, episode["step_index"])
        score += (multiplier - 1.0) * float(weights.get(item.kind, 0.0)) * effective_units * 0.55
        if item.requires_item_ids and not any(required in chosen for required in item.requires_item_ids):
            score -= 0.035 * effective_units
        kind_totals[item.kind] = kind_totals.get(item.kind, 0.0) + amount

    total_allocated = sum(kind_totals.values())
    dominant_share = max(kind_totals.values()) / total_allocated if total_allocated > 0.0 else 0.0
    if dominant_share >= 0.70:
        score += 0.045
    score -= 0.028 * max(0, len(chosen) - 2)
    score -= 0.012 * max(0.0, budget_limit - spent)
    return max(0.0, score)


def solve_task7_oracle_action(
    episode: dict,
    weights: dict[str, float],
    objective: HiddenGoal | None = None,
    step_index: int | None = None,
) -> tuple[LatentGoalOpsAction, float]:
    """Find a strong visible quarterly hiring allocation."""
    current_step = episode["step_index"] if step_index is None else int(step_index)
    allocations, value = solve_oracle_allocations(
        episode["backlog"],
        int(episode["budget_remaining"]),
        weights,
        objective or episode["hidden_goal"],
        episode["company_profile"],
        step_index=current_step,
    )
    return (
        LatentGoalOpsAction(
            task_id=TaskId.TASK7,
            budget_allocations=allocations,
            rationale_summary="Oracle quarterly headcount plan.",
        ),
        value,
    )


def _team_for_kind(teams: list[InternalTeamState], kind: str) -> InternalTeamState | None:
    mapping = {
        "growth": "team_growth",
        "retention": "team_support",
        "revenue": "team_product",
        "efficiency": "team_infra",
    }
    target_id = mapping.get(kind)
    for team in teams:
        if team.team_id == target_id:
            return team
    return None


def _aggregate_effect_channels(effects: list[TemporalEffectRecord]) -> dict[str, float]:
    aggregate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    for effect in effects:
        for channel, value in effect.channel_deltas.items():
            aggregate[channel] += float(value)
    return aggregate


def _apply_pending_hiring_drag(episode: dict, next_step: int) -> None:
    future_pending = [
        effect
        for effect in episode["pending_effects"]
        if effect.scheduled_for_step > next_step
    ]
    if not future_pending:
        return
    pending_load = sum(
        sum(abs(float(value)) for value in effect.channel_deltas.values())
        for effect in future_pending
    )
    pending_count = len(future_pending)
    episode["dashboard"] = _apply_channel_deltas(
        episode["dashboard"],
        {
            "growth": -min(0.012, 0.004 * max(0, pending_count - 1)),
            "retention": -min(0.010, 0.003 * max(0, pending_count - 1)),
            "efficiency": -min(0.028, 0.012 * pending_count + 0.010 * pending_load),
            "revenue": -min(0.010, 0.003 * max(0, pending_count - 2)),
        },
    )
    team_loads: dict[str, int] = {}
    for effect in future_pending:
        for team_id in effect.affected_team_ids:
            team_loads[team_id] = team_loads.get(team_id, 0) + 1
    for team in episode["teams"]:
        overlap = team_loads.get(team.team_id, 0)
        if overlap <= 0:
            continue
        team.capacity = max(0.0, min(1.0, team.capacity - 0.012 * overlap))
        team.execution_reliability = max(0.0, min(1.0, team.execution_reliability - 0.010 * overlap))
        team.cross_team_friction = max(0.0, min(1.0, team.cross_team_friction + 0.012 * max(0, overlap - 1)))
    episode["market_context"].board_pressure_level = max(
        0.0,
        min(
            1.0,
            episode["market_context"].board_pressure_level + min(0.06, 0.015 * pending_count + 0.020 * pending_load),
        ),
    )


def apply_task7_action(rng: random.Random, hidden_goal: HiddenGoal, episode: dict, action: LatentGoalOpsAction) -> dict:
    """Advance the quarterly headcount planner one step."""
    current_step = episode["step_index"]
    decision_id = f"headcount_{current_step}"
    sim_date = sim_date_for_step(episode["start_date"], current_step)
    next_step = current_step + 1
    enable_delayed_effects = bool(episode.get("enable_delayed_effects", True))
    visible = {item.item_id: item for item in episode["backlog"]}
    allocations = {
        item_id: float(int(round(amount)))
        for item_id, amount in action.budget_allocations.items()
        if item_id in visible and float(amount) > 0.0
    }
    spent = sum(allocations.values())
    invalid = spent > episode["budget_remaining"] + 1e-9
    if invalid:
        allocations = {}
        spent = 0.0
        episode["invalid_actions"] = int(episode.get("invalid_actions", 0)) + 1

    scheduled_effects: list[TemporalEffectRecord] = []
    onboarding_drag = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    if not invalid:
        episode["total_budget_remaining"] -= spent
        for item_id, amount in allocations.items():
            item = visible[item_id]
            effective_units = _curve_units(amount, float(item.saturation_point or amount))
            multiplier = _headcount_program_multiplier(episode, hidden_goal, item, current_step)
            for channel in onboarding_drag:
                onboarding_drag[channel] += float(item.kpi_deltas.get(channel, 0.0)) * 0.12 * effective_units * multiplier
            onboarding_drag["efficiency"] -= 0.012 * amount + max(0.0, 1.0 - multiplier) * 0.012 * amount
            if item.kind in {"growth", "revenue"} and episode["dashboard"].support_ticket_volume >= 180:
                onboarding_drag["retention"] -= 0.012 * effective_units
            lag_steps = 2 if item.kind in {"growth", "revenue"} else 1
            scheduled_effects.append(
                _effect_record(
                    effect_id=f"{decision_id}_{item_id}",
                    decision_id=decision_id,
                    source_type="hiring",
                    source_id=item_id,
                    summary=f"Hiring for {item.title} starts compounding into team leverage.",
                    scheduled_for_step=current_step + lag_steps,
                    start_date=episode["start_date"],
                    channel_deltas={
                        channel: float(item.kpi_deltas.get(channel, 0.0)) * 0.72 * effective_units * multiplier
                        for channel in item.kpi_deltas
                    },
                    affected_team_ids=[_team_for_kind(episode["teams"], item.kind).team_id] if _team_for_kind(episode["teams"], item.kind) else [],
                    affected_account_ids=list(item.beneficiary_account_ids),
                )
            )
        if not enable_delayed_effects:
            immediate_from_scheduled = _aggregate_effect_channels(scheduled_effects)
            for channel, value in immediate_from_scheduled.items():
                onboarding_drag[channel] += value
            scheduled_effects = []
        episode["dashboard"] = _apply_channel_deltas(episode["dashboard"], onboarding_drag)

    episode["pending_effects"].extend(scheduled_effects)
    realized_effects: list[TemporalEffectRecord] = []
    remaining_pending: list[TemporalEffectRecord] = []
    for effect in episode["pending_effects"]:
        if effect.scheduled_for_step == next_step:
            realized = effect.model_copy(
                update={
                    "realized_step": next_step,
                    "realized_date": sim_date_for_step(episode["start_date"], next_step),
                },
                deep=True,
            )
            realized_effects.append(realized)
        else:
            remaining_pending.append(effect)
    episode["pending_effects"] = remaining_pending
    _apply_pending_hiring_drag(episode, next_step)
    effect_channels = _aggregate_effect_channels(realized_effects)
    for channel in effect_channels:
        effect_channels[channel] += rng.uniform(-0.004, 0.004)
    episode["dashboard"] = _apply_channel_deltas(episode["dashboard"], effect_channels)

    for item_id, amount in allocations.items():
        team = _team_for_kind(episode["teams"], visible[item_id].kind)
        if team is None:
            continue
        effective_units = _curve_units(amount, float(visible[item_id].saturation_point or amount))
        multiplier = _headcount_program_multiplier(episode, hidden_goal, visible[item_id], current_step)
        team.capacity = max(0.0, min(1.0, team.capacity + 0.045 * effective_units))
        team.burnout_risk = max(0.0, min(1.0, team.burnout_risk - 0.028 * effective_units * multiplier))
        team.execution_reliability = max(0.0, min(1.0, team.execution_reliability + 0.018 * effective_units * multiplier))
    if invalid:
        for team in episode["teams"]:
            team.cross_team_friction = max(0.0, min(1.0, team.cross_team_friction + 0.03))

    episode["market_context"].board_pressure_level = max(
        0.0,
        min(
            1.0,
            episode["market_context"].board_pressure_level
            - 0.03 * max(0.0, effect_channels["revenue"] + effect_channels["efficiency"])
            + (0.05 if invalid else 0.0),
        ),
    )
    episode["market_context"].cash_runway_months = max(
        3,
        episode["market_context"].cash_runway_months - (1 if spent >= 3 and effect_channels["efficiency"] < 0.02 else 0),
    )

    entry = DecisionLedgerEntry(
        decision_id=decision_id,
        step_index=current_step,
        sim_date=sim_date,
        chosen_initiatives=[f"{item_id}:{int(amount)}" for item_id, amount in sorted(allocations.items())],
        rationale=action.rationale_summary or action.rationale,
        expected_channels=_aggregate_effect_channels(scheduled_effects),
        scheduled_effect_ids=[effect.effect_id for effect in scheduled_effects],
    )
    episode["decision_ledger"].append(entry)
    episode["realized_effects"] = realized_effects
    episode["step_index"] = next_step
    episode["action_history"].append(action)
    episode["goal_history"].append(active_goal_name(hidden_goal, episode["step_index"]))
    episode["budget_remaining"] = min(episode["quarterly_budget"], episode["total_budget_remaining"])
    episode["alerts"] = []
    if next_step in episode["events"]:
        episode["alerts"].extend(list(episode["events"][next_step].get("alerts", [])))
    if episode["total_budget_remaining"] <= 2:
        episode["alerts"].append("Remaining annual hiring budget is getting tight.")

    bank = episode["memory_bank"]
    record_decision(
        bank,
        step_index=current_step,
        sim_date=sim_date,
        decision_id=decision_id,
        summary=f"{_quarter_label(current_step)} hiring plan: {', '.join(entry.chosen_initiatives) or 'no hires'}",
        chosen_initiatives=list(allocations.keys()),
        governance_flags=[],
        company_id=episode["company_profile"].company_id,
    )
    for effect in realized_effects:
        record_effect(
            bank,
            step_index=next_step,
            sim_date=sim_date_for_step(episode["start_date"], next_step),
            effect_id=effect.effect_id,
            summary=effect.summary,
            account_ids=list(effect.affected_account_ids),
            team_ids=list(effect.affected_team_ids),
            source_id=effect.source_id,
            tags=["hiring"],
        )
    for team in episode["teams"]:
        record_team_update(
            bank,
            step_index=next_step,
            sim_date=sim_date_for_step(episode["start_date"], next_step),
            team=team,
            summary=f"{team.function.title()} capacity {team.capacity:.2f}, burnout {team.burnout_risk:.2f}, reliability {team.execution_reliability:.2f}.",
            tags=["hiring_plan"],
        )
    apply_agent_writes(bank, action.memory_writes, current_step, sim_date)
    reset_step_budgets(bank)
    visible_entity_ids, visible_tags = auto_visible_entities(
        company_profile=episode["company_profile"],
        accounts=episode["accounts"],
        stakeholders=episode["stakeholders"],
        teams=episode["teams"],
        alerts=episode["alerts"],
    )
    episode["memory_workspace"] = build_memory_workspace(
        bank,
        current_step=episode["step_index"],
        focus_requests=action.memory_focus,
        visible_entity_ids=visible_entity_ids,
        visible_tags=visible_tags,
    )
    episode["latent_utility_history"].append(
        compute_utility(
            episode["dashboard"].to_metric_vector(),
            hidden_goal,
            step_index=episode["step_index"],
        )
    )

    return {
        "invalid": invalid,
        "spent_budget": spent,
        "scheduled_effects": scheduled_effects,
        "realized_effects": realized_effects,
        "decision_id": decision_id,
        "budget_remaining": episode["budget_remaining"],
        "latent_utility": compute_utility(
            episode["dashboard"].to_metric_vector(),
            hidden_goal,
            step_index=episode["step_index"],
        ),
    }
