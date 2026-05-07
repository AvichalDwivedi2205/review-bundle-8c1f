"""Task 3: startup week with explicit simulation time and delayed effects."""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from datetime import date, datetime, timedelta

from latentgoalops.models import (
    CompanyProfile,
    CustomerAccount,
    DashboardState,
    DecisionLedgerEntry,
    GovernanceConstraint,
    InboxItem,
    InitiativeItem,
    InternalTeamState,
    LatentGoalOpsAction,
    MarketContext,
    MessagingAction,
    SimCalendarEvent,
    StakeholderPersona,
    SupportPolicy,
    TaskId,
    TemporalEffectRecord,
)
from latentgoalops.server.memory import (
    apply_agent_writes,
    auto_visible_entities,
    build_memory_workspace,
    initialize_memory_bank,
    mark_conflict_resolved,
    record_account_update,
    record_decision,
    record_effect,
    record_event,
    record_team_update,
    reset_step_budgets,
)
from latentgoalops.server.objective_utils import structured_item_multiplier
from latentgoalops.server.config import load_config
from latentgoalops.server.hidden_goals import HiddenGoal, active_state, active_weights, compute_utility
from latentgoalops.server.tasks.template_bank import load_events, load_initiative_effects


def initial_dashboard() -> DashboardState:
    """Base startup operating state."""
    return DashboardState(
        dau=16000.0,
        mau=76000.0,
        d7_retention=0.41,
        d30_retention=0.24,
        mrr=38000.0,
        arpu=92.0,
        cac=67.0,
        churn_rate=0.048,
        ops_margin=0.34,
        infra_cost_per_unit=2.10,
        support_ticket_volume=142,
    )


def _parse_date(raw: str) -> date:
    return datetime.fromisoformat(raw).date()


def sim_date_for_step(start_date: str, step_index: int) -> str:
    """Return ISO simulation date for a given zero-based step."""
    return (_parse_date(start_date) + timedelta(days=step_index)).isoformat()


def sim_day_label(step_index: int) -> str:
    """Return a readable day label."""
    return f"Day {step_index + 1}"


def _task3_config() -> dict:
    return load_config("tasks.yaml")["task3"]


def _beneficiary_segments(kind: str) -> list[str]:
    if kind == "growth":
        return ["self_serve", "smb", "mid_market"]
    if kind == "retention":
        return ["mid_market", "enterprise", "strategic"]
    if kind == "revenue":
        return ["mid_market", "enterprise", "strategic"]
    return ["smb", "enterprise", "strategic"]


def _policy_tags(name: str, kind: str) -> list[str]:
    tags = []
    if "pricing" in name:
        tags.append("pricing_guardrail")
    if kind == "efficiency":
        tags.append("margin_guardrail")
    if "support" in name or "incident" in name:
        tags.append("sla_guardrail")
    return tags


def _load_initiatives(accounts: list[CustomerAccount], rng: random.Random, split: str) -> list[InitiativeItem]:
    templates = load_initiative_effects(split)
    items: list[InitiativeItem] = []
    for name, template in templates.items():
        kind = str(template["kind"])
        beneficiary_segments = _beneficiary_segments(kind)
        candidate_accounts = [account for account in accounts if account.segment in beneficiary_segments]
        linked_accounts = [
            account.account_id
            for account in rng.sample(candidate_accounts, k=min(len(candidate_accounts), rng.randint(1, 3)))
        ] if candidate_accounts else []
        items.append(
            InitiativeItem(
                item_id=name,
                title=name.replace("_", " ").title(),
                description=f"Operational initiative to {name.replace('_', ' ')}.",
                cost=float(template["cost"]),
                kind=kind,
                kpi_deltas={key: float(value) for key, value in template["deltas"].items()},
                uncertainty_band=0.10,
                stakeholder_tag=str(template.get("stakeholder_tag", "leadership")),
                lag_steps=int(template.get("lag_steps", 1)),
                effect_window=int(template.get("effect_window", 1)),
                delivery_note=str(template.get("delivery_note", "") or ""),
                beneficiary_segments=beneficiary_segments,
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.08, 0.35), 3),
                policy_tags=_policy_tags(name, kind),
            )
        )
    _annotate_initiative_structure(items, {account.account_id: account for account in accounts})
    return items


def _initiative_prefix(item_id: str) -> str:
    return item_id


def _annotate_initiative_structure(
    items: list[InitiativeItem],
    accounts_by_id: dict[str, CustomerAccount],
) -> None:
    by_prefix = {_initiative_prefix(item.item_id): item for item in items}
    referral = by_prefix.get("launch_referral_loop")
    onboarding = by_prefix.get("improve_onboarding")
    pricing = by_prefix.get("ship_usage_pricing")
    analytics = by_prefix.get("launch_admin_analytics")
    infra = by_prefix.get("optimize_infra")
    incident = by_prefix.get("refactor_incident_tooling")
    triage = by_prefix.get("automate_support_triage")
    login_fix = by_prefix.get("fix_login_bug")

    if referral and onboarding:
        referral.requires_item_ids = [onboarding.item_id]
        referral.synergy_item_ids = [onboarding.item_id]
        onboarding.synergy_item_ids = [referral.item_id]
    if pricing and analytics:
        pricing.requires_item_ids = [analytics.item_id]
        pricing.synergy_item_ids = [analytics.item_id]
        analytics.synergy_item_ids = [pricing.item_id]
    if pricing and referral:
        pricing.conflicts_with_ids = [referral.item_id]
        referral.conflicts_with_ids = [pricing.item_id]
    if infra and incident:
        infra.synergy_item_ids = [incident.item_id]
        incident.synergy_item_ids = [infra.item_id]
    if login_fix and onboarding:
        login_fix.synergy_item_ids = sorted(set(login_fix.synergy_item_ids + [onboarding.item_id]))
    if triage:
        triage.risk_notes.append("Can lower queue cost quickly, but service quality risk rises if premium accounts are already tense.")

    for item in items:
        linked_accounts = [accounts_by_id[account_id] for account_id in item.beneficiary_account_ids if account_id in accounts_by_id]
        risk_notes = list(item.risk_notes)
        if item.requires_item_ids:
            risk_notes.append("Some upside is gated on a visible prerequisite landing first.")
        if item.conflicts_with_ids:
            risk_notes.append("Can create mixed operating signals if paired with a conflicting initiative.")
        if any(account.renewal_window_days <= 30 for account in linked_accounts):
            risk_notes.append("Several linked accounts are already close to renewal, so rollout quality matters.")
        if any(account.segment in {"enterprise", "strategic"} for account in linked_accounts):
            risk_notes.append("A meaningful part of the impact lands on high-touch enterprise accounts.")
        item.risk_notes = risk_notes


def _schedule_events(rng: random.Random, horizon: int, start_date: str, split: str) -> dict[int, list[dict]]:
    config = load_events(split)
    names = list(config.keys())
    rng.shuffle(names)
    available_steps = list(range(1, max(horizon - 1, 2)))
    if not available_steps:
        return {}
    max_count = min(3, len(names), len(available_steps))
    min_count = min(2, max_count)
    count = rng.randint(min_count, max_count) if max_count >= 2 else max_count
    event_steps = sorted(rng.sample(available_steps, k=count))
    schedule: dict[int, list[dict]] = {}
    for idx, (step, name) in enumerate(zip(event_steps, names[:count])):
        raw = config[name]
        schedule.setdefault(step, []).append(
            {
                "event_id": f"event_{idx}_{name}",
                "name": name,
                "summary": str(raw.get("summary", name.replace("_", " ").title())),
                "clue_goal": str(raw["clue_goal"]),
                "alerts": list(raw.get("alerts", [])),
                "effects": {key: float(value) for key, value in raw.get("effects", {}).items()},
                "lag_steps": int(raw.get("lag_steps", 1)),
                "sim_date": sim_date_for_step(start_date, step),
            }
        )
    return schedule


def _calendar_events(episode: dict, current_step: int) -> list[SimCalendarEvent]:
    entries: list[SimCalendarEvent] = []
    for step in sorted(episode["events"]):
        if step < current_step or step > current_step + 2:
            continue
        for event in episode["events"][step]:
            entries.append(
                SimCalendarEvent(
                    event_id=event["event_id"],
                    name=event["name"],
                    sim_date=event["sim_date"],
                    status="today" if step == current_step else "upcoming",
                    summary=event["summary"],
                    alerts=list(event["alerts"]),
                )
            )
    return entries


def _select_lead(stakeholders: list[StakeholderPersona]) -> StakeholderPersona:
    ceo = [stakeholder for stakeholder in stakeholders if stakeholder.role == "CEO"]
    if ceo:
        return max(ceo, key=lambda stakeholder: (stakeholder.political_power, stakeholder.credibility))
    return max(stakeholders, key=lambda stakeholder: (stakeholder.political_power, stakeholder.credibility))


def _pressure_inventory(episode: dict) -> dict[str, dict[str, object]]:
    accounts: list[CustomerAccount] = sorted(
        episode["accounts"],
        key=lambda account: (account.renewal_window_days, -account.relationship_health, -account.annual_contract_value),
    )
    market_context: MarketContext = episode["market_context"]
    dashboard: DashboardState = episode["dashboard"]
    near_term_accounts = [account for account in accounts if account.renewal_window_days <= 30]
    expansion_accounts = sorted(accounts, key=lambda account: account.expansion_potential, reverse=True)[:2]
    renewal_names = ", ".join(account.company_name for account in near_term_accounts[:2]) or "top accounts"
    expansion_names = ", ".join(account.company_name for account in expansion_accounts) or "larger accounts"
    mean_relationship_health = (
        sum(account.relationship_health for account in accounts) / len(accounts)
        if accounts
        else 0.0
    )
    growth_severity = (
        max(0.0, 0.58 - market_context.sales_pipeline_health)
        + 0.35 * market_context.competition_intensity
        + max(0.0, 0.23 - dashboard.dau / max(dashboard.mau, 1.0))
    )
    retention_severity = (
        min(len(near_term_accounts), 4) / 4.0
        + min(dashboard.support_ticket_volume / 220.0, 1.0)
        + max(0.0, 0.62 - mean_relationship_health)
    )
    revenue_severity = (
        0.7 * market_context.board_pressure_level
        + max(0.0, 0.42 - dashboard.arpu / 200.0)
        + max(0.0, 0.55 - market_context.sales_pipeline_health) * 0.3
    )
    efficiency_severity = (
        max(0.0, market_context.gross_margin_target - dashboard.ops_margin) * 2.2
        + min(dashboard.infra_cost_per_unit / 3.0, 1.0)
        + max(0.0, 12 - market_context.cash_runway_months) / 12.0
    )
    return {
        "growth": {
            "severity": growth_severity,
            "line": (
                f"New-logo momentum still looks uneven with competition intensity at {market_context.competition_intensity:.2f} "
                f"and pipeline health around {market_context.sales_pipeline_health:.2f}."
            ),
        },
        "retention": {
            "severity": retention_severity,
            "line": (
                f"Renewal pressure is building around {renewal_names}, while support volume sits at {dashboard.support_ticket_volume} "
                "and execution quality matters more than usual."
            ),
        },
        "revenue": {
            "severity": revenue_severity,
            "line": (
                f"Commercial scrutiny is increasing with board pressure at {market_context.board_pressure_level:.2f}, "
                f"and expansion leverage is concentrated in accounts like {expansion_names}."
            ),
        },
        "efficiency": {
            "severity": efficiency_severity,
            "line": (
                f"Ops margin is {dashboard.ops_margin:.2f} against a {market_context.gross_margin_target:.2f} target, "
                f"with infra cost per unit at {dashboard.infra_cost_per_unit:.2f} and runway at {market_context.cash_runway_months} months."
            ),
        },
    }


def _ordered_pressure_lines(episode: dict) -> list[tuple[str, str]]:
    inventory = _pressure_inventory(episode)
    return [
        (channel, str(payload["line"]))
        for channel, payload in sorted(
            inventory.items(),
            key=lambda row: (float(row[1]["severity"]), row[0]),
            reverse=True,
        )
    ]


def _stakeholder_focus_order(
    rng: random.Random,
    stakeholders: list[StakeholderPersona],
    episode: dict,
) -> list[StakeholderPersona]:
    inventory = _pressure_inventory(episode)
    role_weights = {
        "CEO": 1.6 + max(float(payload["severity"]) for payload in inventory.values()),
        "CFO": 1.1 + float(inventory["revenue"]["severity"]) + float(inventory["efficiency"]["severity"]),
        "CTO": 1.1 + 1.1 * float(inventory["efficiency"]["severity"]) + 0.4 * float(inventory["retention"]["severity"]),
        "Head of CS": 1.1 + 1.2 * float(inventory["retention"]["severity"]),
        "Growth Lead": 1.1 + 1.2 * float(inventory["growth"]["severity"]),
    }
    remaining = stakeholders[:]
    ordered: list[StakeholderPersona] = []
    while remaining:
        weights = [
            role_weights.get(stakeholder.role, 1.0)
            + stakeholder.political_power * 0.6
            + stakeholder.credibility * 0.4
            + rng.uniform(0.0, 0.8)
            for stakeholder in remaining
        ]
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def _persona_message(persona: StakeholderPersona, pressure_lines: list[tuple[str, str]]) -> str:
    fallback = pressure_lines[0][1] if pressure_lines else "The visible signals remain mixed enough that sequencing matters."
    by_channel = {channel: line for channel, line in pressure_lines}
    if persona.role == "CEO":
        primary = pressure_lines[0][1] if pressure_lines else fallback
        counter = pressure_lines[1][1] if len(pressure_lines) > 1 else fallback
        return f"Wants a plan that stays legible across functions. {primary} {counter}"
    if persona.role == "CFO":
        revenue_or_efficiency = by_channel.get("revenue") or by_channel.get("efficiency") or fallback
        counter = by_channel.get("retention") or fallback
        return f"Is pushing for downside control and clean payback logic. {revenue_or_efficiency} {counter}"
    if persona.role == "CTO":
        primary = by_channel.get("efficiency") or fallback
        counter = by_channel.get("retention") or fallback
        return f"Keeps pointing at execution drag, reliability debt, and sequencing risk. {primary} {counter}"
    if persona.role == "Head of CS":
        primary = by_channel.get("retention") or fallback
        counter = by_channel.get("revenue") or fallback
        return f"Is worried about trust-sensitive accounts and renewal readiness. {primary} {counter}"
    primary = by_channel.get("growth") or fallback
    counter = by_channel.get("efficiency") or fallback
    return f"Wants the operating plan to compound rather than scatter effort. {primary} {counter}"


def _build_inbox(
    rng: random.Random,
    episode: dict,
    step_index: int,
    sim_date: str,
    realized_effects: list[TemporalEffectRecord],
    calendar_events: list[SimCalendarEvent],
) -> list[InboxItem]:
    stakeholders: list[StakeholderPersona] = episode["stakeholders"]
    accounts: list[CustomerAccount] = sorted(
        episode["accounts"],
        key=lambda account: (
            account.renewal_window_days <= 30,
            account.churn_propensity,
            account.strategic_importance,
        ),
        reverse=True,
    )
    primary, secondary, *remaining_people = _stakeholder_focus_order(rng, stakeholders, episode)
    hot_account = accounts[0]
    pressure_lines = _ordered_pressure_lines(episode)
    items = [
        InboxItem(
            item_id=f"msg_{step_index}_0",
            text=_persona_message(primary, pressure_lines),
            sender=primary.role,
            metadata={"importance": "high", "timestamp": f"{sim_date}T09:00:00Z", "persona_id": primary.persona_id},
        )
    ]
    items.append(
        InboxItem(
            item_id=f"msg_{step_index}_1",
            text=(
                f"{hot_account.company_name} is a {hot_account.segment.replace('_', ' ')} account "
                f"worth about ${hot_account.annual_contract_value:,.0f} ARR with renewal in "
                f"{hot_account.renewal_window_days} days and relationship health {hot_account.relationship_health:.2f}."
            ),
            sender=hot_account.company_name,
            metadata={"importance": "high", "timestamp": f"{sim_date}T10:00:00Z", "account_id": hot_account.account_id},
        )
    )
    items.append(
        InboxItem(
            item_id=f"msg_{step_index}_2",
            text=_persona_message(secondary, list(reversed(pressure_lines))),
            sender=secondary.role,
            metadata={"importance": "medium", "timestamp": f"{sim_date}T10:45:00Z", "persona_id": secondary.persona_id},
        )
    )
    if realized_effects:
        effect = realized_effects[0]
        items.append(
            InboxItem(
                item_id=f"msg_{step_index}_3",
                text=f"Analytics: {effect.summary}",
                sender="analytics",
                metadata={"importance": "medium", "timestamp": f"{sim_date}T11:00:00Z"},
            )
        )
    elif calendar_events:
        event = calendar_events[0]
        items.append(
            InboxItem(
                item_id=f"msg_{step_index}_3",
                text=f"Chief of staff note: {event.summary}",
                sender="chief_of_staff",
                metadata={"importance": "medium", "timestamp": f"{sim_date}T11:00:00Z"},
            )
        )
    return items


def _memory_summary(episode: dict) -> str:
    workspace = episode.get("memory_workspace")
    if workspace and getattr(workspace, "records", None):
        highlights = "; ".join(record.summary for record in workspace.records[:3])
        return f"Retrieved memory highlights: {highlights}."
    recent_decisions = episode["decision_ledger"][-2:]
    recent_effects = episode["realized_effects"][-2:]
    decision_line = "No prior decisions logged yet."
    if recent_decisions:
        decision_parts = []
        for decision in recent_decisions:
            if decision.chosen_initiatives:
                decision_parts.append(
                    f"{decision.sim_date}: launched {', '.join(decision.chosen_initiatives)}"
                )
            elif decision.messaging_action:
                decision_parts.append(f"{decision.sim_date}: shifted messaging to {decision.messaging_action.value}")
        if decision_parts:
            decision_line = "Recent decisions: " + "; ".join(decision_parts) + "."
    effect_line = "No delayed effects landed today."
    if recent_effects:
        effect_line = "Recent realized effects: " + "; ".join(effect.summary for effect in recent_effects) + "."
    pending_count = len(episode["pending_effects"])
    pending_line = f"{pending_count} delayed effect(s) remain in flight."
    return " ".join([decision_line, effect_line, pending_line])


def _narrative(
    step_index: int,
    sim_date: str,
    dashboard: DashboardState,
    accounts: list[CustomerAccount],
    alerts: list[str],
    realized_effects: list[TemporalEffectRecord],
    pending_effects: list[TemporalEffectRecord],
    market_context: MarketContext,
    episode: dict,
    *,
    show_delayed_context: bool = True,
) -> str:
    alert_line = " ".join(alerts) if alerts else "No critical alerts have fired yet."
    pressure_lines = [line for _, line in _ordered_pressure_lines(episode)[:3]]
    pressure_block = " ".join(pressure_lines)
    if show_delayed_context:
        effect_line = (
            " ".join(effect.summary for effect in realized_effects)
            if realized_effects
            else "No delayed rollouts landed overnight."
        )
        delayed_line = (
            f"There are {len(pending_effects)} delayed effect(s) still in flight, so plan across dates rather than just today's KPI move."
        )
    else:
        effect_line = "The operating picture is still unfolding across earlier decisions, so avoid overreacting to one noisy metric move."
        delayed_line = "Some consequences from earlier work are still uncertain, so sequence the next move cautiously."
    return (
        f"{sim_day_label(step_index)} ({sim_date}) begins with DAU at {dashboard.dau:.0f}, MRR at {dashboard.mrr:.0f}, "
        f"and ops margin at {dashboard.ops_margin:.2f}. Runway is {market_context.cash_runway_months} months and board pressure "
        f"is {market_context.board_pressure_level:.2f}. Stakeholders are sending mixed signals. "
        f"{pressure_block} {alert_line} {effect_line} {delayed_line}"
    )


def build_task3_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    horizon: int,
    budget: float,
    capacity: float,
    world: dict,
    split: str = "core",
) -> dict:
    """Create a startup-week episode with scheduled events."""
    config = _task3_config()
    start_date = str(config.get("start_date", "2026-03-02"))
    company_profile = world["company_profile"].model_copy(deep=True)
    accounts = [account.model_copy(deep=True) for account in world["accounts"]]
    stakeholders = [stakeholder.model_copy(deep=True) for stakeholder in world["stakeholders"]]
    teams = [team.model_copy(deep=True) for team in world["teams"]]
    market_context = world["market_context"].model_copy(deep=True)
    governance_constraints = [constraint.model_copy(deep=True) for constraint in world["governance_constraints"]]
    backlog = _load_initiatives(accounts, rng, split)
    rng.shuffle(backlog)
    backlog = backlog[: rng.randint(5, min(10, len(backlog)))]
    schedule = _schedule_events(rng, horizon, start_date, split)
    dashboard = initial_dashboard()
    episode = {
        "company_profile": company_profile,
        "hidden_goal": hidden_goal,
        "dashboard": dashboard,
        "backlog": backlog,
        "accounts": accounts,
        "stakeholders": stakeholders,
        "teams": teams,
        "market_context": market_context,
        "governance_constraints": governance_constraints,
        "budget_remaining": float(budget),
        "capacity_remaining": float(capacity),
        "completed_ids": set(),
        "step_index": 0,
        "horizon": horizon,
        "start_date": start_date,
        "events": schedule,
        "alerts": [],
        "action_history": [],
        "strategy_signature_history": [],
        "decision_ledger": [],
        "pending_effects": [],
        "realized_effects": [],
        "latent_utility_history": [compute_utility(dashboard.to_metric_vector(), hidden_goal, step_index=0)],
        "goal_history": [hidden_goal.archetype.value],
        "decision_quality_history": [],
        "invalid_actions": 0,
        "policy_violations": 0,
        "task_summary": (
            "Operate the startup across a dated operating calendar, track delayed effects from your decisions, "
            "manage high-value accounts and stakeholder agendas, infer the hidden objective, and adapt if it shifts."
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
        TaskId.TASK3.value,
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
    return episode


def current_goal_name(hidden_goal: HiddenGoal, step_index: int) -> str:
    """Return the active goal at this step."""
    return active_state(hidden_goal, step_index).archetype.value


def build_task3_view(rng: random.Random, hidden_goal: HiddenGoal, episode: dict) -> dict:
    """Create the current observable bundle."""
    expose_ledger = bool(episode.get("expose_decision_ledger", True))
    alerts = list(episode["alerts"])
    sim_date = sim_date_for_step(episode["start_date"], episode["step_index"])
    calendar_events = _calendar_events(episode, episode["step_index"])
    if episode["step_index"] in episode["events"]:
        alerts.extend(event_alert for event in episode["events"][episode["step_index"]] for event_alert in event["alerts"])
    realized_effects = [
        effect.model_copy(deep=True) for effect in episode["realized_effects"]
    ] if expose_ledger else []
    pending_effects = [
        effect.model_copy(deep=True)
        for effect in sorted(episode["pending_effects"], key=lambda effect: (effect.scheduled_for_step, effect.effect_id))
    ] if expose_ledger else []
    return {
        "step_index": episode["step_index"],
        "horizon": episode["horizon"],
        "sim_date": sim_date,
        "sim_day_label": sim_day_label(episode["step_index"]),
        "dashboard": deepcopy(episode["dashboard"]),
        "company_profile": episode["company_profile"].model_copy(deep=True),
        "backlog": [item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]],
        "accounts": [
            account.model_copy(deep=True)
            for account in sorted(
                episode["accounts"],
                key=lambda account: (
                    account.renewal_window_days <= 30,
                    account.strategic_importance,
                    account.annual_contract_value,
                ),
                reverse=True,
            )[:6]
        ],
        "stakeholders": [stakeholder.model_copy(deep=True) for stakeholder in episode["stakeholders"]],
        "teams": [team.model_copy(deep=True) for team in episode["teams"]],
        "market_context": episode["market_context"].model_copy(deep=True),
        "governance_constraints": [constraint.model_copy(deep=True) for constraint in episode["governance_constraints"]],
        "alerts": alerts,
        "calendar_events": calendar_events,
        "inbox": _build_inbox(rng, episode, episode["step_index"], sim_date, realized_effects, calendar_events),
        "narrative": _narrative(
            episode["step_index"],
            sim_date,
            episode["dashboard"],
            episode["accounts"],
            alerts,
            realized_effects,
            pending_effects,
            episode["market_context"],
            episode,
            show_delayed_context=expose_ledger,
        ),
        "budget_remaining": episode["budget_remaining"],
        "capacity_remaining": episode["capacity_remaining"],
        "task_summary": episode["task_summary"],
        "decision_ledger": [entry.model_copy(deep=True) for entry in episode["decision_ledger"]],
        "pending_effects": pending_effects,
        "realized_effects": realized_effects,
        "memory_summary": _memory_summary(episode) if expose_ledger else None,
        "memory_workspace": episode.get("memory_workspace"),
        "memory_budget_remaining": episode.get("memory_bank", {}).get("retrieval_budget_remaining", 0),
        "memory_write_budget_remaining": episode.get("memory_bank", {}).get("write_budget_remaining", 0),
    }


def _apply_channel_deltas(dashboard: DashboardState, channel_deltas: dict[str, float]) -> DashboardState:
    updated = dashboard.model_copy(deep=True)
    updated.dau += channel_deltas.get("growth", 0.0) * 2400.0
    updated.mau += channel_deltas.get("growth", 0.0) * 6200.0
    updated.d7_retention = max(0.0, min(1.0, updated.d7_retention + channel_deltas.get("retention", 0.0) * 0.15))
    updated.d30_retention = max(0.0, min(1.0, updated.d30_retention + channel_deltas.get("retention", 0.0) * 0.12))
    updated.mrr += channel_deltas.get("revenue", 0.0) * 18000.0
    updated.arpu += channel_deltas.get("revenue", 0.0) * 18.0
    updated.churn_rate = max(
        0.0,
        updated.churn_rate - channel_deltas.get("retention", 0.0) * 0.06 + max(0.0, -channel_deltas.get("revenue", 0.0)) * 0.02,
    )
    updated.ops_margin = max(0.0, min(1.0, updated.ops_margin + channel_deltas.get("efficiency", 0.0) * 0.20))
    updated.infra_cost_per_unit = max(0.1, updated.infra_cost_per_unit - channel_deltas.get("efficiency", 0.0) * 1.1)
    updated.support_ticket_volume = max(
        0,
        int(updated.support_ticket_volume - channel_deltas.get("efficiency", 0.0) * 60 - channel_deltas.get("retention", 0.0) * 25 + max(0.0, -channel_deltas.get("growth", 0.0)) * 20),
    )
    return updated


def _apply_dashboard_deltas(dashboard: DashboardState, dashboard_deltas: dict[str, float]) -> DashboardState:
    updated = dashboard.model_copy(deep=True)
    for key, value in dashboard_deltas.items():
        if key == "support_ticket_volume":
            updated.support_ticket_volume = max(0, int(updated.support_ticket_volume + value))
        elif hasattr(updated, key):
            setattr(updated, key, getattr(updated, key) + value)
    updated.d7_retention = max(0.0, min(1.0, updated.d7_retention))
    updated.d30_retention = max(0.0, min(1.0, updated.d30_retention))
    updated.churn_rate = max(0.0, updated.churn_rate)
    updated.ops_margin = max(0.0, min(1.0, updated.ops_margin))
    updated.infra_cost_per_unit = max(0.1, updated.infra_cost_per_unit)
    return updated


def _effect_record(
    *,
    effect_id: str,
    decision_id: str | None,
    source_type: str,
    source_id: str,
    summary: str,
    scheduled_for_step: int,
    start_date: str,
    channel_deltas: dict[str, float] | None = None,
    dashboard_deltas: dict[str, float] | None = None,
    affected_account_ids: list[str] | None = None,
    affected_team_ids: list[str] | None = None,
) -> TemporalEffectRecord:
    return TemporalEffectRecord(
        effect_id=effect_id,
        decision_id=decision_id,
        source_type=source_type,
        source_id=source_id,
        summary=summary,
        channel_deltas=channel_deltas or {},
        dashboard_deltas=dashboard_deltas or {},
        affected_account_ids=affected_account_ids or [],
        affected_team_ids=affected_team_ids or [],
        scheduled_for_step=scheduled_for_step,
        scheduled_for_date=sim_date_for_step(start_date, scheduled_for_step),
    )


def _team_for_kind(episode: dict, kind: str) -> InternalTeamState | None:
    target_team = {
        "growth": "team_growth",
        "retention": "team_support",
        "revenue": "team_product",
        "efficiency": "team_infra",
    }.get(kind)
    for team in episode["teams"]:
        if team.team_id == target_team:
            return team
    return None


def _initiative_outcome_multiplier(
    episode: dict,
    hidden_goal: HiddenGoal,
    item: InitiativeItem,
    current_step: int,
) -> float:
    team = _team_for_kind(episode, item.kind)
    team_factor = 1.0
    if team is not None:
        team_factor += (team.capacity - 0.5) * 0.35
        team_factor += (team.execution_reliability - 0.5) * 0.25
        team_factor -= team.burnout_risk * 0.18
        team_factor -= team.cross_team_friction * 0.12
    account_factor = 1.0
    linked_accounts = [
        account
        for account in episode["accounts"]
        if account.account_id in set(item.beneficiary_account_ids)
    ]
    if linked_accounts:
        near_renewal = sum(1 for account in linked_accounts if account.renewal_window_days <= 30) / len(linked_accounts)
        mean_health = sum(account.relationship_health for account in linked_accounts) / len(linked_accounts)
        if item.kind == "retention":
            account_factor += near_renewal * 0.22 + max(0.0, 0.65 - mean_health) * 0.18
        elif item.kind == "revenue":
            account_factor -= near_renewal * 0.20
        elif item.kind == "growth":
            account_factor += max(0.0, 0.55 - episode["market_context"].sales_pipeline_health) * 0.20
        elif item.kind == "efficiency":
            account_factor += max(0.0, episode["market_context"].gross_margin_target - episode["dashboard"].ops_margin) * 0.35
    structured = structured_item_multiplier(
        item,
        hidden_goal,
        step_index=current_step,
        company_profile=episode.get("company_profile"),
    )
    return max(0.45, min(1.55, team_factor * account_factor * structured))


def _message_effect(action) -> dict[str, float]:
    mapping = {
        MessagingAction.GROWTH_PUSH: {"growth": 0.03},
        MessagingAction.RETENTION_CAMPAIGN: {"retention": 0.03},
        MessagingAction.REVENUE_UPSELL: {"revenue": 0.04},
        MessagingAction.COST_COMMS: {"efficiency": 0.03},
    }
    return mapping.get(action.messaging_action, {})


def _support_immediate_effect(action) -> dict[str, float]:
    if action.support_policy == SupportPolicy.PREMIUM_SLA:
        return {"retention": 0.02, "efficiency": -0.01}
    if action.support_policy == SupportPolicy.AUTOMATION_FIRST:
        return {"efficiency": 0.02, "retention": -0.005}
    if action.support_policy == SupportPolicy.INCIDENT_SWARM:
        return {"retention": 0.01, "efficiency": 0.01}
    return {"retention": 0.01, "efficiency": 0.01}


def _schedule_initiative_effects(
    episode: dict,
    hidden_goal: HiddenGoal,
    chosen_items: list[InitiativeItem],
    current_step: int,
    start_date: str,
    decision_id: str,
) -> tuple[dict[str, float], list[TemporalEffectRecord]]:
    immediate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    scheduled: list[TemporalEffectRecord] = []
    for item in chosen_items:
        multiplier = _initiative_outcome_multiplier(episode, hidden_goal, item, current_step)
        immediate_fraction = 0.25
        delayed_fraction = 0.75
        base_lag = int(item.lag_steps)
        window = max(1, item.effect_window)
        if item.kind in {"growth", "revenue"}:
            immediate_fraction = 0.18
            delayed_fraction = 0.82
            base_lag += 1
            window = max(2, window)
        elif item.kind == "efficiency":
            immediate_fraction = 0.22
            delayed_fraction = 0.78
        for channel, value in item.kpi_deltas.items():
            immediate[channel] += float(value) * immediate_fraction * multiplier
        for offset in range(window):
            scheduled_step = current_step + base_lag + offset
            if scheduled_step > int(episode["horizon"]):
                continue
            scheduled.append(
                _effect_record(
                    effect_id=f"{decision_id}_{item.item_id}_{offset}",
                    decision_id=decision_id,
                    source_type="initiative",
                    source_id=item.item_id,
                    summary=(
                        item.delivery_note
                        or f"{item.title} shipped and is now moving operating metrics."
                    ),
                    scheduled_for_step=scheduled_step,
                    start_date=start_date,
                    channel_deltas={
                        channel: float(value) * delayed_fraction * multiplier / window
                        for channel, value in item.kpi_deltas.items()
                    },
                    affected_account_ids=item.beneficiary_account_ids,
                    affected_team_ids=[
                        "team_growth" if item.kind == "growth" else (
                            "team_support" if item.kind == "retention" else (
                                "team_product" if item.kind == "revenue" else "team_infra"
                            )
                        )
                    ],
                )
            )
    return immediate, scheduled


def _apply_pending_queue_drag(episode: dict, next_step: int) -> None:
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
    dashboard_drag = {
        "support_ticket_volume": min(20.0, 3.0 * pending_count + 14.0 * pending_load),
        "ops_margin": -min(0.035, 0.004 * pending_count + 0.016 * pending_load),
    }
    if pending_count >= 2:
        dashboard_drag["d30_retention"] = -min(0.015, 0.004 * (pending_count - 1))
    episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], dashboard_drag)
    team_loads: dict[str, int] = {}
    for effect in future_pending:
        for team_id in effect.affected_team_ids:
            team_loads[team_id] = team_loads.get(team_id, 0) + 1
    for team in episode["teams"]:
        overlap = team_loads.get(team.team_id, 0)
        if overlap <= 0:
            continue
        team.capacity = _clamp(team.capacity - 0.01 * overlap)
        team.execution_reliability = _clamp(team.execution_reliability - 0.012 * overlap)
        team.cross_team_friction = _clamp(team.cross_team_friction + 0.01 * max(0, overlap - 1))
    episode["market_context"].board_pressure_level = _clamp(
        episode["market_context"].board_pressure_level + min(0.05, 0.012 * pending_count + 0.025 * pending_load)
    )


def _schedule_support_effect(action, current_step: int, start_date: str, decision_id: str) -> list[TemporalEffectRecord]:
    if action.support_policy == SupportPolicy.PREMIUM_SLA:
        delayed = {"retention": 0.015}
        summary = "Premium SLA follow-through stabilized a portion of at-risk accounts."
    elif action.support_policy == SupportPolicy.AUTOMATION_FIRST:
        delayed = {"efficiency": 0.02}
        summary = "Automation-first support policy reduced manual queue load."
    elif action.support_policy == SupportPolicy.INCIDENT_SWARM:
        delayed = {"retention": 0.01, "efficiency": 0.01}
        summary = "Incident swarm created cross-functional momentum after the firefight."
    else:
        delayed = {"retention": 0.005, "efficiency": 0.005}
        summary = "Balanced support policy produced a modest follow-through improvement."
    return [
        _effect_record(
            effect_id=f"{decision_id}_support",
            decision_id=decision_id,
            source_type="support_policy",
            source_id=action.support_policy.value if action.support_policy else "none",
            summary=summary,
            scheduled_for_step=current_step + 1,
            start_date=start_date,
            channel_deltas=delayed,
            affected_team_ids=["team_support"],
        )
    ]


def _schedule_pricing_effect(
    episode: dict,
    hidden_goal: HiddenGoal,
    action,
    current_step: int,
    start_date: str,
    decision_id: str,
) -> tuple[dict[str, float], list[TemporalEffectRecord]]:
    immediate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    if action.pricing_change_pct is None or abs(action.pricing_change_pct) < 1e-9:
        return immediate, []
    enterprise_exposure = sum(
        1
        for account in episode["accounts"]
        if account.segment in {"enterprise", "strategic"} and account.renewal_window_days <= 30
    )
    enterprise_pressure = min(1.0, enterprise_exposure / 4.0)
    active_objective = active_state(hidden_goal, current_step)
    company_family = episode.get("company_profile").seed_family if episode.get("company_profile") else ""
    if action.pricing_change_pct > 0:
        immediate["revenue"] += action.pricing_change_pct * (0.14 if company_family == "support_automation" else 0.18)
        summary = "A delayed churn response arrived after the price increase."
        delayed = {
            "retention": -action.pricing_change_pct * (0.10 + enterprise_pressure * 0.08 + (0.05 if active_objective.governance_strictness.value == "strict" else 0.0))
        }
    else:
        immediate["growth"] += abs(action.pricing_change_pct) * (0.16 if company_family in {"plg_analytics", "developer_api"} else 0.12)
        summary = "A delayed monetization headwind arrived after the discounting move."
        delayed = {"revenue": action.pricing_change_pct * (0.08 + 0.03 * enterprise_pressure)}
    return immediate, [
        _effect_record(
            effect_id=f"{decision_id}_pricing",
            decision_id=decision_id,
            source_type="pricing",
            source_id="pricing_change",
            summary=summary,
            scheduled_for_step=current_step + 2,
            start_date=start_date,
            channel_deltas=delayed,
        )
    ]


def _schedule_event_effects(step_events: list[dict], current_step: int, start_date: str) -> list[TemporalEffectRecord]:
    records: list[TemporalEffectRecord] = []
    for event in step_events:
        records.append(
            _effect_record(
                effect_id=f"{event['event_id']}_effect",
                decision_id=None,
                source_type="event",
                source_id=event["event_id"],
                summary=event["summary"],
                scheduled_for_step=current_step + int(event.get("lag_steps", 1)),
                start_date=start_date,
                dashboard_deltas=event.get("effects", {}),
            )
        )
    return records


def _aggregate_effect_channels(effects: list[TemporalEffectRecord]) -> dict[str, float]:
    aggregate = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    for effect in effects:
        for channel, value in effect.channel_deltas.items():
            aggregate[channel] += float(value)
    return aggregate


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _beneficiary_account_ids(
    episode: dict,
    chosen_items: list[InitiativeItem],
    realized_effects: list[TemporalEffectRecord],
) -> set[str]:
    account_ids = {account_id for item in chosen_items for account_id in item.beneficiary_account_ids}
    for effect in realized_effects:
        account_ids.update(effect.affected_account_ids)
    return account_ids


def _governance_flags(episode: dict, action) -> list[GovernanceConstraint]:
    by_id = {constraint.constraint_id: constraint for constraint in episode["governance_constraints"]}
    flags: list[GovernanceConstraint] = []
    strategic_renewals = [
        account
        for account in episode["accounts"]
        if account.segment in {"enterprise", "strategic"} and account.renewal_window_days <= 30
    ]
    if (
        action.pricing_change_pct is not None
        and action.pricing_change_pct > 0.05
        and strategic_renewals
        and "pricing_guardrail" in by_id
    ):
        flags.append(by_id["pricing_guardrail"])
    if (
        action.support_policy == SupportPolicy.AUTOMATION_FIRST
        and any(account.support_tier == "premium" or account.segment == "strategic" for account in strategic_renewals)
        and "sla_guardrail" in by_id
    ):
        flags.append(by_id["sla_guardrail"])
    if (
        action.messaging_action == MessagingAction.GROWTH_PUSH
        and "margin_guardrail" in by_id
        and episode["dashboard"].ops_margin < float(by_id["margin_guardrail"].threshold or 0.0)
    ):
        flags.append(by_id["margin_guardrail"])
    return flags


def _schedule_governance_effects(
    flags: list[GovernanceConstraint],
    episode: dict,
    current_step: int,
    start_date: str,
    decision_id: str,
) -> list[TemporalEffectRecord]:
    risk_accounts = [
        account.account_id
        for account in episode["accounts"]
        if account.segment in {"enterprise", "strategic"} or account.renewal_window_days <= 30
    ][:4]
    records: list[TemporalEffectRecord] = []
    for flag in flags:
        if flag.constraint_id == "pricing_guardrail":
            records.append(
                _effect_record(
                    effect_id=f"{decision_id}_gov_pricing",
                    decision_id=decision_id,
                    source_type="governance",
                    source_id=flag.constraint_id,
                    summary="Strategic accounts reacted poorly to a pricing move near renewal windows.",
                    scheduled_for_step=current_step + 1,
                    start_date=start_date,
                    channel_deltas={"retention": -0.05, "revenue": -0.02},
                    affected_account_ids=risk_accounts,
                )
            )
        elif flag.constraint_id == "sla_guardrail":
            records.append(
                _effect_record(
                    effect_id=f"{decision_id}_gov_sla",
                    decision_id=decision_id,
                    source_type="governance",
                    source_id=flag.constraint_id,
                    summary="Automation pressure spilled into premium support queues and trust slipped.",
                    scheduled_for_step=current_step + 1,
                    start_date=start_date,
                    channel_deltas={"retention": -0.04, "efficiency": -0.01},
                    affected_account_ids=risk_accounts,
                    affected_team_ids=["team_support"],
                )
            )
        elif flag.constraint_id == "margin_guardrail":
            records.append(
                _effect_record(
                    effect_id=f"{decision_id}_gov_margin",
                    decision_id=decision_id,
                    source_type="governance",
                    source_id=flag.constraint_id,
                    summary="Board pressure increased because growth spend ran ahead of margin reality.",
                    scheduled_for_step=current_step + 1,
                    start_date=start_date,
                    channel_deltas={"growth": -0.02, "efficiency": -0.04},
                    affected_team_ids=["team_growth", "team_infra"],
                )
            )
    return records


def _support_total_effect_channels(action) -> dict[str, float]:
    total = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    for channel, value in _support_immediate_effect(action).items():
        total[channel] += float(value)
    delayed_map = {
        SupportPolicy.PREMIUM_SLA: {"retention": 0.015},
        SupportPolicy.AUTOMATION_FIRST: {"efficiency": 0.02},
        SupportPolicy.INCIDENT_SWARM: {"retention": 0.01, "efficiency": 0.01},
        SupportPolicy.BALANCED_TRIAGE: {"retention": 0.005, "efficiency": 0.005},
        None: {"retention": 0.005, "efficiency": 0.005},
    }
    for channel, value in delayed_map.get(action.support_policy, {}).items():
        total[channel] += float(value)
    return total


def _pricing_total_effect_channels(action) -> dict[str, float]:
    total = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    if action.pricing_change_pct is None or abs(action.pricing_change_pct) < 1e-9:
        return total
    if action.pricing_change_pct > 0:
        total["revenue"] += action.pricing_change_pct * 0.18
        total["retention"] += -action.pricing_change_pct * 0.10
    else:
        total["growth"] += abs(action.pricing_change_pct) * 0.14
        total["revenue"] += action.pricing_change_pct * 0.08
    return total


def _initiative_bundle_value(
    episode: dict,
    objective: HiddenGoal,
    chosen_items: list[InitiativeItem],
    weights: dict[str, float],
) -> float:
    chosen_ids = {item.item_id for item in chosen_items}
    score = 0.0
    for item in chosen_items:
        base = sum(float(item.kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)
        base *= _initiative_outcome_multiplier(episode, objective, item, episode["step_index"])
        if item.requires_item_ids and not any(required in chosen_ids for required in item.requires_item_ids):
            base *= 0.55
        if item.synergy_item_ids and any(synergy in chosen_ids for synergy in item.synergy_item_ids):
            base *= 1.12
        score += base
    for item in chosen_items:
        for conflict_id in item.conflicts_with_ids:
            if conflict_id in chosen_ids and item.item_id < conflict_id:
                score -= 0.18 * sum(
                    float(item.kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0))
                    for channel in weights
                )
    return max(0.0, score)


def evaluate_task3_action_value(
    episode: dict,
    action: LatentGoalOpsAction,
    weights: dict[str, float],
    objective: HiddenGoal | None = None,
) -> float:
    """Approximate the strategic value of a task-3 action under the active goal."""
    visible_backlog = {item.item_id: item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]}
    chosen_items = [visible_backlog[item_id] for item_id in action.chosen_initiatives if item_id in visible_backlog]
    spent_budget = sum(item.cost for item in chosen_items)
    spent_capacity = sum(max(1.0, item.cost / 2.0) for item in chosen_items)
    if spent_budget > episode["budget_remaining"] + 1e-9 or spent_capacity > episode["capacity_remaining"] + 1e-9:
        return 0.0

    objective = objective or episode["hidden_goal"]
    score = _initiative_bundle_value(episode, objective, chosen_items, weights)
    score += sum(float(_message_effect(action).get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)
    score += sum(
        float(_support_total_effect_channels(action).get(channel, 0.0)) * float(weights.get(channel, 0.0))
        for channel in weights
    )
    score += sum(
        float(_pricing_total_effect_channels(action).get(channel, 0.0)) * float(weights.get(channel, 0.0))
        for channel in weights
    )

    governance_flags = _governance_flags(episode, action)
    governance_penalty = 0.03 * len(governance_flags)
    execution_penalty = sum(
        item.implementation_risk
        * (0.03 if active_state(objective, episode["step_index"]).risk_posture.value == "aggressive" else 0.05)
        for item in chosen_items
    )
    concentration_bonus = 0.02 if chosen_items and len({item.kind for item in chosen_items}) == 1 else 0.0
    idle_penalty = 0.01 if not chosen_items and action.messaging_action == MessagingAction.NONE else 0.0
    return max(0.0, score + concentration_bonus - governance_penalty - execution_penalty - idle_penalty)


def solve_task3_oracle_action(
    episode: dict,
    weights: dict[str, float],
    objective: HiddenGoal | None = None,
) -> tuple[LatentGoalOpsAction, float]:
    """Brute-force a high-quality visible action under the active goal."""
    visible = [item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]]
    best_action = LatentGoalOpsAction(
        task_id=TaskId.TASK3,
        chosen_initiatives=[],
        messaging_action=MessagingAction.NONE,
        pricing_change_pct=0.0,
        support_policy=SupportPolicy.BALANCED_TRIAGE,
        rationale="Oracle action using active-goal rollout value.",
    )
    best_value = evaluate_task3_action_value(episode, best_action, weights, objective=objective)
    pricing_choices = [-0.05, 0.0, 0.05]

    for subset_size in range(len(visible) + 1):
        for subset in itertools.combinations(visible, subset_size):
            subset_ids = [item.item_id for item in subset]
            subset_budget = sum(item.cost for item in subset)
            subset_capacity = sum(max(1.0, item.cost / 2.0) for item in subset)
            if subset_budget > episode["budget_remaining"] + 1e-9:
                continue
            if subset_capacity > episode["capacity_remaining"] + 1e-9:
                continue
            for messaging_action in MessagingAction:
                for support_policy in SupportPolicy:
                    for pricing_change_pct in pricing_choices:
                        candidate = LatentGoalOpsAction(
                            task_id=TaskId.TASK3,
                            chosen_initiatives=subset_ids,
                            messaging_action=messaging_action,
                            pricing_change_pct=pricing_change_pct,
                            support_policy=support_policy,
                            rationale="Oracle action using active-goal rollout value.",
                        )
                        candidate_value = evaluate_task3_action_value(
                            episode,
                            candidate,
                            weights,
                            objective=objective,
                        )
                        if candidate_value > best_value:
                            best_value = candidate_value
                            best_action = candidate
    return best_action, best_value


def _update_accounts(
    episode: dict,
    chosen_items: list[InitiativeItem],
    action,
    immediate_channels: dict[str, float],
    delayed_channels: dict[str, float],
    governance_flags: list[GovernanceConstraint],
    realized_effects: list[TemporalEffectRecord],
) -> list[TemporalEffectRecord]:
    targeted_ids = _beneficiary_account_ids(episode, chosen_items, realized_effects)
    renewal_records: list[TemporalEffectRecord] = []
    for account in episode["accounts"]:
        multiplier = 1.0 if account.account_id in targeted_ids else 0.35
        segment_weight = 1.0
        if account.segment in {"self_serve", "smb"}:
            segment_weight = 1.0 + max(0.0, immediate_channels["growth"] + delayed_channels["growth"]) * 1.2
        elif account.segment in {"enterprise", "strategic"}:
            segment_weight = 1.0 + max(0.0, immediate_channels["retention"] + delayed_channels["retention"]) * 1.1
        account.relationship_health = _clamp(
            account.relationship_health
            + (immediate_channels["retention"] * 0.10 + delayed_channels["retention"] * 0.08) * multiplier * segment_weight
            + (immediate_channels["growth"] * 0.03) * multiplier
            - max(0.0, immediate_channels["efficiency"]) * (0.025 if account.support_tier == "premium" else -0.01) * multiplier
        )
        account.churn_propensity = _clamp(
            account.churn_propensity
            - (immediate_channels["retention"] * 0.10 + delayed_channels["retention"] * 0.08) * multiplier
            + max(0.0, -delayed_channels["retention"]) * 0.06 * multiplier
        )
        account.expansion_potential = _clamp(
            account.expansion_potential
            + (immediate_channels["revenue"] * 0.07 + delayed_channels["revenue"] * 0.05) * multiplier
            + max(0.0, immediate_channels["growth"]) * 0.03
        )
        if action.pricing_change_pct and action.pricing_change_pct > 0 and account.renewal_window_days <= 30:
            account.relationship_health = _clamp(account.relationship_health - action.pricing_change_pct * 0.45 * multiplier)
            account.churn_propensity = _clamp(account.churn_propensity + action.pricing_change_pct * 0.30 * multiplier)
        if action.support_policy == SupportPolicy.PREMIUM_SLA and account.support_tier == "premium":
            account.relationship_health = _clamp(account.relationship_health + 0.03 * multiplier)
            account.churn_propensity = _clamp(account.churn_propensity - 0.025 * multiplier)
        if action.support_policy == SupportPolicy.AUTOMATION_FIRST and account.support_tier == "premium":
            account.relationship_health = _clamp(account.relationship_health - 0.04 * multiplier)
            account.churn_propensity = _clamp(account.churn_propensity + 0.03 * multiplier)
        if governance_flags and account.account_id in targeted_ids:
            account.relationship_health = _clamp(account.relationship_health - 0.03)
            account.churn_propensity = _clamp(account.churn_propensity + 0.03)

    next_step = episode["step_index"] + 1
    for account in episode["accounts"]:
        previous_window = account.renewal_window_days
        account.renewal_window_days = max(0, account.renewal_window_days - 1)
        if previous_window > 0 and account.renewal_window_days == 0:
            renewal_score = (
                account.relationship_health * 0.45
                + account.champion_strength * 0.20
                + (1.0 - account.churn_propensity) * 0.20
                + account.strategic_importance * 0.15
            )
            if renewal_score >= 0.62:
                renewal_delta = round(account.annual_contract_value / 12.0 * (0.04 + 0.05 * account.expansion_potential), 2)
                renewal_records.append(
                    TemporalEffectRecord(
                        effect_id=f"renewal_{account.account_id}_{next_step}",
                        source_type="renewal",
                        source_id=account.account_id,
                        summary=f"{account.company_name} renewed and modestly expanded after a stable operating stretch.",
                        dashboard_deltas={"mrr": renewal_delta, "churn_rate": -0.002},
                        affected_account_ids=[account.account_id],
                        scheduled_for_step=next_step,
                        scheduled_for_date=sim_date_for_step(episode["start_date"], next_step),
                        realized_step=next_step,
                        realized_date=sim_date_for_step(episode["start_date"], next_step),
                    )
                )
                account.relationship_health = _clamp(account.relationship_health + 0.04)
                account.adoption_stage = "steady"
            else:
                churn_delta = round(account.annual_contract_value / 12.0 * 0.08, 2)
                renewal_records.append(
                    TemporalEffectRecord(
                        effect_id=f"renewal_{account.account_id}_{next_step}",
                        source_type="renewal",
                        source_id=account.account_id,
                        summary=f"{account.company_name} entered a renewal scare and near-term revenue slipped.",
                        dashboard_deltas={"mrr": -churn_delta, "churn_rate": 0.004, "support_ticket_volume": 8.0},
                        affected_account_ids=[account.account_id],
                        scheduled_for_step=next_step,
                        scheduled_for_date=sim_date_for_step(episode["start_date"], next_step),
                        realized_step=next_step,
                        realized_date=sim_date_for_step(episode["start_date"], next_step),
                    )
                )
                account.relationship_health = _clamp(account.relationship_health - 0.05)
                account.adoption_stage = "at_risk"
    episode["accounts"].sort(
        key=lambda account: (
            account.renewal_window_days <= 30,
            account.strategic_importance,
            account.annual_contract_value,
        ),
        reverse=True,
    )
    return renewal_records


def _update_teams(
    episode: dict,
    chosen_items: list[InitiativeItem],
    action,
    invalid: bool,
    governance_flags: list[GovernanceConstraint],
    delayed_channels: dict[str, float],
) -> None:
    team_by_id = {team.team_id: team for team in episode["teams"]}
    if action.messaging_action == MessagingAction.GROWTH_PUSH and "team_growth" in team_by_id:
        team_by_id["team_growth"].burnout_risk = _clamp(team_by_id["team_growth"].burnout_risk + 0.04)
    if action.support_policy == SupportPolicy.PREMIUM_SLA and "team_support" in team_by_id:
        team_by_id["team_support"].burnout_risk = _clamp(team_by_id["team_support"].burnout_risk + 0.05)
        team_by_id["team_support"].capacity = _clamp(team_by_id["team_support"].capacity - 0.03)
    if action.support_policy == SupportPolicy.AUTOMATION_FIRST and "team_support" in team_by_id:
        team_by_id["team_support"].capacity = _clamp(team_by_id["team_support"].capacity + 0.04)
        team_by_id["team_support"].burnout_risk = _clamp(team_by_id["team_support"].burnout_risk - 0.03)
    for item in chosen_items:
        target_team = {
            "growth": "team_growth",
            "retention": "team_support",
            "revenue": "team_product",
            "efficiency": "team_infra",
        }.get(item.kind)
        if target_team and target_team in team_by_id:
            team = team_by_id[target_team]
            team.capacity = _clamp(team.capacity - 0.02)
            team.execution_reliability = _clamp(team.execution_reliability + 0.01 * max(0.0, 1.0 - item.implementation_risk))
    if invalid:
        for team in team_by_id.values():
            team.cross_team_friction = _clamp(team.cross_team_friction + 0.03)
    if governance_flags:
        for flag in governance_flags:
            if flag.constraint_id == "margin_guardrail" and "team_growth" in team_by_id:
                team_by_id["team_growth"].cross_team_friction = _clamp(team_by_id["team_growth"].cross_team_friction + 0.04)
            if flag.constraint_id == "sla_guardrail" and "team_support" in team_by_id:
                team_by_id["team_support"].execution_reliability = _clamp(team_by_id["team_support"].execution_reliability - 0.03)
    if delayed_channels["efficiency"] > 0 and "team_infra" in team_by_id:
        team_by_id["team_infra"].capacity = _clamp(team_by_id["team_infra"].capacity + 0.02)
        team_by_id["team_infra"].burnout_risk = _clamp(team_by_id["team_infra"].burnout_risk - 0.02)


def _update_market_context(
    episode: dict,
    immediate_channels: dict[str, float],
    delayed_channels: dict[str, float],
    governance_flags: list[GovernanceConstraint],
) -> None:
    market_context: MarketContext = episode["market_context"]
    board_delta = 0.0
    board_delta += -0.05 * max(0.0, immediate_channels["revenue"] + delayed_channels["revenue"])
    board_delta += -0.04 * max(0.0, immediate_channels["efficiency"] + delayed_channels["efficiency"])
    board_delta += 0.05 * len(governance_flags)
    if episode["dashboard"].ops_margin < market_context.gross_margin_target:
        board_delta += 0.03
    market_context.board_pressure_level = _clamp(market_context.board_pressure_level + board_delta)
    pipeline_delta = 0.06 * max(0.0, immediate_channels["growth"] + delayed_channels["growth"])
    pipeline_delta += 0.04 * max(0.0, immediate_channels["revenue"] + delayed_channels["revenue"])
    pipeline_delta -= 0.03 * len(governance_flags)
    market_context.sales_pipeline_health = _clamp(market_context.sales_pipeline_health + pipeline_delta)
    if immediate_channels["efficiency"] + delayed_channels["efficiency"] > 0.08:
        market_context.cash_runway_months = min(36, market_context.cash_runway_months + 1)
    elif len(governance_flags) >= 2 or episode["dashboard"].ops_margin < max(0.20, market_context.gross_margin_target - 0.12):
        market_context.cash_runway_months = max(3, market_context.cash_runway_months - 1)


def apply_task3_action(rng: random.Random, hidden_goal: HiddenGoal, episode: dict, action) -> dict:
    """Mutate the task 3 episode with the selected action."""
    current_step = episode["step_index"]
    start_date = episode["start_date"]
    enable_delayed_effects = bool(episode.get("enable_delayed_effects", True))
    decision_id = f"decision_{current_step}"
    visible_backlog = {item.item_id: item for item in episode["backlog"] if item.item_id not in episode["completed_ids"]}
    chosen_items = [visible_backlog[item_id] for item_id in action.chosen_initiatives if item_id in visible_backlog]
    spent_budget = sum(item.cost for item in chosen_items)
    spent_capacity = sum(max(1.0, item.cost / 2.0) for item in chosen_items)
    invalid = spent_budget > episode["budget_remaining"] + 1e-9 or spent_capacity > episode["capacity_remaining"] + 1e-9

    step_events = episode["events"].get(current_step, [])
    previous_dashboard = episode["dashboard"].model_copy(deep=True)
    scheduled_effects: list[TemporalEffectRecord] = []
    immediate_channels = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    governance_flags: list[GovernanceConstraint] = []
    policy_alerts: list[str] = []

    if not invalid:
        episode["budget_remaining"] -= spent_budget
        episode["capacity_remaining"] -= spent_capacity
        for item in chosen_items:
            episode["completed_ids"].add(item.item_id)

        initiative_immediate, initiative_scheduled = _schedule_initiative_effects(
            episode,
            hidden_goal,
            chosen_items,
            current_step,
            start_date,
            decision_id,
        )
        for channel, value in initiative_immediate.items():
            immediate_channels[channel] += value
        scheduled_effects.extend(initiative_scheduled)

        for effect in (_message_effect(action), _support_immediate_effect(action)):
            for channel, value in effect.items():
                immediate_channels[channel] += float(value)

        pricing_immediate, pricing_scheduled = _schedule_pricing_effect(
            episode,
            hidden_goal,
            action,
            current_step,
            start_date,
            decision_id,
        )
        for channel, value in pricing_immediate.items():
            immediate_channels[channel] += value
        scheduled_effects.extend(pricing_scheduled)
        scheduled_effects.extend(_schedule_support_effect(action, current_step, start_date, decision_id))

        governance_flags = _governance_flags(episode, action)
        if governance_flags:
            episode["policy_violations"] += len(governance_flags)
            policy_alerts = [f"Governance risk: {constraint.title}." for constraint in governance_flags]
            scheduled_effects.extend(
                _schedule_governance_effects(governance_flags, episode, current_step, start_date, decision_id)
            )

        if not enable_delayed_effects:
            immediate_from_scheduled = _aggregate_effect_channels(scheduled_effects)
            for channel, value in immediate_from_scheduled.items():
                immediate_channels[channel] += value
            scheduled_effects = []

        for channel in immediate_channels:
            immediate_channels[channel] += rng.uniform(-0.01, 0.01)
        episode["dashboard"] = _apply_channel_deltas(episode["dashboard"], immediate_channels)
    else:
        episode["invalid_actions"] += 1

    scheduled_effects.extend(_schedule_event_effects(step_events, current_step, start_date))
    if not enable_delayed_effects:
        event_effects = scheduled_effects[:]
        scheduled_effects = []
        for effect in event_effects:
            if effect.dashboard_deltas:
                episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], effect.dashboard_deltas)
    episode["pending_effects"].extend(scheduled_effects)
    episode["pending_effects"].sort(key=lambda effect: (effect.scheduled_for_step, effect.effect_id))

    next_step = current_step + 1
    realized_effects: list[TemporalEffectRecord] = []
    remaining_pending: list[TemporalEffectRecord] = []
    for effect in episode["pending_effects"]:
        if effect.scheduled_for_step == next_step:
            realized = effect.model_copy(
                update={
                    "realized_step": next_step,
                    "realized_date": sim_date_for_step(start_date, next_step),
                },
                deep=True,
            )
            realized_effects.append(realized)
        else:
            remaining_pending.append(effect)
    episode["pending_effects"] = remaining_pending
    _apply_pending_queue_drag(episode, next_step)

    delayed_channels = _aggregate_effect_channels(realized_effects)
    for channel in delayed_channels:
        delayed_channels[channel] += rng.uniform(-0.005, 0.005)
    if any(abs(value) > 1e-9 for value in delayed_channels.values()):
        episode["dashboard"] = _apply_channel_deltas(episode["dashboard"], delayed_channels)
    for effect in realized_effects:
        if effect.dashboard_deltas:
            episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], effect.dashboard_deltas)

    renewal_effects = _update_accounts(
        episode,
        chosen_items,
        action,
        immediate_channels,
        delayed_channels,
        governance_flags,
        realized_effects,
    )
    for effect in renewal_effects:
        if effect.dashboard_deltas:
            episode["dashboard"] = _apply_dashboard_deltas(episode["dashboard"], effect.dashboard_deltas)
    realized_effects.extend(renewal_effects)
    _update_teams(episode, chosen_items, action, invalid, governance_flags, delayed_channels)
    _update_market_context(episode, immediate_channels, delayed_channels, governance_flags)

    decision = DecisionLedgerEntry(
        decision_id=decision_id,
        step_index=current_step,
        sim_date=sim_date_for_step(start_date, current_step),
        chosen_initiatives=[item.item_id for item in chosen_items],
        messaging_action=action.messaging_action,
        pricing_change_pct=action.pricing_change_pct,
        support_policy=action.support_policy,
        rationale=action.rationale or action.rationale_summary,
        expected_channels=_aggregate_effect_channels(scheduled_effects),
        scheduled_effect_ids=[effect.effect_id for effect in scheduled_effects],
        observed_alerts=[alert for event in step_events for alert in event["alerts"]] + policy_alerts,
        governance_flags=[constraint.constraint_id for constraint in governance_flags],
    )
    episode["decision_ledger"].append(decision)
    decision_lookup = {entry.decision_id: entry for entry in episode["decision_ledger"]}
    for effect in realized_effects:
        if effect.decision_id and effect.decision_id in decision_lookup:
            decision_lookup[effect.decision_id].realized_effect_ids.append(effect.effect_id)

    bank = episode.get("memory_bank")
    step_sim_date = sim_date_for_step(start_date, current_step)
    next_sim_date = sim_date_for_step(start_date, next_step)
    if bank is not None:
        for event in step_events:
            record_event(
                bank,
                step_index=current_step,
                sim_date=step_sim_date,
                event_id=event["event_id"],
                summary=event["summary"],
                alerts=list(event.get("alerts", [])),
            )
        decision_summary = (
            f"Day {current_step + 1}: chose {', '.join(item.item_id for item in chosen_items) or 'no initiatives'}, "
            f"messaging={action.messaging_action.value if action.messaging_action else 'none'}, "
            f"pricing={action.pricing_change_pct if action.pricing_change_pct is not None else 0.0:+.2f}, "
            f"support={action.support_policy.value if action.support_policy else 'balanced_triage'}."
        )
        record_decision(
            bank,
            step_index=current_step,
            sim_date=step_sim_date,
            decision_id=decision_id,
            summary=decision_summary,
            chosen_initiatives=[item.item_id for item in chosen_items],
            governance_flags=[constraint.constraint_id for constraint in governance_flags],
            company_id=episode["company_profile"].company_id,
        )
        for effect in realized_effects:
            record_effect(
                bank,
                step_index=next_step,
                sim_date=next_sim_date,
                effect_id=effect.effect_id,
                summary=effect.summary,
                account_ids=list(effect.affected_account_ids),
                team_ids=list(effect.affected_team_ids),
                source_id=effect.source_id,
                tags=[effect.source_type],
            )
        focus_accounts = [
            account
            for account in episode["accounts"][:4]
            if account.renewal_window_days <= 45 or account.account_id in _beneficiary_account_ids(episode, chosen_items, realized_effects)
        ][:3]
        for account in focus_accounts:
            record_account_update(
                bank,
                step_index=next_step,
                sim_date=next_sim_date,
                account=account,
                summary=(
                    f"{account.company_name} now has renewal window {account.renewal_window_days} days "
                    f"and relationship health {account.relationship_health:.2f}."
                ),
                tags=["renewal_watch"] if account.renewal_window_days <= 30 else ["account_health"],
                conflict_group="pricing_vs_trust" if account.renewal_window_days <= 30 else None,
            )
        for team in episode["teams"]:
            record_team_update(
                bank,
                step_index=next_step,
                sim_date=next_sim_date,
                team=team,
                summary=(
                    f"{team.function.title()} team now shows capacity {team.capacity:.2f}, "
                    f"burnout {team.burnout_risk:.2f}, reliability {team.execution_reliability:.2f}."
                ),
            )
        if any(flag.constraint_id == "pricing_guardrail" for flag in governance_flags) or any(
            "pricing" in effect.summary.lower() for effect in realized_effects
        ):
            mark_conflict_resolved(
                bank,
                "pricing_vs_trust",
                "Observed pricing-sensitive fallout clarified the pricing-vs-trust tradeoff.",
                next_step,
                next_sim_date,
            )
        if action.support_policy == SupportPolicy.AUTOMATION_FIRST or any(
            "automation" in effect.summary.lower() or "incident" in effect.summary.lower()
            for effect in realized_effects
        ):
            mark_conflict_resolved(
                bank,
                "speed_vs_reliability",
                "Operational evidence resolved part of the speed-vs-reliability debate.",
                next_step,
                next_sim_date,
            )
        apply_agent_writes(bank, action.memory_writes, current_step, step_sim_date)

    renewal_alerts = [effect.summary for effect in renewal_effects]
    episode["alerts"] = [alert for event in step_events for alert in event["alerts"]] + policy_alerts + renewal_alerts
    episode["realized_effects"] = realized_effects
    episode["step_index"] = next_step
    if bank is not None:
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
    episode["action_history"].append(action)
    episode["goal_history"].append(current_goal_name(hidden_goal, episode["step_index"]))
    episode["latent_utility_history"].append(
        compute_utility(
            episode["dashboard"].to_metric_vector(),
            hidden_goal,
            step_index=episode["step_index"],
        )
    )

    return {
        "invalid": invalid,
        "previous_dashboard": previous_dashboard,
        "new_dashboard": episode["dashboard"].model_copy(deep=True),
        "spent_budget": spent_budget,
        "remaining_budget": episode["budget_remaining"],
        "step_events": step_events,
        "scheduled_effects": scheduled_effects,
        "realized_effects": realized_effects,
        "decision_id": decision_id,
        "governance_flags": [constraint.constraint_id for constraint in governance_flags],
        "policy_alerts": policy_alerts,
    }
