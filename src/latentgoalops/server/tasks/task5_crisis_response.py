"""Task 5: one-shot crisis response package."""

from __future__ import annotations

import itertools
import random

from latentgoalops.models import (
    CustomerAccount,
    DashboardState,
    InboxItem,
    InitiativeItem,
    LatentGoalOpsAction,
    MessagingAction,
    SupportPolicy,
    TaskId,
)
from latentgoalops.server.hidden_goals import HiddenGoal
from latentgoalops.server.objective_utils import structured_item_multiplier
from latentgoalops.server.tasks.template_bank import load_initiative_effects


def _initiative_value(kpi_deltas: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(kpi_deltas.get(channel, 0.0)) * float(weights.get(channel, 0.0)) for channel in weights)


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
    if "pricing" in name or "packaging" in name or "procurement" in name:
        tags.append("pricing_guardrail")
    if kind == "efficiency":
        tags.append("margin_guardrail")
    if "support" in name or "incident" in name or "reliability" in name:
        tags.append("sla_guardrail")
    return tags


def _initial_dashboard() -> DashboardState:
    return DashboardState(
        dau=17600.0,
        mau=79800.0,
        d7_retention=0.40,
        d30_retention=0.23,
        mrr=40200.0,
        arpu=94.0,
        cac=70.0,
        churn_rate=0.051,
        ops_margin=0.31,
        infra_cost_per_unit=2.18,
        support_ticket_volume=164,
    )


def _annotate_structure(items: list[InitiativeItem], accounts_by_id: dict[str, CustomerAccount]) -> None:
    by_id = {item.item_id: item for item in items}
    pairings = [
        ("improve_onboarding", "launch_referral_loop"),
        ("launch_admin_analytics", "ship_usage_pricing"),
        ("security_assurance_pool", "platform_resilience_reserve"),
        ("enterprise_procurement_desk", "seat_tier_packaging_lab"),
    ]
    conflicts = [
        ("launch_referral_loop", "ship_usage_pricing"),
        ("partner_marketplace_seed", "enterprise_procurement_desk"),
    ]
    for left_id, right_id in pairings:
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        if left and right:
            left.synergy_item_ids = sorted(set(left.synergy_item_ids + [right.item_id]))
            right.synergy_item_ids = sorted(set(right.synergy_item_ids + [left.item_id]))
    for left_id, right_id in conflicts:
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        if left and right:
            left.conflicts_with_ids = sorted(set(left.conflicts_with_ids + [right.item_id]))
            right.conflicts_with_ids = sorted(set(right.conflicts_with_ids + [left.item_id]))

    for item in items:
        linked_accounts = [
            accounts_by_id[account_id]
            for account_id in item.beneficiary_account_ids
            if account_id in accounts_by_id
        ]
        if any(account.renewal_window_days <= 30 for account in linked_accounts):
            item.risk_notes.append("Several linked accounts are already close to renewal.")
        if any(account.segment in {"enterprise", "strategic"} for account in linked_accounts):
            item.risk_notes.append("Rollout quality matters because much of the impact lands on high-touch accounts.")
        if item.conflicts_with_ids:
            item.risk_notes.append("Can create mixed external signals if paired with a conflicting move.")
        if item.synergy_item_ids:
            item.risk_notes.append("Visible upside improves if paired with a linked initiative.")


def _load_response_initiatives(accounts: list[CustomerAccount], rng: random.Random, split: str) -> list[InitiativeItem]:
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
                description=f"Immediate response lever to {name.replace('_', ' ')}.",
                cost=float(template["cost"]),
                kind=kind,
                kpi_deltas={key: float(value) for key, value in template["deltas"].items()},
                uncertainty_band=0.10,
                stakeholder_tag=str(template.get("stakeholder_tag", "war_room")),
                lag_steps=1,
                effect_window=1,
                delivery_note=str(template.get("delivery_note", "") or ""),
                beneficiary_segments=beneficiary_segments,
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.10, 0.35), 3),
                policy_tags=_policy_tags(name, kind),
                risk_notes=[],
            )
        )
    _annotate_structure(items, {account.account_id: account for account in accounts})
    rng.shuffle(items)
    return items[:6]


def _pressure_lines(accounts: list[CustomerAccount], market_context, dashboard: DashboardState) -> list[str]:
    renewal_names = ", ".join(
        account.company_name for account in sorted(accounts, key=lambda account: account.renewal_window_days)[:2]
    ) or "top accounts"
    return [
        (
            f"Everyone agrees the package has to address trust around {renewal_names}, "
            f"but board pressure at {market_context.board_pressure_level:.2f} is making commercial tradeoffs harder."
        ),
        (
            f"Runway sits at {market_context.cash_runway_months} months, ops margin is {dashboard.ops_margin:.2f}, "
            "and nobody wants a response that creates more organizational drag than it removes."
        ),
    ]


def _select_focus_account(accounts: list[CustomerAccount]) -> CustomerAccount:
    return min(
        accounts,
        key=lambda account: (
            account.renewal_window_days,
            -account.strategic_importance,
            -account.annual_contract_value,
            account.relationship_health,
        ),
    )


def _select_exec_sender(rng: random.Random, stakeholders, accounts: list[CustomerAccount], market_context, dashboard: DashboardState) -> str:
    renewal_pressure = sum(1 for account in accounts if account.renewal_window_days <= 30) / max(len(accounts), 1)
    trust_gap = sum(max(0.0, 0.66 - account.relationship_health) for account in accounts) / max(len(accounts), 1)
    support_stress = max(0.0, min(1.0, (dashboard.support_ticket_volume - 150.0) / 120.0))
    margin_gap = max(0.0, min(1.0, (market_context.gross_margin_target - dashboard.ops_margin) / 0.24))
    runway_tightness = max(0.0, min(1.0, (11.0 - float(market_context.cash_runway_months)) / 7.0))
    growth_pressure = max(0.0, 0.60 - float(market_context.sales_pipeline_health))
    role_weights = {
        "CEO": 1.5 + max(renewal_pressure, support_stress, runway_tightness, growth_pressure),
        "CFO": 1.2 + 1.4 * float(market_context.board_pressure_level) + 1.2 * runway_tightness + 1.0 * margin_gap,
        "CTO": 1.1 + 1.3 * support_stress + 1.1 * margin_gap,
        "Head of CS": 1.1 + 1.5 * renewal_pressure + 1.4 * trust_gap,
        "Growth Lead": 1.1 + 1.4 * growth_pressure + 0.4 * float(market_context.competition_intensity),
    }
    weights = [
        role_weights.get(stakeholder.role, 1.0)
        + stakeholder.political_power * 0.5
        + stakeholder.credibility * 0.4
        + rng.uniform(0.0, 0.8)
        for stakeholder in stakeholders
    ]
    chosen = rng.choices(stakeholders, weights=weights, k=1)[0]
    return chosen.role


def _exec_message(role: str, signal_lines: list[str], market_context, dashboard: DashboardState) -> str:
    if role == "CFO":
        return (
            f"Board pressure is {market_context.board_pressure_level:.2f} with runway at {market_context.cash_runway_months} months, "
            f"so the package needs a defendable downside profile. {signal_lines[1]}"
        )
    if role == "CTO":
        return (
            f"Ops margin is {dashboard.ops_margin:.2f}, support volume is {dashboard.support_ticket_volume}, "
            "and the response cannot create more execution drag than it removes. "
            f"{signal_lines[1]}"
        )
    if role == "Head of CS":
        return f"{signal_lines[0]} Rollout quality matters because trust-sensitive accounts are already watching closely."
    if role == "Growth Lead":
        return (
            f"{signal_lines[0]} The response also has to preserve enough market momentum that the company does not freeze itself."
        )
    return f"{signal_lines[0]} {signal_lines[1]}"


def _build_inbox(
    rng: random.Random,
    stakeholders,
    accounts: list[CustomerAccount],
    market_context,
    dashboard: DashboardState,
) -> list[InboxItem]:
    hot_accounts = sorted(
        accounts,
        key=lambda account: (
            account.renewal_window_days <= 30,
            account.strategic_importance,
            account.annual_contract_value,
        ),
        reverse=True,
    )[:2]
    signal_lines = _pressure_lines(accounts, market_context, dashboard)
    focus_account = _select_focus_account(accounts)
    exec_sender = _select_exec_sender(rng, stakeholders, accounts, market_context, dashboard)
    items = [
        InboxItem(
            item_id="crisis_msg_0",
            sender=exec_sender,
            text=_exec_message(exec_sender, signal_lines, market_context, dashboard),
            metadata={"importance": "high"},
        ),
        InboxItem(
            item_id="crisis_msg_1",
            sender=focus_account.company_name if hot_accounts else "customer",
            text=(
                f"{focus_account.company_name if hot_accounts else 'A strategic account'} is inside a "
                f"{focus_account.renewal_window_days if hot_accounts else 18}-day decision window with "
                f"{focus_account.annual_contract_value:,.0f} in ARR and relationship health at {focus_account.relationship_health:.2f}."
            ),
            metadata={"importance": "high"},
        ),
        InboxItem(
            item_id="crisis_msg_2",
            sender="CEO" if exec_sender != "CEO" else "CFO",
            text=(
                f"{signal_lines[0]} {signal_lines[1]} The package has to stay coherent enough that leadership can explain it in one meeting."
            ),
            metadata={"importance": "medium"},
        ),
    ]
    return items


def _narrative(accounts: list[CustomerAccount], market_context, dashboard: DashboardState) -> str:
    signals = _pressure_lines(accounts, market_context, dashboard)
    top_account = max(accounts, key=lambda account: account.annual_contract_value)
    return (
        f"The company has one operating cycle to respond to a visible crisis. DAU is {dashboard.dau:.0f}, MRR is {dashboard.mrr:.0f}, "
        f"ops margin is {dashboard.ops_margin:.2f}, and support volume is {dashboard.support_ticket_volume}. "
        f"{top_account.company_name} remains strategically important, while several enterprise accounts are near renewal. "
        f"{signals[0]} {signals[1]} The response package needs to balance customer trust, revenue risk, and execution discipline in one shot."
    )


def _message_effect(messaging_action: MessagingAction | None) -> dict[str, float]:
    mapping = {
        MessagingAction.GROWTH_PUSH: {"growth": 0.030, "retention": -0.004},
        MessagingAction.RETENTION_CAMPAIGN: {"retention": 0.032, "growth": -0.002},
        MessagingAction.REVENUE_UPSELL: {"revenue": 0.032, "retention": -0.008},
        MessagingAction.COST_COMMS: {"efficiency": 0.030, "growth": -0.006},
        MessagingAction.NONE: {},
        None: {},
    }
    return mapping[messaging_action]


def _support_effect(support_policy: SupportPolicy | None) -> dict[str, float]:
    mapping = {
        SupportPolicy.PREMIUM_SLA: {"retention": 0.040, "efficiency": -0.015},
        SupportPolicy.BALANCED_TRIAGE: {"retention": 0.020, "efficiency": 0.010},
        SupportPolicy.AUTOMATION_FIRST: {"efficiency": 0.040, "retention": -0.020},
        SupportPolicy.INCIDENT_SWARM: {"retention": 0.032, "efficiency": -0.008},
        None: {},
    }
    return mapping[support_policy]


def _pricing_effect(pricing_change_pct: float | None) -> dict[str, float]:
    price = float(pricing_change_pct or 0.0)
    if price > 0.0:
        return {
            "growth": -0.010 * min(price / 0.06, 1.0),
            "retention": -0.018 * min(price / 0.06, 1.0),
            "revenue": 0.040 * min(price / 0.06, 1.0),
            "efficiency": 0.004 * min(price / 0.06, 1.0),
        }
    if price < 0.0:
        discount = min(abs(price) / 0.06, 1.0)
        return {
            "growth": 0.022 * discount,
            "retention": 0.010 * discount,
            "revenue": -0.028 * discount,
            "efficiency": -0.004 * discount,
        }
    return {}


def _initiative_bundle_value(
    chosen_items: list[InitiativeItem],
    weights: dict[str, float],
    hidden_goal: HiddenGoal | None = None,
    company_profile=None,
) -> float:
    if not chosen_items:
        return 0.0
    chosen_ids = {item.item_id for item in chosen_items}
    score = 0.0
    for item in chosen_items:
        base = _initiative_value(item.kpi_deltas, weights)
        if hidden_goal is not None:
            base *= structured_item_multiplier(item, hidden_goal, step_index=0, company_profile=company_profile)
        if item.synergy_item_ids and any(synergy_id in chosen_ids for synergy_id in item.synergy_item_ids):
            base *= 1.12
        if item.conflicts_with_ids and any(conflict_id in chosen_ids for conflict_id in item.conflicts_with_ids):
            base *= 0.86
        score += base
    score -= sum(item.implementation_risk * 0.025 for item in chosen_items)
    if len({item.kind for item in chosen_items}) == 1:
        score += 0.02
    return max(0.0, score)


def evaluate_task5_action_value(
    episode: dict,
    action: LatentGoalOpsAction,
    weights: dict[str, float],
) -> tuple[float, float, list[str]]:
    visible = {item.item_id: item for item in episode["backlog"]}
    chosen_items = [visible[item_id] for item_id in action.chosen_initiatives if item_id in visible]
    value = _initiative_bundle_value(chosen_items, weights, episode.get("hidden_goal"), episode.get("company_profile"))
    for channel, delta in _message_effect(action.messaging_action).items():
        value += float(delta) * float(weights.get(channel, 0.0))
    for channel, delta in _support_effect(action.support_policy).items():
        value += float(delta) * float(weights.get(channel, 0.0))
    for channel, delta in _pricing_effect(action.pricing_change_pct).items():
        value += float(delta) * float(weights.get(channel, 0.0))

    flags: list[str] = []
    high_touch_accounts = [
        account
        for account in episode["accounts"]
        if account.support_tier == "premium"
        or (account.segment in {"enterprise", "strategic"} and account.renewal_window_days <= 30)
    ]
    if high_touch_accounts and action.support_policy == SupportPolicy.AUTOMATION_FIRST:
        flags.append("sla_guardrail")
        value -= 0.06
    if high_touch_accounts and float(action.pricing_change_pct or 0.0) > 0.03:
        flags.append("pricing_guardrail")
        value -= 0.07
    if (
        episode["dashboard"].ops_margin <= 0.32
        and action.support_policy == SupportPolicy.PREMIUM_SLA
        and any(item.kind == "growth" for item in chosen_items)
    ):
        flags.append("margin_guardrail")
        value -= 0.04

    if high_touch_accounts and (
        action.support_policy in {SupportPolicy.PREMIUM_SLA, SupportPolicy.INCIDENT_SWARM}
        or any(item.kind == "retention" for item in chosen_items)
    ):
        value += 0.02
    if episode["market_context"].board_pressure_level >= 0.68 and (
        action.messaging_action == MessagingAction.REVENUE_UPSELL
        or any(item.kind == "revenue" for item in chosen_items)
    ):
        value += 0.015

    channel_support: dict[str, float] = {}
    for item in chosen_items:
        channel_support[item.kind] = channel_support.get(item.kind, 0.0) + 1.0
    message_channel = {
        MessagingAction.GROWTH_PUSH: "growth",
        MessagingAction.RETENTION_CAMPAIGN: "retention",
        MessagingAction.REVENUE_UPSELL: "revenue",
        MessagingAction.COST_COMMS: "efficiency",
    }.get(action.messaging_action)
    if message_channel is not None:
        channel_support[message_channel] = channel_support.get(message_channel, 0.0) + 0.5
    if action.support_policy == SupportPolicy.PREMIUM_SLA:
        channel_support["retention"] = channel_support.get("retention", 0.0) + 0.4
    elif action.support_policy == SupportPolicy.AUTOMATION_FIRST:
        channel_support["efficiency"] = channel_support.get("efficiency", 0.0) + 0.4
    if channel_support:
        dominant_share = max(channel_support.values()) / sum(channel_support.values())
        value += 0.02 if dominant_share >= 0.5 else -0.02

    constraint_score = max(0.0, 1.0 - 0.35 * len(set(flags)))
    return max(0.0, value), constraint_score, sorted(set(flags))


def solve_task5_oracle_action(episode: dict, weights: dict[str, float]) -> tuple[LatentGoalOpsAction, float, float]:
    """Search over the visible mixed action space to find the best package."""
    best_action = LatentGoalOpsAction(
        task_id=episode["task_id"],
        chosen_initiatives=[],
        messaging_action=MessagingAction.NONE,
        pricing_change_pct=0.0,
        support_policy=SupportPolicy.BALANCED_TRIAGE,
        rationale="Oracle crisis package.",
    )
    best_value = -1.0
    best_constraint_score = 1.0
    backlog = episode["backlog"]
    pricing_grid = [-0.06, -0.03, 0.0, 0.03, 0.06]
    max_initiatives = int(episode["capacity_remaining"])
    budget = float(episode["budget_remaining"])

    for subset_size in range(0, min(max_initiatives, len(backlog)) + 1):
        for subset in itertools.combinations(backlog, subset_size):
            if sum(item.cost for item in subset) > budget + 1e-9:
                continue
            chosen_ids = [item.item_id for item in subset]
            for messaging in MessagingAction:
                for support in SupportPolicy:
                    for pricing in pricing_grid:
                        action = LatentGoalOpsAction(
                            task_id=episode["task_id"],
                            chosen_initiatives=chosen_ids,
                            messaging_action=messaging,
                            pricing_change_pct=pricing,
                            support_policy=support,
                            rationale="Oracle crisis package.",
                        )
                        value, constraint_score, _ = evaluate_task5_action_value(episode, action, weights)
                        composite = value + 0.12 * constraint_score
                        if composite > best_value:
                            best_action = action
                            best_value = composite
                            best_constraint_score = constraint_score
    return best_action, max(0.0, best_value), best_constraint_score


def random_baseline_value(
    rng: random.Random,
    episode: dict,
    weights: dict[str, float],
    samples: int = 160,
) -> float:
    """Estimate a reproducible random baseline for the crisis package."""
    values: list[float] = []
    visible = episode["backlog"][:]
    for _ in range(samples):
        rng.shuffle(visible)
        chosen = []
        remaining_budget = float(episode["budget_remaining"])
        for item in visible:
            if len(chosen) >= int(episode["capacity_remaining"]):
                break
            if item.cost > remaining_budget:
                continue
            if rng.random() > 0.55:
                chosen.append(item.item_id)
                remaining_budget -= item.cost
        action = LatentGoalOpsAction(
            task_id=episode["task_id"],
            chosen_initiatives=chosen,
            messaging_action=rng.choice(list(MessagingAction)),
            pricing_change_pct=rng.choice([-0.06, -0.03, 0.0, 0.03, 0.06]),
            support_policy=rng.choice(list(SupportPolicy)),
        )
        value, constraint_score, _ = evaluate_task5_action_value(episode, action, weights)
        values.append(value + 0.12 * constraint_score)
    return sum(values) / max(len(values), 1)


def build_task5_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    budget: float,
    capacity: int,
    world: dict,
    split: str = "core",
) -> dict:
    """Create the one-shot crisis-response episode."""
    accounts = [account.model_copy(deep=True) for account in world["accounts"]]
    stakeholders = [stakeholder.model_copy(deep=True) for stakeholder in world["stakeholders"]]
    teams = [team.model_copy(deep=True) for team in world["teams"]]
    market_context = world["market_context"].model_copy(deep=True)
    governance_constraints = [constraint.model_copy(deep=True) for constraint in world["governance_constraints"]]
    dashboard = _initial_dashboard()
    backlog = _load_response_initiatives(accounts, rng, split)
    narrative = _narrative(accounts, market_context, dashboard)
    inbox = _build_inbox(rng, stakeholders, accounts, market_context, dashboard)
    episode = {
        "task_id": TaskId.TASK5,
        "company_profile": world["company_profile"].model_copy(deep=True),
        "hidden_goal": hidden_goal,
        "dashboard": dashboard,
        "backlog": backlog,
        "accounts": accounts[:6],
        "stakeholders": stakeholders[:4],
        "teams": teams,
        "market_context": market_context,
        "governance_constraints": governance_constraints,
        "budget_remaining": float(budget),
        "capacity_remaining": int(capacity),
        "narrative": narrative,
        "inbox": inbox,
        "alerts": [
            "Support backlog is spiking around premium accounts.",
            "Leadership wants one coherent response package before the next customer and board touchpoint.",
        ],
        "task_summary": (
            "Choose one crisis-response package by combining visible initiatives with pricing, messaging, and support policy. "
            "The right package depends on the hidden objective, but the visible downside risks are real."
        ),
    }
    oracle_action, oracle_value, oracle_constraint_score = solve_task5_oracle_action(episode, hidden_goal.weights)
    random_value = random_baseline_value(random.Random(rng.randint(0, 10_000_000)), episode, hidden_goal.weights)
    episode.update(
        {
            "oracle_action": oracle_action,
            "oracle_value": oracle_value,
            "oracle_constraint_score": oracle_constraint_score,
            "random_value": random_value,
        }
    )
    return episode
