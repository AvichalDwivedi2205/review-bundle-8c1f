"""Task 4: capital allocation under uncertainty."""

from __future__ import annotations

import random

from latentgoalops.models import CustomerAccount, DashboardState, InitiativeItem, StakeholderPersona
from latentgoalops.server.hidden_goals import HiddenGoal
from latentgoalops.server.objective_utils import structured_item_multiplier


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


def _capital_state_signals(accounts: list[CustomerAccount], market_context, dashboard: DashboardState) -> dict[str, float]:
    commercial_accounts = [
        account for account in accounts if account.segment in {"mid_market", "enterprise", "strategic"}
    ] or list(accounts)
    near_renewal_accounts = [account for account in commercial_accounts if account.renewal_window_days <= 35]
    renewal_pressure = len(near_renewal_accounts) / max(len(commercial_accounts), 1)
    trust_gap = sum(max(0.0, 0.68 - account.relationship_health) for account in commercial_accounts) / max(
        len(commercial_accounts),
        1,
    )
    expansion_pressure = sum(account.expansion_potential for account in commercial_accounts) / max(
        len(commercial_accounts),
        1,
    )
    support_stress = max(0.0, min(1.0, (dashboard.support_ticket_volume - 118.0) / 120.0))
    margin_gap = max(0.0, min(1.0, (market_context.gross_margin_target - dashboard.ops_margin) / 0.24))
    runway_tightness = max(0.0, min(1.0, (11.0 - float(market_context.cash_runway_months)) / 7.0))
    pipeline_slack = max(0.0, min(1.0, (0.60 - market_context.sales_pipeline_health) / 0.35))
    return {
        "renewal_pressure": renewal_pressure,
        "trust_gap": trust_gap,
        "expansion_pressure": expansion_pressure,
        "support_stress": support_stress,
        "margin_gap": margin_gap,
        "runway_tightness": runway_tightness,
        "pipeline_slack": pipeline_slack,
    }


def _capital_context_note(kind: str, signals: dict[str, float]) -> str:
    if kind == "growth":
        if signals["pipeline_slack"] >= 0.30 and signals["support_stress"] < 0.35:
            return "pipeline coverage feels softer than leadership wants and support load is still manageable"
        if signals["support_stress"] >= 0.40:
            return "support strain is already visible, so activation bets need operational cover to pay back cleanly"
        return "new demand needs to compound into healthier activation rather than a noisy acquisition spike"
    if kind == "retention":
        if signals["renewal_pressure"] >= 0.30 or signals["trust_gap"] >= 0.20:
            return "renewal risk and trust recovery are visible in the room"
        return "leadership is sensitive to whether coverage and trust can hold through the next renewal cycle"
    if kind == "revenue":
        if signals["expansion_pressure"] >= 0.45:
            return "there are credible expansion-ready accounts if commercial friction gets removed"
        if signals["trust_gap"] >= 0.20:
            return "commercial upside exists, but it only pays back if trust is not already wobbling"
        return "pricing and packaging work needs visible expansion potential to justify the spend"
    if signals["margin_gap"] >= 0.20 or signals["runway_tightness"] >= 0.20:
        return "margin and runway pressure are becoming hard to ignore"
    return "operating leverage matters because leadership wants cleaner execution and lower cost-to-serve"


def _capital_impact_summary(kind: str, signals: dict[str, float]) -> str:
    labels = {
        "growth": "new-logo momentum",
        "retention": "account stability",
        "revenue": "commercial capture",
        "efficiency": "operating leverage",
    }
    tradeoff = {
        "growth": "operating leverage",
        "retention": "new-logo momentum",
        "revenue": "account stability",
        "efficiency": "new-logo momentum",
    }[kind]
    return (
        f"Best for {labels[kind]} when {_capital_context_note(kind, signals)}. "
        f"Visible trade-offs on {tradeoff} if the portfolio gets too scattered."
    )


def _dashboard_baseline() -> DashboardState:
    return DashboardState(
        dau=25500.0,
        mau=112000.0,
        d7_retention=0.45,
        d30_retention=0.28,
        mrr=59000.0,
        arpu=108.0,
        cac=69.0,
        churn_rate=0.039,
        ops_margin=0.36,
        infra_cost_per_unit=1.84,
        support_ticket_volume=128,
    )


def _program_bank(split: str) -> list[dict]:
    core = [
        {
            "item_id": "plg_onboarding_fund",
            "title": "PLG Onboarding Fund",
            "kind": "growth",
            "description": "Increase activation by funding guided onboarding improvements and lifecycle nudges.",
            "deltas": {"growth": 0.030, "retention": 0.010, "revenue": 0.006, "efficiency": -0.004},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["pricing_packaging_lab"],
            "synergy": ["referral_acceleration_pool"],
        },
        {
            "item_id": "referral_acceleration_pool",
            "title": "Referral Acceleration Pool",
            "kind": "growth",
            "description": "Layer referral incentives and ambassador loops on top of a stronger activation funnel.",
            "deltas": {"growth": 0.034, "retention": 0.004, "revenue": 0.008, "efficiency": -0.006},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": ["plg_onboarding_fund"],
            "conflicts": [],
            "synergy": ["plg_onboarding_fund"],
        },
        {
            "item_id": "renewal_rescue_pod",
            "title": "Renewal Rescue Pod",
            "kind": "retention",
            "description": "Fund executive outreach and success coverage for renewal-risk accounts.",
            "deltas": {"growth": 0.004, "retention": 0.034, "revenue": 0.012, "efficiency": -0.006},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["reliability_hardening_reserve"],
        },
        {
            "item_id": "reliability_hardening_reserve",
            "title": "Reliability Hardening Reserve",
            "kind": "efficiency",
            "description": "Reduce incident risk and noisy escalations through focused reliability work.",
            "deltas": {"growth": 0.002, "retention": 0.018, "revenue": 0.006, "efficiency": 0.030},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["renewal_rescue_pod"],
        },
        {
            "item_id": "pricing_packaging_lab",
            "title": "Pricing Packaging Lab",
            "kind": "revenue",
            "description": "Fund packaging analysis and monetization experiments for higher-value segments.",
            "deltas": {"growth": -0.004, "retention": -0.006, "revenue": 0.038, "efficiency": 0.006},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["plg_onboarding_fund"],
            "synergy": ["sales_enablement_grants"],
        },
        {
            "item_id": "sales_enablement_grants",
            "title": "Sales Enablement Grants",
            "kind": "revenue",
            "description": "Equip account teams with ROI assets and pilot support for expansion conversations.",
            "deltas": {"growth": 0.010, "retention": 0.008, "revenue": 0.030, "efficiency": 0.000},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["pricing_packaging_lab"],
        },
        {
            "item_id": "workflow_automation_pool",
            "title": "Workflow Automation Pool",
            "kind": "efficiency",
            "description": "Automate repetitive support and finance operations to protect margin and team capacity.",
            "deltas": {"growth": 0.000, "retention": 0.012, "revenue": 0.008, "efficiency": 0.034},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": [],
        },
    ]
    heldout = [
        {
            "item_id": "partner_marketplace_seed",
            "title": "Partner Marketplace Seed",
            "kind": "growth",
            "description": "Seed co-marketing and channel experiments through a partner marketplace program.",
            "deltas": {"growth": 0.032, "retention": 0.006, "revenue": 0.010, "efficiency": -0.004},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["enterprise_procurement_desk"],
            "synergy": ["developer_conversion_grants"],
        },
        {
            "item_id": "developer_conversion_grants",
            "title": "Developer Conversion Grants",
            "kind": "growth",
            "description": "Fund migration tooling and trial assistance for developer-led accounts.",
            "deltas": {"growth": 0.030, "retention": 0.010, "revenue": 0.006, "efficiency": -0.005},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["partner_marketplace_seed"],
        },
        {
            "item_id": "security_assurance_pool",
            "title": "Security Assurance Pool",
            "kind": "retention",
            "description": "Fund security reviews, trust recovery, and hands-on assurance for sensitive accounts.",
            "deltas": {"growth": 0.002, "retention": 0.036, "revenue": 0.010, "efficiency": -0.003},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["platform_resilience_reserve"],
        },
        {
            "item_id": "enterprise_procurement_desk",
            "title": "Enterprise Procurement Desk",
            "kind": "revenue",
            "description": "Fund commercial packaging support and procurement acceleration for expansion-ready accounts.",
            "deltas": {"growth": 0.004, "retention": 0.006, "revenue": 0.040, "efficiency": 0.004},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": ["partner_marketplace_seed"],
            "synergy": ["seat_tier_packaging_lab"],
        },
        {
            "item_id": "seat_tier_packaging_lab",
            "title": "Seat Tier Packaging Lab",
            "kind": "revenue",
            "description": "Support pricing ops and packaging tests for seat-tier monetization.",
            "deltas": {"growth": -0.004, "retention": -0.004, "revenue": 0.036, "efficiency": 0.008},
            "allocation_max": 3.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["enterprise_procurement_desk"],
        },
        {
            "item_id": "platform_resilience_reserve",
            "title": "Platform Resilience Reserve",
            "kind": "efficiency",
            "description": "Fund resiliency and ops hardening to lower cost-to-serve and surprise incidents.",
            "deltas": {"growth": 0.000, "retention": 0.016, "revenue": 0.006, "efficiency": 0.036},
            "allocation_max": 4.0,
            "saturation_point": 2.0,
            "requires": [],
            "conflicts": [],
            "synergy": ["security_assurance_pool"],
        },
        {
            "item_id": "backoffice_standardization_pool",
            "title": "Backoffice Standardization Pool",
            "kind": "efficiency",
            "description": "Standardize support and finance workflows to reduce overhead.",
            "deltas": {"growth": 0.000, "retention": 0.010, "revenue": 0.008, "efficiency": 0.034},
            "allocation_max": 4.0,
            "saturation_point": 3.0,
            "requires": [],
            "conflicts": [],
            "synergy": [],
        },
    ]
    return heldout if split == "heldout" else core


def _stakeholder_note(
    persona: StakeholderPersona,
    accounts: list[CustomerAccount],
    market_context,
) -> str:
    near_term = sorted(accounts, key=lambda account: account.renewal_window_days)[:2]
    expansion = sorted(accounts, key=lambda account: account.expansion_potential, reverse=True)[:2]
    names = ", ".join(account.company_name for account in near_term) or "renewal-sensitive accounts"
    expansion_names = ", ".join(account.company_name for account in expansion) or "expansion-ready accounts"
    overlay = (
        f"The room keeps circling the same pressures: fragile renewals around {names}, "
        f"commercial upside in {expansion_names}, and runway discipline with {market_context.cash_runway_months} months left."
    )
    role_prefix = {
        "CEO": "wants one crisp capital story rather than a scattered budget.",
        "CFO": "is focused on payback discipline and how quickly each dollar turns into visible progress.",
        "CTO": "wants fewer fragmented bets and more programs that the team can execute cleanly.",
        "Head of CS": "keeps bringing the discussion back to renewals, trust, and support load.",
        "Growth Lead": "cares about whether the budget compounds into a durable demand engine.",
    }
    return f"{persona.name} ({persona.role}) {role_prefix.get(persona.role, 'wants a focused allocation plan.')} {overlay}"


def _build_programs(
    rng: random.Random,
    accounts: list[CustomerAccount],
    split: str,
    state_signals: dict[str, float],
) -> list[InitiativeItem]:
    items: list[InitiativeItem] = []
    for raw in _program_bank(split):
        candidate_accounts = [
            account for account in accounts if account.segment in _beneficiary_segments(str(raw["kind"]))
        ]
        linked_accounts = [
            account.account_id
            for account in rng.sample(candidate_accounts, k=min(len(candidate_accounts), rng.randint(1, 3)))
        ] if candidate_accounts else []
        risk_notes = [
            f"Capital can be deployed in 1-point increments up to {raw['allocation_max']:.0f}.",
            f"Visible returns look strongest through about {raw['saturation_point']:.0f} points before tapering.",
            _capital_context_note(str(raw["kind"]), state_signals).capitalize() + ".",
        ]
        if raw["requires"]:
            risk_notes.append("Part of the upside depends on another visible program being funded alongside it.")
        if raw["conflicts"]:
            risk_notes.append("Can create mixed strategic signals if heavily funded beside a conflicting program.")
        if any(
            account.renewal_window_days <= 30
            for account in candidate_accounts
            if account.account_id in linked_accounts
        ):
            risk_notes.append("Some beneficiary accounts are already near renewal, so rollout quality matters.")
        items.append(
            InitiativeItem(
                item_id=str(raw["item_id"]),
                title=str(raw["title"]),
                description=str(raw["description"]),
                cost=1.0,
                kind=str(raw["kind"]),
                kpi_deltas={key: float(value) for key, value in raw["deltas"].items()},
                uncertainty_band=0.10,
                stakeholder_tag="capital_committee",
                lag_steps=1,
                effect_window=2,
                delivery_note="Allocate discrete budget points rather than selecting the program all-or-nothing.",
                beneficiary_segments=_beneficiary_segments(str(raw["kind"])),
                beneficiary_account_ids=linked_accounts,
                implementation_risk=round(rng.uniform(0.08, 0.28), 3),
                policy_tags=[],
                requires_item_ids=list(raw["requires"]),
                conflicts_with_ids=list(raw["conflicts"]),
                synergy_item_ids=list(raw["synergy"]),
                risk_notes=risk_notes,
                allocation_unit=1.0,
                allocation_max=float(raw["allocation_max"]),
                saturation_point=float(raw["saturation_point"]),
                impact_summary=_capital_impact_summary(str(raw["kind"]), state_signals),
            )
        )
    return items


def _curve_units(amount: float, saturation_point: float) -> float:
    if amount <= 0.0:
        return 0.0
    if amount <= saturation_point:
        return amount
    return saturation_point + 0.55 * (amount - saturation_point)


def _allocation_value(
    budget_allocations: dict[str, float],
    backlog: list[InitiativeItem],
    weights: dict[str, float],
    hidden_goal: HiddenGoal | None = None,
    company_profile=None,
    step_index: int = 0,
) -> float:
    chosen: dict[str, tuple[InitiativeItem, float]] = {}
    for item in backlog:
        raw_amount = float(budget_allocations.get(item.item_id, 0.0))
        amount = max(0.0, min(raw_amount, float(item.allocation_max or 0.0)))
        if amount > 0.0:
            chosen[item.item_id] = (item, amount)

    if not chosen:
        return 0.0

    score = 0.0
    risk_penalty = 0.0
    allocated_by_kind: dict[str, float] = {}
    for item_id, (item, amount) in chosen.items():
        effective_units = _curve_units(amount, float(item.saturation_point or amount))
        base = _initiative_value(item.kpi_deltas, weights) * effective_units
        if hidden_goal is not None:
            base *= structured_item_multiplier(
                item,
                hidden_goal,
                step_index=step_index,
                company_profile=company_profile,
            )
        if item.requires_item_ids and not any(required in chosen for required in item.requires_item_ids):
            base *= 0.60
        if item.synergy_item_ids and any(synergy in chosen for synergy in item.synergy_item_ids):
            base *= 1.10
        score += base
        risk_penalty += float(item.implementation_risk) * 0.016 * amount
        if amount > float(item.saturation_point or amount) + 1.0:
            risk_penalty += 0.018 * (amount - float(item.saturation_point or amount) - 1.0)
        allocated_by_kind[item.kind] = allocated_by_kind.get(item.kind, 0.0) + amount

    conflict_penalty = 0.0
    for item_id, (item, amount) in chosen.items():
        for conflict_id in item.conflicts_with_ids:
            if conflict_id in chosen and item_id < conflict_id:
                other_item, other_amount = chosen[conflict_id]
                conflict_penalty += 0.10 * min(
                    _initiative_value(item.kpi_deltas, weights) * amount,
                    _initiative_value(other_item.kpi_deltas, weights) * other_amount,
                )

    active_programs = len(chosen)
    spread_penalty = 0.018 * max(0, active_programs - 4)
    total_allocated = sum(amount for _, amount in chosen.values())
    focus_bonus = 0.0
    if total_allocated > 0.0:
        dominant_share = max(allocated_by_kind.values()) / total_allocated
        if dominant_share >= 0.55:
            focus_bonus = 0.03
    return max(0.0, score - risk_penalty - conflict_penalty - spread_penalty + focus_bonus)


def _stakeholder_salience(persona: StakeholderPersona, state_signals: dict[str, float]) -> float:
    by_role = {
        "CEO": 1.5 + max(
            state_signals["renewal_pressure"],
            state_signals["expansion_pressure"],
            state_signals["margin_gap"],
            state_signals["runway_tightness"],
        ),
        "CFO": 1.2 + 1.5 * state_signals["margin_gap"] + 1.4 * state_signals["runway_tightness"],
        "CTO": 1.1 + 1.2 * state_signals["support_stress"] + 1.0 * state_signals["margin_gap"],
        "Head of CS": 1.1 + 1.5 * state_signals["renewal_pressure"] + 1.3 * state_signals["trust_gap"],
        "Growth Lead": 1.1 + 1.5 * state_signals["pipeline_slack"] + 0.6 * state_signals["expansion_pressure"],
    }
    return (
        by_role.get(persona.role, 1.0)
        + persona.political_power * 0.5
        + persona.credibility * 0.4
    )


def solve_oracle_allocations(
    backlog: list[InitiativeItem],
    budget: int,
    weights: dict[str, float],
    hidden_goal: HiddenGoal | None = None,
    company_profile=None,
    step_index: int = 0,
) -> tuple[dict[str, float], float]:
    """Exhaustive integer search over visible allocation programs."""
    best_allocations: dict[str, float] = {}
    best_value = 0.0
    allocation_limits = [int(round(float(item.allocation_max or 0.0))) for item in backlog]

    def search(index: int, remaining: int, current: dict[str, float]) -> None:
        nonlocal best_allocations, best_value
        if index >= len(backlog):
            value = _allocation_value(current, backlog, weights, hidden_goal, company_profile, step_index)
            if value > best_value:
                best_value = value
                best_allocations = {key: float(value) for key, value in current.items() if value > 0.0}
            return

        item = backlog[index]
        limit = min(allocation_limits[index], remaining)
        for amount in range(limit, -1, -1):
            if amount > 0:
                current[item.item_id] = float(amount)
            else:
                current.pop(item.item_id, None)
            search(index + 1, remaining - amount, current)
        current.pop(item.item_id, None)

    search(0, int(budget), {})
    return best_allocations, best_value


def random_baseline_value(
    rng: random.Random,
    backlog: list[InitiativeItem],
    budget: int,
    weights: dict[str, float],
    hidden_goal: HiddenGoal | None = None,
    company_profile=None,
    step_index: int = 0,
    samples: int = 160,
) -> float:
    """Estimate a reproducible random-allocation baseline."""
    values: list[float] = []
    for _ in range(samples):
        remaining = int(budget)
        allocations: dict[str, float] = {}
        shuffled = backlog[:]
        rng.shuffle(shuffled)
        for item in shuffled:
            if remaining <= 0:
                break
            upper = min(remaining, int(round(float(item.allocation_max or 0.0))))
            amount = rng.randint(0, upper)
            if amount > 0:
                allocations[item.item_id] = float(amount)
                remaining -= amount
        values.append(_allocation_value(allocations, backlog, weights, hidden_goal, company_profile, step_index))
    return sum(values) / max(len(values), 1)


def build_task4_episode(
    rng: random.Random,
    hidden_goal: HiddenGoal,
    budget: int,
    world: dict,
    split: str = "core",
) -> dict:
    """Create the capital-allocation episode."""
    accounts: list[CustomerAccount] = world["accounts"]
    stakeholders: list[StakeholderPersona] = world["stakeholders"]
    dashboard = _dashboard_baseline()
    state_signals = _capital_state_signals(accounts, world["market_context"], dashboard)
    backlog = _build_programs(rng, accounts, split, state_signals)
    oracle_allocations, oracle_value = solve_oracle_allocations(
        backlog,
        budget,
        hidden_goal.weights,
        hidden_goal,
        world["company_profile"],
    )
    random_value = random_baseline_value(
        random.Random(rng.randint(0, 10_000_000)),
        backlog,
        budget,
        hidden_goal.weights,
        hidden_goal,
        world["company_profile"],
    )
    visible_stakeholders = sorted(
        stakeholders[:4],
        key=lambda persona: _stakeholder_salience(persona, state_signals),
        reverse=True,
    )
    stakeholder_notes = [
        _stakeholder_note(persona, accounts, world["market_context"])
        for persona in visible_stakeholders
    ]
    return {
        "company_profile": world["company_profile"],
        "hidden_goal": hidden_goal,
        "dashboard": dashboard,
        "backlog": backlog,
        "accounts": accounts[:6],
        "stakeholders": visible_stakeholders,
        "teams": world["teams"],
        "market_context": world["market_context"],
        "governance_constraints": world["governance_constraints"],
        "sprint_budget": float(budget),
        "stakeholder_notes": stakeholder_notes,
        "state_signals": state_signals,
        "oracle_allocations": oracle_allocations,
        "oracle_value": oracle_value,
        "random_value": random_value,
        "task_summary": (
            "Allocate discrete budget points across visible operating programs. "
            "Returns taper after saturation, some programs interact, and leadership only indirectly reveals which business outcome matters most."
        ),
    }


def allocation_value(
    budget_allocations: dict[str, float],
    backlog: list[InitiativeItem],
    weights: dict[str, float],
    hidden_goal: HiddenGoal | None = None,
    company_profile=None,
    step_index: int = 0,
) -> float:
    """Score an allocation plan under the hidden utility weights."""
    return _allocation_value(budget_allocations, backlog, weights, hidden_goal, company_profile, step_index)
