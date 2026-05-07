"""Deterministic synthetic startup-world generation."""

from __future__ import annotations

import random

from latentgoalops.models import (
    CompanyProfile,
    CustomerAccount,
    GovernanceConstraint,
    InternalTeamState,
    MarketContext,
    StakeholderPersona,
)


COMPANY_PREFIXES = [
    "Aster",
    "Blue",
    "North",
    "Nova",
    "Clear",
    "Summit",
    "Vector",
    "Copper",
    "Horizon",
    "Pioneer",
    "Atlas",
    "Meridian",
]
COMPANY_SUFFIXES = ["Cloud", "Analytics", "Works", "Dynamics", "Health", "Logistics", "Capital", "Flow"]
REGIONS = ["us-east", "us-west", "eu", "uk", "apac"]
SEGMENTS = ["self_serve", "smb", "mid_market", "enterprise", "strategic"]
COMMUNICATION_STYLES = ["concise", "demanding", "collaborative", "skeptical", "formal"]
ADOPTION_STAGES = ["trial", "ramping", "steady", "power_user", "at_risk"]

FAMILY_TEMPLATES = {
    "plg_analytics": {
        "split": "core",
        "product_type": "self-serve product analytics workspace",
        "gtm_motion": "product-led with light sales assist",
        "pricing_model": "seat + usage hybrid",
        "compliance_regime": "soc2-lite",
        "deployment_model": "multi-tenant cloud",
        "core_user_workflow": "instrument product events and monitor funnels",
        "ideal_customer_profile": "B2B SaaS teams scaling product-led growth",
        "primary_buyer": "VP Product",
        "board_narrative": "compound activation efficiently without breaking monetization",
        "margin_profile": "moderate gross margin with infra-sensitive analytics workloads",
        "sector": "saas",
        "seed_family": "plg_analytics",
        "risk_envelope": "tolerates product bets if payback is visible within a quarter",
        "segment_weights": [3, 3, 3, 2, 1],
        "industry_choices": ["saas", "education", "retail"],
        "team_bias": {"growth": 0.12, "product": 0.08},
    },
    "enterprise_security": {
        "split": "core",
        "product_type": "enterprise security operations workflow",
        "gtm_motion": "sales-led enterprise motion",
        "pricing_model": "annual platform license with premium support",
        "compliance_regime": "soc2 + iso27001",
        "deployment_model": "hybrid cloud",
        "core_user_workflow": "triage alerts, enforce controls, and document remediation",
        "ideal_customer_profile": "regulated mid-market and enterprise security teams",
        "primary_buyer": "CISO",
        "board_narrative": "win trust-sensitive enterprise renewals without margin collapse",
        "margin_profile": "high ACV but service-heavy delivery",
        "sector": "cybersecurity",
        "seed_family": "enterprise_security",
        "risk_envelope": "prefers safer launches near renewals and compliance milestones",
        "segment_weights": [1, 2, 3, 3, 2],
        "industry_choices": ["cybersecurity", "fintech", "healthcare"],
        "team_bias": {"support": 0.10, "infra": 0.12},
    },
    "support_automation": {
        "split": "core",
        "product_type": "AI-assisted support operations suite",
        "gtm_motion": "hybrid PLG and expansion sales",
        "pricing_model": "per-agent subscription + automation overage",
        "compliance_regime": "soc2 + gdpr",
        "deployment_model": "multi-tenant cloud",
        "core_user_workflow": "route, answer, and audit support conversations",
        "ideal_customer_profile": "support and CX teams with mixed SMB-to-enterprise portfolios",
        "primary_buyer": "VP Support",
        "board_narrative": "improve efficiency without eroding trust in premium queues",
        "margin_profile": "healthy gross margin if automation quality holds",
        "sector": "saas",
        "seed_family": "support_automation",
        "risk_envelope": "accepts moderate automation risk but not enterprise trust failures",
        "segment_weights": [2, 3, 3, 2, 1],
        "industry_choices": ["saas", "retail", "logistics"],
        "team_bias": {"support": 0.15, "product": 0.05},
    },
    "fintech_backoffice": {
        "split": "core",
        "product_type": "finance and reconciliation operations platform",
        "gtm_motion": "sales-led with implementation services",
        "pricing_model": "platform subscription + transaction volume",
        "compliance_regime": "soc2 + pci + internal audit obligations",
        "deployment_model": "private cloud",
        "core_user_workflow": "reconcile ledgers, track exceptions, and close books",
        "ideal_customer_profile": "finance ops teams at regulated scale-ups and enterprises",
        "primary_buyer": "CFO",
        "board_narrative": "show commercial discipline while maintaining operational credibility",
        "margin_profile": "good software margin with expensive implementations",
        "sector": "fintech",
        "seed_family": "fintech_backoffice",
        "risk_envelope": "low tolerance for governance or pricing mistakes near strategic accounts",
        "segment_weights": [1, 2, 3, 3, 2],
        "industry_choices": ["fintech", "logistics", "saas"],
        "team_bias": {"infra": 0.08, "product": 0.07},
    },
    "healthcare_ops": {
        "split": "heldout",
        "product_type": "care operations coordination system",
        "gtm_motion": "enterprise sales with long procurement",
        "pricing_model": "annual site license",
        "compliance_regime": "hipaa + soc2",
        "deployment_model": "private cloud with compliance controls",
        "core_user_workflow": "coordinate patient operations and staff handoffs",
        "ideal_customer_profile": "healthcare providers and care operations teams",
        "primary_buyer": "COO",
        "board_narrative": "preserve trust and reliability in regulated workflows",
        "margin_profile": "service-heavy enterprise deployments",
        "sector": "healthcare",
        "seed_family": "healthcare_ops",
        "risk_envelope": "very low tolerance for outages, pricing shock, or governance sloppiness",
        "segment_weights": [0, 1, 2, 4, 3],
        "industry_choices": ["healthcare"],
        "team_bias": {"support": 0.08, "infra": 0.15},
    },
    "developer_api": {
        "split": "heldout",
        "product_type": "developer workflow API platform",
        "gtm_motion": "developer-led with enterprise expansion",
        "pricing_model": "usage-based API billing",
        "compliance_regime": "soc2 + enterprise security review",
        "deployment_model": "multi-region cloud",
        "core_user_workflow": "integrate APIs into developer and data workflows",
        "ideal_customer_profile": "platform and engineering teams scaling API consumption",
        "primary_buyer": "VP Engineering",
        "board_narrative": "grow usage responsibly while keeping infra economics credible",
        "margin_profile": "high upside but infra-cost sensitive",
        "sector": "developer_tools",
        "seed_family": "developer_api",
        "risk_envelope": "accepts experimentation but punishes reliability regressions and pricing confusion",
        "segment_weights": [3, 2, 3, 2, 1],
        "industry_choices": ["saas", "cybersecurity", "education"],
        "team_bias": {"growth": 0.08, "infra": 0.12},
    },
}


def _company_name(rng: random.Random) -> str:
    return f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)}"


def _segment_ranges(segment: str) -> tuple[int, int, int, int]:
    if segment == "self_serve":
        return 5, 30, 500, 4_000
    if segment == "smb":
        return 20, 120, 4_000, 18_000
    if segment == "mid_market":
        return 80, 350, 18_000, 65_000
    if segment == "enterprise":
        return 200, 900, 65_000, 180_000
    return 600, 2_000, 180_000, 600_000


def _sample_family(rng: random.Random, split: str) -> tuple[str, dict]:
    choices = [
        (family_id, spec)
        for family_id, spec in FAMILY_TEMPLATES.items()
        if spec["split"] == split
    ]
    if not choices:
        choices = list(FAMILY_TEMPLATES.items())
    return rng.choice(choices)


def _generate_company_profile(rng: random.Random, family_id: str, spec: dict) -> CompanyProfile:
    name = _company_name(rng)
    return CompanyProfile(
        company_id=f"company_{family_id}",
        company_name=name,
        archetype_id=family_id,
        product_type=str(spec["product_type"]),
        gtm_motion=str(spec["gtm_motion"]),
        pricing_model=str(spec["pricing_model"]),
        compliance_regime=str(spec["compliance_regime"]),
        deployment_model=str(spec["deployment_model"]),
        core_user_workflow=str(spec["core_user_workflow"]),
        ideal_customer_profile=str(spec["ideal_customer_profile"]),
        primary_buyer=str(spec["primary_buyer"]),
        board_narrative=str(spec["board_narrative"]),
        margin_profile=str(spec["margin_profile"]),
        sector=str(spec["sector"]),
        seed_family=str(spec["seed_family"]),
        risk_envelope=str(spec["risk_envelope"]),
    )


def _generate_accounts(rng: random.Random, profile: CompanyProfile, spec: dict) -> list[CustomerAccount]:
    accounts: list[CustomerAccount] = []
    segment_weights = list(spec.get("segment_weights", [2, 3, 3, 2, 1]))
    for index in range(12):
        segment = rng.choices(SEGMENTS, weights=segment_weights, k=1)[0]
        seats_low, seats_high, acv_low, acv_high = _segment_ranges(segment)
        annual_contract_value = float(rng.randint(acv_low, acv_high))
        renewal_window_days = rng.randint(7, 120)
        expansion_potential = round(rng.uniform(0.1, 1.0), 3)
        relationship_health = round(rng.uniform(0.35, 0.95), 3)
        security_bonus = 0.15 if profile.compliance_regime in {"hipaa + soc2", "soc2 + iso27001", "soc2 + pci + internal audit obligations"} else 0.0
        influence_weight = round(min(1.0, 0.22 + annual_contract_value / 600_000 + rng.uniform(0.0, 0.18)), 3)
        strategic_importance = round(min(1.0, influence_weight + (0.14 if segment in {"enterprise", "strategic"} else 0.02)), 3)
        accounts.append(
            CustomerAccount(
                account_id=f"acct_{index + 1}",
                company_name=f"{profile.company_name} Customer {index + 1}",
                segment=segment,
                industry=rng.choice(list(spec.get("industry_choices", [profile.sector]))),
                region=rng.choice(REGIONS),
                seat_count=rng.randint(seats_low, seats_high),
                annual_contract_value=annual_contract_value,
                expansion_potential=expansion_potential,
                renewal_window_days=renewal_window_days,
                payment_risk=round(rng.uniform(0.0, 0.5), 3),
                support_tier="premium" if segment in {"enterprise", "strategic"} else rng.choice(["standard", "priority"]),
                sla_level="24x7" if segment == "strategic" else rng.choice(["business_hours", "priority", "24x7"]),
                security_sensitivity=round(min(1.0, rng.uniform(0.2, 0.9) + security_bonus), 3),
                integration_complexity=round(rng.uniform(0.1, 1.0), 3),
                adoption_stage=rng.choice(ADOPTION_STAGES),
                relationship_health=relationship_health,
                champion_strength=round(rng.uniform(0.1, 1.0), 3),
                decision_maker_involved=bool(rng.random() < 0.45),
                churn_propensity=round(rng.uniform(0.05, 0.8), 3),
                communication_style=rng.choice(COMMUNICATION_STYLES),
                influence_weight=influence_weight,
                strategic_importance=strategic_importance,
            )
        )
    accounts.sort(key=lambda account: (account.strategic_importance, account.annual_contract_value), reverse=True)
    return accounts


def _generate_stakeholders(rng: random.Random, profile: CompanyProfile) -> list[StakeholderPersona]:
    role_specs = [
        ("st_1", "Mira Chen", "CEO", "portfolio_balance", ["dau", "mrr"]),
        ("st_2", "Jon Alvarez", "CFO", "margin_discipline", ["mrr", "ops_margin"]),
        ("st_3", "Priya Rao", "CTO", "platform_reliability", ["ops_margin", "infra_cost_per_unit"]),
        ("st_4", "Lena Brooks", "Head of CS", "renewal_protection", ["d30_retention", "support_ticket_volume"]),
        ("st_5", "Evan Park", "Growth Lead", "topline_expansion", ["dau", "cac"]),
    ]
    family_soft_preferences = {
        "plg_analytics": ["product-led compounding", "clean onboarding signals"],
        "enterprise_security": ["renewal safety", "trust with regulated buyers"],
        "support_automation": ["balanced automation", "premium queue quality"],
        "fintech_backoffice": ["commercial discipline", "governance-safe monetization"],
        "healthcare_ops": ["operational reliability", "regulatory readiness"],
        "developer_api": ["usage growth", "infra efficiency"],
    }
    stakeholders: list[StakeholderPersona] = []
    for persona_id, name, role, bias, metrics in role_specs:
        hard_constraints: list[str] = []
        if role == "CFO":
            hard_constraints.append("Keep decisions legible to the board narrative.")
        if role == "CTO" and "cloud" in profile.deployment_model:
            hard_constraints.append("Avoid changes that create hidden reliability debt.")
        if role == "Head of CS" and profile.compliance_regime != "soc2-lite":
            hard_constraints.append("Protect regulated or strategic renewals from sloppy rollouts.")
        stakeholders.append(
            StakeholderPersona(
                persona_id=persona_id,
                name=name,
                role=role,
                agenda_bias=bias,
                risk_tolerance=round(rng.uniform(0.2, 0.9), 3),
                time_horizon=rng.choice(["short", "medium", "long"]),
                communication_style=rng.choice(COMMUNICATION_STYLES),
                political_power=round(rng.uniform(0.4, 1.0), 3),
                credibility=round(rng.uniform(0.5, 1.0), 3),
                patience=round(rng.uniform(0.2, 0.9), 3),
                favorite_metrics=metrics,
                hard_constraints=hard_constraints,
                soft_preferences=list(family_soft_preferences.get(profile.seed_family, [])),
            )
        )
    return stakeholders


def _generate_teams(rng: random.Random, profile: CompanyProfile, spec: dict) -> list[InternalTeamState]:
    team_specs = [
        ("team_product", "product", ["onboarding", "pricing", "analytics"]),
        ("team_infra", "infra", ["latency", "reliability", "cost"]),
        ("team_support", "support", ["sla", "triage", "incident_response"]),
        ("team_growth", "growth", ["activation", "referrals", "campaigns"]),
    ]
    bias = spec.get("team_bias", {})
    teams: list[InternalTeamState] = []
    for team_id, function, specialization in team_specs:
        capacity_base = rng.uniform(0.4, 0.95) + float(bias.get(function, 0.0))
        reliability_base = rng.uniform(0.45, 0.92) + (0.04 if profile.compliance_regime != "soc2-lite" and function in {"infra", "support"} else 0.0)
        teams.append(
            InternalTeamState(
                team_id=team_id,
                function=function,
                capacity=round(min(1.0, capacity_base), 3),
                burnout_risk=round(rng.uniform(0.1, 0.8), 3),
                execution_reliability=round(min(1.0, reliability_base), 3),
                specialization=specialization,
                cross_team_friction=round(rng.uniform(0.05, 0.4), 3),
                delivery_latency_days=rng.randint(1, 4),
            )
        )
    return teams


def _generate_market_context(rng: random.Random, accounts: list[CustomerAccount], profile: CompanyProfile) -> MarketContext:
    strategic_accounts_share = sum(1 for account in accounts if account.segment in {"enterprise", "strategic"}) / max(len(accounts), 1)
    board_pressure_level = round(rng.uniform(0.3, 0.95), 3)
    compliance_bonus = 0.15 if profile.compliance_regime != "soc2-lite" else 0.0
    return MarketContext(
        funding_stage=rng.choice(["seed", "series_a", "series_b", "growth"]),
        cash_runway_months=rng.randint(8, 24),
        gross_margin_target=round(rng.uniform(0.45, 0.7), 3),
        board_pressure_level=board_pressure_level,
        competition_intensity=round(rng.uniform(0.25, 0.95), 3),
        seasonality=rng.choice(["steady", "end_of_quarter", "holiday_push", "budget_reset"]),
        compliance_exposure=round(min(1.0, rng.uniform(0.1, 0.9) + compliance_bonus), 3),
        strategic_accounts_share=round(strategic_accounts_share, 3),
        sales_pipeline_health=round(rng.uniform(0.3, 0.9), 3),
    )


def _generate_governance_constraints(
    rng: random.Random,
    accounts: list[CustomerAccount],
    market_context: MarketContext,
    profile: CompanyProfile,
) -> list[GovernanceConstraint]:
    has_strategic = any(account.segment == "strategic" for account in accounts)
    regulated = profile.compliance_regime != "soc2-lite"
    constraints = [
        GovernanceConstraint(
            constraint_id="pricing_guardrail",
            title="Pricing change guardrail",
            description="Avoid enterprise price changes above 5% when strategic renewals are within 30 days.",
            severity="high",
            affected_channels=["revenue", "retention"],
            threshold=0.05,
            penalty_hint="Higher churn and governance penalties if violated.",
        ),
        GovernanceConstraint(
            constraint_id="sla_guardrail",
            title="Strategic account SLA guardrail",
            description="Do not aggressively automate support for strategic or renewal-risk accounts.",
            severity="high" if has_strategic else "medium",
            affected_channels=["retention", "efficiency"],
            penalty_hint="Trust and renewal risk increase if violated.",
        ),
        GovernanceConstraint(
            constraint_id="margin_guardrail",
            title="Margin preservation guardrail",
            description="Do not run growth-heavy campaigns when ops margin is already below the board target.",
            severity="medium",
            affected_channels=["growth", "efficiency"],
            threshold=market_context.gross_margin_target,
            penalty_hint="Board pressure and cash efficiency worsen if violated.",
        ),
    ]
    if regulated:
        constraints.append(
            GovernanceConstraint(
                constraint_id="compliance_guardrail",
                title="Regulated workflow guardrail",
                description=f"Changes touching {profile.core_user_workflow} must remain auditable for {profile.compliance_regime}.",
                severity="high",
                affected_channels=["retention", "efficiency"],
                penalty_hint="Audit and trust risk increase if compliance-sensitive changes are pushed carelessly.",
            )
        )
    rng.shuffle(constraints)
    return constraints


def build_world(rng: random.Random, split: str = "core") -> dict:
    """Create the persistent startup world shared across tasks for one seed."""
    family_id, spec = _sample_family(rng, split)
    company_profile = _generate_company_profile(rng, family_id, spec)
    accounts = _generate_accounts(rng, company_profile, spec)
    stakeholders = _generate_stakeholders(rng, company_profile)
    teams = _generate_teams(rng, company_profile, spec)
    market_context = _generate_market_context(rng, accounts, company_profile)
    governance_constraints = _generate_governance_constraints(rng, accounts, market_context, company_profile)
    return {
        "company_profile": company_profile,
        "accounts": accounts,
        "stakeholders": stakeholders,
        "teams": teams,
        "market_context": market_context,
        "governance_constraints": governance_constraints,
    }
