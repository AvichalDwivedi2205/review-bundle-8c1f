"""Helpers for structured latent-objective alignment."""

from __future__ import annotations

from latentgoalops.models import CompanyProfile, GovernanceStrictness, InitiativeItem, RiskPosture
from latentgoalops.server.hidden_goals import HiddenGoal, active_state


def segment_focus_alignment(item: InitiativeItem, segment_focus: str) -> float:
    """How well an item targets the active latent segment."""
    if not item.beneficiary_segments:
        return 0.0
    if segment_focus in item.beneficiary_segments:
        return 1.0
    if segment_focus in {"enterprise", "strategic"} and any(segment in {"enterprise", "strategic"} for segment in item.beneficiary_segments):
        return 0.7
    if segment_focus in {"self_serve", "smb"} and any(segment in {"self_serve", "smb"} for segment in item.beneficiary_segments):
        return 0.7
    if segment_focus == "mid_market" and "mid_market" in item.beneficiary_segments:
        return 0.85
    return 0.2


def governance_penalty(item: InitiativeItem, strictness: GovernanceStrictness) -> float:
    """Penalty for governance-sensitive items under stricter latent objectives."""
    if strictness == GovernanceStrictness.FLEXIBLE:
        return 0.0
    sensitive_tags = {"pricing_guardrail", "sla_guardrail", "margin_guardrail"}
    overlap = len(sensitive_tags.intersection(item.policy_tags))
    if overlap <= 0:
        return 0.0
    multiplier = 0.03 if strictness == GovernanceStrictness.MODERATE else 0.06
    return overlap * multiplier


def risk_penalty(item: InitiativeItem, posture: RiskPosture) -> float:
    """Penalty for execution risk under the active risk posture."""
    if posture == RiskPosture.AGGRESSIVE:
        return item.implementation_risk * 0.01
    if posture == RiskPosture.BALANCED:
        return item.implementation_risk * 0.03
    return item.implementation_risk * 0.06


def company_archetype_bonus(item: InitiativeItem, company_profile: CompanyProfile | None) -> float:
    """Bonus for items that fit the instantiated company family."""
    if company_profile is None:
        return 0.0
    family = company_profile.seed_family
    if family == "plg_analytics" and item.kind in {"growth", "revenue"}:
        return 0.02
    if family == "enterprise_security" and item.kind in {"retention", "efficiency"}:
        return 0.025
    if family == "support_automation" and item.kind in {"efficiency", "retention"}:
        return 0.025
    if family == "fintech_backoffice" and item.kind in {"revenue", "efficiency"}:
        return 0.02
    if family == "healthcare_ops" and item.kind in {"retention", "efficiency"}:
        return 0.025
    if family == "developer_api" and item.kind in {"growth", "efficiency"}:
        return 0.025
    return 0.0


def structured_item_multiplier(
    item: InitiativeItem,
    hidden_goal: HiddenGoal,
    *,
    step_index: int,
    company_profile: CompanyProfile | None = None,
) -> float:
    """Multiplier that converts four-channel value into richer objective alignment."""
    state = active_state(hidden_goal, step_index)
    segment_bonus = 0.10 * segment_focus_alignment(item, state.segment_focus)
    company_bonus = company_archetype_bonus(item, company_profile)
    total = 1.0 + segment_bonus + company_bonus
    total -= governance_penalty(item, state.governance_strictness)
    total -= risk_penalty(item, state.risk_posture)
    return max(0.4, total)
