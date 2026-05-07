"""Public-facing initiative descriptions and visible-only heuristic helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CHANNELS = ("growth", "retention", "revenue", "efficiency")

_GOAL_HINT_KEYWORDS = {
    "retention": (
        "renewal",
        "churn",
        "trust",
        "support backlog",
        "service",
        "customer confidence",
        "stability",
        "reliability",
        "escalation",
        "fragile account",
    ),
    "revenue": (
        "pricing",
        "upsell",
        "expansion",
        "procurement",
        "contract",
        "board",
        "commercial",
        "monetization",
        "acv",
        "pipeline quality",
    ),
    "efficiency": (
        "margin",
        "runway",
        "cost",
        "throughput",
        "automation",
        "infra",
        "latency",
        "manual work",
        "operating",
        "delivery drag",
    ),
    "growth": (
        "activation",
        "acquisition",
        "pipeline",
        "demand",
        "conversion",
        "self-serve",
        "onboarding",
        "adoption",
        "referral",
        "new logo",
    ),
}

_POSITIVE_SIGNALS = {
    "growth": (
        "cleaner top-of-funnel conversion",
        "better activation and adoption",
        "stronger demand creation",
        "more self-serve pull",
    ),
    "retention": (
        "stronger renewal posture",
        "better customer confidence",
        "less churn pressure",
        "steadier service outcomes",
    ),
    "revenue": (
        "clearer pricing leverage",
        "more expansion headroom",
        "better contract economics",
        "stronger commercial follow-through",
    ),
    "efficiency": (
        "less delivery drag",
        "better team throughput",
        "lower cost-to-serve",
        "cleaner operating handoffs",
    ),
}

_NEGATIVE_SIGNALS = {
    "growth": (
        "more dependence on a concentrated launch motion",
        "extra conversion risk during rollout",
        "slower new-user pull if execution slips",
        "a weaker acquisition story if sequencing drifts",
    ),
    "retention": (
        "more renewal fragility",
        "extra trust-sensitive rollout risk",
        "more service volatility during changeover",
        "a greater chance of customer pushback",
    ),
    "revenue": (
        "harder procurement conversations",
        "more pricing sensitivity",
        "a noisier commercialization path",
        "more contract friction if timing slips",
    ),
    "efficiency": (
        "extra operational overhead",
        "more delivery complexity",
        "a heavier support burden in the near term",
        "more coordination drag across teams",
    ),
}

_SEGMENT_LABELS = {
    "self_serve": "self-serve",
    "smb": "SMB",
    "mid_market": "mid-market",
    "enterprise": "enterprise",
    "strategic": "strategic accounts",
}

_POLICY_HINTS = {
    "pricing_guardrail": "Expect pricing and packaging scrutiny.",
    "margin_guardrail": "Expect margin review before rollout expands.",
    "sla_guardrail": "Expect service-quality review from support leadership.",
}


def _stable_index(seed_text: str, count: int) -> int:
    return sum(ord(char) for char in seed_text) % max(count, 1)


def _pick_phrase(candidates: tuple[str, ...], seed_text: str) -> str:
    return candidates[_stable_index(seed_text, len(candidates))]


def infer_goal_hint_from_evidence(evidence: str) -> str:
    """Infer a rough visible objective from public narrative evidence only."""
    lowered = evidence.lower()
    best_goal = "growth"
    best_score = -1
    for goal, keywords in _GOAL_HINT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_goal = goal
            best_score = score
    return best_goal


def build_public_impact_summary(
    *,
    item_id: str,
    title: str,
    kpi_deltas: Mapping[str, float],
    beneficiary_segments: list[str] | None = None,
    beneficiary_account_names: list[str] | None = None,
    lag_steps: int = 1,
    implementation_risk: float = 0.0,
    policy_tags: list[str] | None = None,
    delivery_note: str | None = None,
) -> str:
    """Create a varied public summary without exposing internal benchmark labels."""
    beneficiary_segments = beneficiary_segments or []
    beneficiary_account_names = beneficiary_account_names or []
    policy_tags = policy_tags or []
    positives = sorted(
        ((channel, float(value)) for channel, value in kpi_deltas.items() if float(value) > 0.0),
        key=lambda row: row[1],
        reverse=True,
    )
    negatives = sorted(
        ((channel, float(value)) for channel, value in kpi_deltas.items() if float(value) < 0.0),
        key=lambda row: row[1],
    )

    summary_parts: list[str] = []
    if positives:
        channel = positives[0][0]
        summary_parts.append(
            "The clearest visible win is "
            + _pick_phrase(_POSITIVE_SIGNALS[channel], f"{item_id}:{title}:positive")
            + "."
        )
    else:
        summary_parts.append("The visible upside is mixed and depends on execution quality.")

    if beneficiary_segments:
        segment_text = ", ".join(
            _SEGMENT_LABELS.get(segment, segment.replace("_", " "))
            for segment in beneficiary_segments[:2]
        )
        summary_parts.append(f"It looks most relevant for {segment_text}.")

    if beneficiary_account_names:
        summary_parts.append(
            "The most exposed accounts right now are "
            + ", ".join(beneficiary_account_names[:2])
            + "."
        )

    if lag_steps > 1:
        summary_parts.append("Most of the payoff lands gradually rather than in the same decision cycle.")
    else:
        summary_parts.append("The first-order impact should show up quickly if rollout stays clean.")

    if negatives:
        channel = negatives[0][0]
        summary_parts.append(
            "The main watch-out is "
            + _pick_phrase(_NEGATIVE_SIGNALS[channel], f"{item_id}:{title}:negative")
            + "."
        )

    if implementation_risk >= 0.30:
        summary_parts.append("Execution risk looks meaningfully above average.")
    elif implementation_risk <= 0.12:
        summary_parts.append("Execution risk looks comparatively contained.")

    for tag in policy_tags[:2]:
        hint = _POLICY_HINTS.get(tag)
        if hint:
            summary_parts.append(hint)

    if delivery_note:
        delivery_note = delivery_note.strip()
        if delivery_note:
            summary_parts.append(delivery_note.rstrip(".") + ".")
    return " ".join(summary_parts)


def visible_item_proxy(
    item: Mapping[str, Any],
    *,
    accounts_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Score a visible item using public attributes only."""
    proxy = {channel: 0.0 for channel in CHANNELS}

    raw_deltas = item.get("kpi_deltas") or {}
    if isinstance(raw_deltas, Mapping) and raw_deltas:
        for channel in CHANNELS:
            proxy[channel] += max(-0.08, min(0.08, float(raw_deltas.get(channel, 0.0) or 0.0)))

    kind = str(item.get("kind", "") or "")
    if kind in proxy:
        proxy[kind] += 0.02

    segments = {str(segment) for segment in item.get("beneficiary_segments", [])}
    if segments & {"self_serve", "smb"}:
        proxy["growth"] += 0.035
    if "mid_market" in segments:
        proxy["growth"] += 0.012
        proxy["revenue"] += 0.012
    if segments & {"enterprise", "strategic"}:
        proxy["retention"] += 0.020
        proxy["revenue"] += 0.025

    accounts_by_id = accounts_by_id or {}
    for account_id in item.get("beneficiary_account_ids", []):
        account = accounts_by_id.get(str(account_id))
        if not account:
            continue
        if int(account.get("renewal_window_days", 999) or 999) <= 45:
            proxy["retention"] += 0.030
        if float(account.get("relationship_health", 1.0) or 1.0) <= 0.60:
            proxy["retention"] += 0.022
        if float(account.get("churn_propensity", 0.0) or 0.0) >= 0.45:
            proxy["retention"] += 0.018
        if float(account.get("strategic_importance", 0.0) or 0.0) >= 0.70:
            proxy["retention"] += 0.016
            proxy["revenue"] += 0.014
        if float(account.get("annual_contract_value", 0.0) or 0.0) >= 90000.0:
            proxy["revenue"] += 0.020
        if float(account.get("expansion_potential", 0.0) or 0.0) >= 0.60:
            proxy["revenue"] += 0.018
        if str(account.get("support_tier", "") or "") == "premium":
            proxy["retention"] += 0.014

    text_fragments = [
        str(item.get("title", "")),
        str(item.get("description", "")),
        str(item.get("impact_summary", "")),
        str(item.get("delivery_note", "")),
        " ".join(str(note) for note in item.get("risk_notes", [])),
        " ".join(str(tag) for tag in item.get("policy_tags", [])),
    ]
    lowered = " ".join(text_fragments).lower()
    keyword_bumps = {
        "growth": ("activation", "onboarding", "demand", "pipeline", "conversion", "referral", "adoption"),
        "retention": ("renewal", "trust", "support", "reliability", "service", "churn", "escalation"),
        "revenue": ("pricing", "upsell", "expansion", "contract", "commercial", "monetization", "procurement"),
        "efficiency": ("cost", "margin", "automation", "throughput", "infra", "operating", "tooling", "manual"),
    }
    for channel, keywords in keyword_bumps.items():
        matches = sum(1 for keyword in keywords if keyword in lowered)
        proxy[channel] += 0.010 * min(matches, 3)

    if "pricing_guardrail" in item.get("policy_tags", []):
        proxy["revenue"] += 0.020
    if "margin_guardrail" in item.get("policy_tags", []):
        proxy["efficiency"] += 0.025
    if "sla_guardrail" in item.get("policy_tags", []):
        proxy["retention"] += 0.020

    lag_steps = int(item.get("lag_steps", 1) or 1)
    effect_window = int(item.get("effect_window", 1) or 1)
    if lag_steps <= 1:
        proxy["growth"] += 0.006
        proxy["revenue"] += 0.006
    else:
        proxy["retention"] += 0.008
        proxy["efficiency"] += 0.008
    if effect_window >= 2:
        proxy["retention"] += 0.006
        proxy["efficiency"] += 0.004

    return proxy


def dominant_visible_focus(proxy: Mapping[str, float]) -> str | None:
    positive = {channel: max(0.0, float(proxy.get(channel, 0.0) or 0.0)) for channel in CHANNELS}
    if max(positive.values(), default=0.0) <= 0.0:
        return None
    return max(positive, key=positive.get)
