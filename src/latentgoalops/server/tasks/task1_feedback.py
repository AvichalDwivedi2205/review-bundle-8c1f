"""Task 1: weighted feedback triage."""

from __future__ import annotations

import random

from latentgoalops.models import CustomerAccount, DashboardState, FeedbackLabel, InboxItem
from latentgoalops.server.hidden_goals import HiddenGoal, feedback_category_weight


TEMPLATES: dict[FeedbackLabel, list[str]] = {
    FeedbackLabel.BUG: [
        "The mobile checkout button disappears after the second tap.",
        "Users report a blank screen after submitting the onboarding survey.",
        "The analytics tab crashes whenever I filter by geography.",
    ],
    FeedbackLabel.FEATURE_REQUEST: [
        "Please add team-level dashboards so managers can compare cohorts.",
        "It would help if exports supported CSV and scheduled delivery.",
        "Customers want automated follow-up sequences after demos.",
    ],
    FeedbackLabel.CHURN_RISK: [
        "I’m seriously considering canceling because onboarding took too long.",
        "My team hasn’t seen value yet and finance is asking whether to renew.",
        "We may switch next month unless support gets faster and clearer.",
    ],
    FeedbackLabel.PRAISE: [
        "Your incident response last week was excellent and very transparent.",
        "The new dashboard saved our ops team hours already.",
        "Support was incredibly helpful during setup.",
    ],
    FeedbackLabel.BILLING_ISSUE: [
        "We were charged twice for the enterprise add-on this month.",
        "My invoice still shows seats we removed two weeks ago.",
        "Billing failed after the card update and now premium features are locked.",
    ],
    FeedbackLabel.LATENCY_COMPLAINT: [
        "The dashboard takes almost thirty seconds to load for our team.",
        "Response times spiked badly during afternoon usage.",
        "Search feels slow enough that reps are avoiding the product.",
    ],
}

USER_TIERS = ("free", "pro", "enterprise", "enterprise")
CHANNELS = ("email", "slack", "support_portal", "sales_escalation")

STYLE_PREFIX = {
    "concise": "Keeping this brief:",
    "demanding": "This needs immediate attention:",
    "collaborative": "Wanted to flag this and work together on it:",
    "skeptical": "I need to understand why this keeps happening:",
    "formal": "Please review the following issue:",
}


def _compose_feedback_text(label: FeedbackLabel, account: CustomerAccount, base_text: str) -> str:
    prefix = STYLE_PREFIX.get(account.communication_style, "Please review:")
    account_context = (
        f"{account.company_name} is a {account.segment.replace('_', ' ')} account in {account.industry}. "
    )
    renewal_hint = (
        f"Our renewal is in {account.renewal_window_days} days. "
        if account.renewal_window_days <= 30
        else ""
    )
    security_hint = (
        "This touches a security-sensitive workflow. "
        if account.security_sensitivity >= 0.75 and label in {FeedbackLabel.BUG, FeedbackLabel.LATENCY_COMPLAINT}
        else ""
    )
    return f"{prefix} {account_context}{base_text} {renewal_hint}{security_hint}".strip()


def _baseline_dashboard() -> DashboardState:
    return DashboardState(
        dau=18000.0,
        mau=84000.0,
        d7_retention=0.44,
        d30_retention=0.28,
        mrr=42000.0,
        arpu=96.0,
        cac=58.0,
        churn_rate=0.036,
        ops_margin=0.41,
        infra_cost_per_unit=1.84,
        support_ticket_volume=138,
    )


def _label_sampling_weights() -> dict[FeedbackLabel, float]:
    """Keep inbox composition roughly stable across latent objectives."""
    return {
        FeedbackLabel.BUG: 1.05,
        FeedbackLabel.FEATURE_REQUEST: 1.0,
        FeedbackLabel.CHURN_RISK: 1.1,
        FeedbackLabel.PRAISE: 0.9,
        FeedbackLabel.BILLING_ISSUE: 1.0,
        FeedbackLabel.LATENCY_COMPLAINT: 0.95,
    }


def build_task1_episode(rng: random.Random, hidden_goal: HiddenGoal, item_count: int, world: dict) -> dict:
    """Create a deterministic feedback triage episode."""
    labels = list(TEMPLATES.keys())
    label_weights = _label_sampling_weights()
    inbox: list[InboxItem] = []
    true_labels: dict[str, FeedbackLabel] = {}
    true_priorities: dict[str, int] = {}
    scored_items: list[tuple[str, float]] = []
    visible_accounts: list[CustomerAccount] = []
    account_pool: list[CustomerAccount] = world["accounts"]

    for index in range(item_count):
        label = rng.choices(labels, weights=[label_weights[label] for label in labels], k=1)[0]
        account = rng.choices(
            account_pool,
            weights=[
                account.strategic_importance
                + (0.25 if account.renewal_window_days <= 30 else 0.0)
                + account.churn_propensity * 0.2
                for account in account_pool
            ],
            k=1,
        )[0]
        text = _compose_feedback_text(label, account, rng.choice(TEMPLATES[label]))
        item_id = f"fb_{index + 1}"
        severity = rng.randint(1, 5)
        tier = account.segment if account.segment in {"self_serve", "smb", "enterprise"} else "enterprise"
        tier_bonus = {
            "self_serve": 0.0,
            "smb": 0.10,
            "enterprise": 0.28,
            "strategic": 0.40,
        }.get(account.segment, 0.15)
        weighted_importance = (
            severity / 5.0
            + tier_bonus
            + feedback_category_weight(hidden_goal, label)
            + min(account.annual_contract_value / 250_000.0, 0.35)
            + (0.20 if account.renewal_window_days <= 30 else 0.0)
            + account.strategic_importance * 0.15
        )
        priority = max(1, min(5, round(weighted_importance * 3.3)))

        inbox.append(
            InboxItem(
                item_id=item_id,
                text=text,
                sender=account.company_name,
                metadata={
                    "timestamp": f"2026-03-{10 + index:02d}T09:00:00Z",
                    "channel": rng.choice(CHANNELS),
                    "user_tier": tier,
                    "severity": severity,
                    "account_id": account.account_id,
                    "segment": account.segment,
                    "annual_contract_value": account.annual_contract_value,
                    "renewal_window_days": account.renewal_window_days,
                    "support_tier": account.support_tier,
                    "relationship_health": account.relationship_health,
                },
            )
        )
        if account.account_id not in {existing.account_id for existing in visible_accounts}:
            visible_accounts.append(account)
        true_labels[item_id] = label
        true_priorities[item_id] = priority
        scored_items.append((item_id, weighted_importance))

    scored_items.sort(key=lambda item: item[1], reverse=True)
    oracle_escalations = [item_id for item_id, _ in scored_items[:3]]

    return {
        "company_profile": world["company_profile"],
        "dashboard": _baseline_dashboard(),
        "inbox": inbox,
        "accounts": visible_accounts,
        "stakeholders": world["stakeholders"][:2],
        "teams": world["teams"],
        "market_context": world["market_context"],
        "governance_constraints": world["governance_constraints"],
        "true_labels": true_labels,
        "true_priorities": true_priorities,
        "oracle_escalations": oracle_escalations,
        "task_summary": "Label, prioritize, and escalate customer feedback under limited escalation capacity while inferring which business outcomes matter most.",
    }
