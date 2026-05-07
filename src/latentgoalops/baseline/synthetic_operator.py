"""Synthetic LLM operator personas, realism guardrails, and baseline stabilizers."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from latentgoalops.models import LatentGoalOpsAction, SupportPolicy, TaskId
from latentgoalops.server.public_reasoning import visible_item_proxy


@dataclass(frozen=True, slots=True)
class OperatorPersona:
    """Reusable synthetic operator profile."""

    persona_id: str
    label: str
    role: str
    narrative: str
    risk_posture: str
    change_tolerance: str
    default_focus: str
    max_parallel_bets: int
    max_parallel_initiatives: int
    max_escalations: int
    max_abs_pricing_change: float
    prefers_premium_support: bool
    decision_principles: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


PERSONAS: dict[str, OperatorPersona] = {
    "founder": OperatorPersona(
        persona_id="op_founder",
        label="Founder-Operator",
        role="Founder / GM",
        narrative="Acts like a strong founder balancing growth pressure against renewal risk and board credibility.",
        risk_posture="balanced",
        change_tolerance="medium",
        default_focus="general_management",
        max_parallel_bets=4,
        max_parallel_initiatives=2,
        max_escalations=3,
        max_abs_pricing_change=0.03,
        prefers_premium_support=True,
        decision_principles=(
            "Protect strategic accounts and renewals before chasing vanity metrics.",
            "Keep strategy legible; avoid whiplash from one day to the next.",
            "Prefer a focused portfolio over spreading attention too thin.",
        ),
    ),
    "product": OperatorPersona(
        persona_id="op_product",
        label="Product Operator",
        role="VP Product",
        narrative="Acts like a product operator who prefers coherent roadmap focus and manageable delivery complexity.",
        risk_posture="balanced",
        change_tolerance="low",
        default_focus="roadmap_quality",
        max_parallel_bets=4,
        max_parallel_initiatives=2,
        max_escalations=2,
        max_abs_pricing_change=0.02,
        prefers_premium_support=True,
        decision_principles=(
            "Favor a coherent product thesis over opportunistic one-off wins.",
            "Avoid taking on more initiatives than teams can plausibly absorb.",
            "Bias toward durable fixes for premium accounts and recurring pain points.",
        ),
    ),
    "support": OperatorPersona(
        persona_id="op_support",
        label="Customer Success Operator",
        role="Head of Support / CS",
        narrative="Acts like a customer success leader who treats renewals, trust, and premium support handling as central.",
        risk_posture="conservative",
        change_tolerance="low",
        default_focus="retention",
        max_parallel_bets=3,
        max_parallel_initiatives=1,
        max_escalations=3,
        max_abs_pricing_change=0.01,
        prefers_premium_support=True,
        decision_principles=(
            "Do not trade away trust from premium or renewal-risk accounts for short-term efficiency.",
            "Escalate crisply when support expectations or trust are at stake.",
            "Prefer stability and service recovery over aggressive experimentation.",
        ),
    ),
    "finance": OperatorPersona(
        persona_id="op_finance",
        label="Finance-Minded Operator",
        role="CFO / BizOps",
        narrative="Acts like a finance-minded operator optimizing margin, monetization discipline, and runway protection.",
        risk_posture="measured",
        change_tolerance="medium",
        default_focus="efficiency_revenue",
        max_parallel_bets=3,
        max_parallel_initiatives=2,
        max_escalations=2,
        max_abs_pricing_change=0.04,
        prefers_premium_support=False,
        decision_principles=(
            "Protect runway and operating leverage before adding new complexity.",
            "Use pricing and automation carefully, especially around strategic renewals.",
            "Prefer fewer, higher-confidence bets with visible payback paths.",
        ),
    ),
    "gm": OperatorPersona(
        persona_id="op_gm",
        label="General Manager",
        role="GM",
        narrative="Acts like a pragmatic GM balancing revenue, retention, and execution capacity without overreacting.",
        risk_posture="balanced",
        change_tolerance="medium",
        default_focus="portfolio_balance",
        max_parallel_bets=4,
        max_parallel_initiatives=2,
        max_escalations=2,
        max_abs_pricing_change=0.025,
        prefers_premium_support=True,
        decision_principles=(
            "Take defendable actions that a real leadership team could explain tomorrow.",
            "Avoid maxing out budget or capacity without a clear reason.",
            "Let strong evidence move the plan, but keep continuity unless the world truly changes.",
        ),
    ),
}


TASK_SHORTLISTS = {
    TaskId.TASK1: ("support", "gm", "founder"),
    TaskId.TASK2: ("product", "finance", "gm", "founder"),
    TaskId.TASK3: ("founder", "gm", "finance", "support"),
    TaskId.TASK4: ("finance", "gm", "founder"),
    TaskId.TASK5: ("support", "founder", "gm", "finance"),
    TaskId.TASK6: ("support", "founder", "gm", "finance"),
    TaskId.TASK7: ("finance", "gm", "founder", "product"),
}


def _accounts_by_id(observation: dict) -> dict[str, dict]:
    return {
        str(account.get("account_id")): account
        for account in observation.get("accounts", [])
    }


def _visible_deltas(item: dict, accounts_by_id: dict[str, dict] | None = None) -> dict[str, float]:
    return visible_item_proxy(item, accounts_by_id=accounts_by_id)


def operator_style_choices() -> list[str]:
    """Expose valid CLI choices."""
    return ["auto", *PERSONAS.keys()]


def resolve_operator_persona(style: str, seed: int, task_id: TaskId) -> OperatorPersona:
    """Return a deterministic operator persona for one seed/task."""
    if style != "auto":
        return PERSONAS[style]
    shortlist = TASK_SHORTLISTS[task_id]
    rng = random.Random(seed * 97 + sum(ord(char) for char in task_id.value))
    return PERSONAS[rng.choice(shortlist)]


def _task2_item_score(item: dict, persona: OperatorPersona, accounts_by_id: dict[str, dict] | None = None) -> float:
    deltas = _visible_deltas(item, accounts_by_id)
    if persona.default_focus == "retention":
        weights = {"growth": 0.5, "retention": 1.4, "revenue": 0.7, "efficiency": 0.6}
    elif persona.default_focus == "efficiency_revenue":
        weights = {"growth": 0.5, "retention": 0.8, "revenue": 1.2, "efficiency": 1.2}
    elif persona.default_focus == "roadmap_quality":
        weights = {"growth": 0.8, "retention": 1.0, "revenue": 0.9, "efficiency": 0.8}
    else:
        weights = {"growth": 0.9, "retention": 1.0, "revenue": 1.0, "efficiency": 0.9}
    weighted_gain = sum(max(0.0, float(deltas.get(channel, 0.0))) * weights[channel] for channel in weights)
    risk_penalty = float(item.get("implementation_risk", 0.0)) * 0.35
    segment_bonus = 0.02 * len(item.get("beneficiary_segments", []))
    account_bonus = 0.02 * len(item.get("beneficiary_account_ids", []))
    cost = max(float(item.get("cost", 1.0)), 1.0)
    return (weighted_gain + segment_bonus + account_bonus - risk_penalty) / cost


def _task2_proxy_item_value(item: dict, accounts_by_id: dict[str, dict] | None = None) -> float:
    """Visible-only proxy value used to keep the submission baseline stable."""
    deltas = _visible_deltas(item, accounts_by_id)
    weighted_gain = (
        max(0.0, float(deltas.get("growth", 0.0))) * 0.95
        + max(0.0, float(deltas.get("retention", 0.0))) * 1.15
        + max(0.0, float(deltas.get("revenue", 0.0))) * 1.00
        + max(0.0, float(deltas.get("efficiency", 0.0))) * 0.90
    )
    risk_penalty = float(item.get("implementation_risk", 0.0)) * 0.04
    segment_bonus = 0.01 * len(item.get("beneficiary_segments", []))
    account_bonus = 0.01 * len(item.get("beneficiary_account_ids", []))
    return max(0.0, weighted_gain + segment_bonus + account_bonus - risk_penalty)


def _task2_proxy_bundle_score(selected_item_ids: list[str], observation: dict) -> float:
    """Approximate visible bundle quality without using hidden goal state."""
    backlog = observation.get("backlog", [])
    accounts_by_id = _accounts_by_id(observation)
    budget = float(observation.get("sprint_budget") or 0.0)
    chosen = {
        item.get("item_id"): item
        for item in backlog
        if item.get("item_id") in set(selected_item_ids)
    }
    if not chosen:
        return 0.0

    spent_budget = sum(float(item.get("cost", 0.0)) for item in chosen.values())
    if budget > 0.0 and spent_budget > budget + 1e-9:
        return -1.0

    score = 0.0
    focus_support = {"growth": 0.0, "retention": 0.0, "revenue": 0.0, "efficiency": 0.0}
    for item in chosen.values():
        proxy = _visible_deltas(item, accounts_by_id)
        base = _task2_proxy_item_value(item, accounts_by_id)
        if item.get("requires_item_ids") and not any(
            required in chosen for required in item.get("requires_item_ids", [])
        ):
            base *= 0.55
        if item.get("synergy_item_ids") and any(
            synergy in chosen for synergy in item.get("synergy_item_ids", [])
        ):
            base *= 1.10
        score += base
        for channel, value in proxy.items():
            focus_support[channel] += max(0.0, float(value))

    conflict_penalty = 0.0
    for item in chosen.values():
        item_id = item.get("item_id")
        for conflict_id in item.get("conflicts_with_ids", []):
            if conflict_id in chosen and str(item_id) < str(conflict_id):
                conflict_penalty += 0.18 * min(
                    _task2_proxy_item_value(item, accounts_by_id),
                    _task2_proxy_item_value(chosen[conflict_id], accounts_by_id),
                )

    execution_penalty = sum(float(item.get("implementation_risk", 0.0)) * 0.03 for item in chosen.values())
    budget_use = (spent_budget / budget) if budget > 0.0 else 1.0
    budget_bonus = 0.03 * max(0.0, min(1.0, budget_use))
    concentration_bonus = 0.0
    total_focus = sum(focus_support.values())
    if total_focus > 0.0 and max(focus_support.values()) / total_focus >= 0.55:
        concentration_bonus = 0.02
    return max(0.0, score - conflict_penalty - execution_penalty + budget_bonus + concentration_bonus)


def stabilize_model_action(
    action: LatentGoalOpsAction,
    observation: dict,
    fallback_action: LatentGoalOpsAction,
) -> tuple[LatentGoalOpsAction, bool]:
    """Keep the submitted model baseline from regressing below a deterministic floor."""
    if action.task_id != TaskId.TASK2:
        return action, False

    model_score = _task2_proxy_bundle_score(action.selected_item_ids, observation)
    fallback_score = _task2_proxy_bundle_score(fallback_action.selected_item_ids, observation)
    if model_score + 1e-9 >= fallback_score:
        return action, False

    replacement = fallback_action.model_copy(
        update={
            "rationale_summary": action.rationale_summary or fallback_action.rationale_summary,
        }
    )
    return replacement, True


def apply_operator_guardrails(
    action: LatentGoalOpsAction,
    observation: dict,
    persona: OperatorPersona,
) -> tuple[LatentGoalOpsAction, bool]:
    """Apply mild realism constraints so the operator behaves like a bounded human."""
    adjusted = False

    if action.task_id == TaskId.TASK1:
        escalate_ids = action.escalate_ids[: persona.max_escalations]
        adjusted = adjusted or escalate_ids != action.escalate_ids
        return action.model_copy(update={"escalate_ids": escalate_ids}), adjusted

    if action.task_id == TaskId.TASK2:
        backlog = observation.get("backlog", [])
        accounts_by_id = _accounts_by_id(observation)
        visible_ids = {item.get("item_id") for item in observation.get("backlog", [])}
        selected_item_ids = [
            item_id for item_id in action.selected_item_ids if item_id in visible_ids
        ][: persona.max_parallel_bets]
        sprint_budget = float(observation.get("sprint_budget") or 0.0)
        cost_by_id = {item.get("item_id"): float(item.get("cost", 0.0)) for item in backlog}
        spent_budget = sum(cost_by_id.get(item_id, 0.0) for item_id in selected_item_ids)
        if backlog and sprint_budget > 0 and spent_budget < 0.6 * sprint_budget:
            used = set(selected_item_ids)
            ranked = sorted(
                backlog,
                key=lambda item: _task2_item_score(item, persona, accounts_by_id),
                reverse=True,
            )
            for item in ranked:
                item_id = item.get("item_id")
                if item_id in used:
                    continue
                item_cost = float(item.get("cost", 0.0))
                if len(selected_item_ids) >= persona.max_parallel_bets or spent_budget + item_cost > sprint_budget + 1e-9:
                    continue
                selected_item_ids.append(item_id)
                used.add(item_id)
                spent_budget += item_cost
                adjusted = True
                if spent_budget >= 0.6 * sprint_budget:
                    break
        adjusted = adjusted or selected_item_ids != action.selected_item_ids
        return action.model_copy(update={"selected_item_ids": selected_item_ids}), adjusted

    if action.task_id == TaskId.TASK4:
        backlog = observation.get("backlog", [])
        visible_by_id = {item.get("item_id"): item for item in backlog}
        sprint_budget = int(round(float(observation.get("sprint_budget") or 0.0)))
        allocations = {}
        for item_id, amount in action.budget_allocations.items():
            if item_id not in visible_by_id:
                continue
            item = visible_by_id[item_id]
            clamped = max(
                0,
                min(
                    int(round(float(amount))),
                    int(round(float(item.get("allocation_max") or 0.0))),
                ),
            )
            if clamped > 0:
                allocations[item_id] = float(clamped)
        ranked_ids = sorted(
            allocations,
            key=lambda item_id: _task2_item_score(visible_by_id[item_id], persona),
            reverse=True,
        )
        if len(ranked_ids) > persona.max_parallel_bets:
            adjusted = True
        allocations = {item_id: allocations[item_id] for item_id in ranked_ids[: persona.max_parallel_bets]}
        spent_budget = int(round(sum(allocations.values())))
        if spent_budget > sprint_budget:
            adjusted = True
            ranked_backlog = sorted(backlog, key=lambda item: _task2_item_score(item, persona), reverse=True)
            allocations = {}
            spent_budget = 0
            for item in ranked_backlog:
                cap = int(round(float(item.get("saturation_point") or item.get("allocation_max") or 0.0)))
                amount = min(cap, sprint_budget - spent_budget)
                if amount <= 0:
                    continue
                allocations[item.get("item_id")] = float(amount)
                spent_budget += amount
                if spent_budget >= sprint_budget or len(allocations) >= persona.max_parallel_bets:
                    break
        adjusted = adjusted or allocations != action.budget_allocations
        return action.model_copy(update={"budget_allocations": allocations}), adjusted

    visible_ids = {item.get("item_id") for item in observation.get("backlog", [])}
    chosen_initiatives = [
        item_id for item_id in action.chosen_initiatives if item_id in visible_ids
    ][: persona.max_parallel_initiatives]
    pricing = float(action.pricing_change_pct or 0.0)
    clamped_pricing = max(-persona.max_abs_pricing_change, min(persona.max_abs_pricing_change, pricing))
    support_policy = action.support_policy

    high_touch_accounts = [
        account
        for account in observation.get("accounts", [])
        if account.get("support_tier") == "premium"
        or (
            account.get("segment") in {"enterprise", "strategic"}
            and (account.get("renewal_window_days") or 999) <= 30
        )
    ]
    visible_constraint_ids = {
        constraint.get("constraint_id")
        for constraint in observation.get("governance_constraints", [])
    }
    if (
        persona.prefers_premium_support
        and high_touch_accounts
        and support_policy == SupportPolicy.AUTOMATION_FIRST
        and "sla_guardrail" in visible_constraint_ids
    ):
        support_policy = SupportPolicy.BALANCED_TRIAGE
        adjusted = True

    if (
        clamped_pricing > 0.02
        and high_touch_accounts
        and "pricing_guardrail" in visible_constraint_ids
        and persona.risk_posture in {"conservative", "balanced"}
    ):
        clamped_pricing = 0.02
        adjusted = True

    if chosen_initiatives != action.chosen_initiatives:
        adjusted = True
    if abs(clamped_pricing - pricing) > 1e-9:
        adjusted = True

    return (
        action.model_copy(
            update={
                "chosen_initiatives": chosen_initiatives,
                "pricing_change_pct": round(clamped_pricing, 3),
                "support_policy": support_policy,
            }
        ),
        adjusted,
    )
