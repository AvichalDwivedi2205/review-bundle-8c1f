"""Prompt builders for model baselines."""

from __future__ import annotations

import json
import re
from typing import Any

from latentgoalops.models import FeedbackLabel, MessagingAction, SupportPolicy, TaskId


def _shared_system_prefix() -> str:
    return (
        "You are an expert operator inside a research benchmark for startup operations. "
        "Your real task is not just to react locally, but to infer the latent business objective from indirect evidence "
        "such as KPI dashboards, stakeholder messages, backlog structure, alerts, and resource constraints. "
        "Maintain one coherent working hypothesis unless new evidence strongly contradicts it. "
        "Do not explain your reasoning. Return exactly one compact JSON object for the active task fields only and nothing else."
    )


def _operator_prefix(operator_profile: Any | None) -> str:
    if operator_profile is None:
        return _shared_system_prefix()
    return (
        f"You are a synthetic startup operator acting as {operator_profile.label} ({operator_profile.role}). "
        "Do not optimize for benchmark score directly. Make realistic, defendable decisions under uncertainty. "
        f"Risk posture: {operator_profile.risk_posture}. Change tolerance: {operator_profile.change_tolerance}. "
        f"Keep plans focused: roadmap <= {operator_profile.max_parallel_bets} items, startup initiatives <= {operator_profile.max_parallel_initiatives}, "
        f"pricing moves roughly within +/-{operator_profile.max_abs_pricing_change:.2f}. "
        "Prefer continuity, lower-regret choices, and careful handling of premium or renewal-risk accounts. "
        "Do not explain hidden reasoning. Return exactly one compact JSON object for the active task fields only and nothing else."
    )


def _task1_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 1 is weighted feedback triage. Label every inbox item, set a 1-5 priority for every item, and escalate at most 3 IDs. "
        "Use the hidden objective only to influence prioritization and escalation, never to change the underlying label semantics. "
        "Pay attention to account economics such as annual_contract_value, renewal_window_days, support_tier, and relationship_health. "
        "Allowed labels: bug, feature_request, churn_risk, praise, billing_issue, latency_complaint. "
        "Do not include task_id or explanations. Output schema: "
        "{"
        "\"labels\":{\"item_id\":\"bug\"},"
        "\"priorities\":{\"item_id\":3},"
        "\"escalate_ids\":[\"...\"]"
        "}."
    )


def _task2_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 2 is roadmap prioritization under a hard sprint budget. Select only visible backlog item IDs. "
        "Prefer a coherent bundle of initiatives that all support the same latent objective rather than a noisy mix of conflicting bets. "
        "Use customer segments, beneficiary accounts, stakeholder agendas, implementation risk, and governance constraints as part of the decision. "
        "Treat multiple IDs from the same initiative family as alternative variants and usually choose at most one variant per family. "
        "Never exceed the budget. Under-spending is usually a visible mistake: unless conflicts or missing prerequisites clearly block you, "
        "aim to spend most of sprint_budget and usually choose roughly four to six mutually coherent items rather than a tiny two-item bundle. "
        "If uncertain, choose the highest visible KPI gain per unit cost that matches your inferred objective. "
        "Do not include task_id or rationale text unless truly needed. Output schema: "
        "{"
        "\"selected\":[\"...\"]"
        "}."
    )


def _task3_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 3 is a startup operations simulation with an explicit calendar, prior decision ledger, pending delayed effects, "
        "realized consequences from earlier choices, and a retrieved memory workspace. You must act coherently across dates, preserve budget and capacity, "
        "manage a portfolio of customer accounts, recurring stakeholder agendas, team execution constraints, and governance guardrails, "
        "and adapt if the evidence suggests the objective has silently shifted. "
        "Avoid mixing growth, retention, revenue, and efficiency actions without a clear reason. Prefer plans whose delayed effects remain aligned over multiple days. "
        "memory_focus, memory_writes, and belief_report are optional diagnostics: only include them when they materially help the next step. "
        "Choose only visible initiative IDs. Keep pricing_change_pct in [-0.20, 0.20]. "
        "Allowed messaging_action values: growth_push, retention_campaign, revenue_upsell, cost_comms, none. "
        "Allowed support_policy values: premium_sla, balanced_triage, automation_first, incident_swarm. "
        "Do not include task_id. Omit optional fields you do not need. Prefer the minimal schema: "
        "{"
        "\"chosen\":[\"...\"],"
        "\"msg\":\"retention_campaign\","
        "\"price\":0.0,"
        "\"support\":\"balanced_triage\""
        "}. "
        "Only include focus, writes, or belief if they are short and materially useful."
    )


def _task6_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 6 is a multi-day incident response week. Track noisy evidence, delayed recovery effects, and open conflicts in memory while adapting to possible latent-goal shifts. "
        "Prefer coherent recovery sequences over one-step KPI chasing. "
        "For strict evaluation, prioritize a short valid control payload over verbose bookkeeping: memory_focus, memory_writes, and belief_report are optional. "
        "Choose only visible initiative IDs. Keep pricing_change_pct in [-0.20, 0.20]. "
        "Allowed messaging_action values: growth_push, retention_campaign, revenue_upsell, cost_comms, none. "
        "Allowed support_policy values: premium_sla, balanced_triage, automation_first, incident_swarm. "
        "Do not include task_id. Prefer the minimal schema: "
        "{"
        "\"chosen\":[\"...\"],"
        "\"msg\":\"retention_campaign\","
        "\"price\":0.0,"
        "\"support\":\"incident_swarm\""
        "}. "
        "Only include focus, writes, or belief if they are short and genuinely useful."
    )


def _task4_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 4 is capital allocation under uncertainty. Allocate discrete budget points across visible program IDs using budget_allocations. "
        "Programs have visible allocation_max, saturation hints, dependencies, conflicts, and stakeholder pressure. "
        "Prefer one focused capital story rather than sprinkling tiny amounts across every program. "
        "In most episodes, that means one anchor program and one or two enabling programs, usually with 2 to 4 funded IDs total rather than a five-program scattershot mix. "
        "Use the visible renewal concentration, expansion upside, runway pressure, support strain, dependencies, and conflicts to decide which capital story actually fits the room. "
        "If two programs visibly send conflicting strategic signals, do not fund both heavily. Spend most of the budget unless visible dependencies or conflicts make that unsafe. "
        "Never exceed the total budget. Use integer budget points only. "
        "Do not include task_id or rationale text unless truly needed. Output schema: "
        "{"
        "\"allocations\":{\"program_id\":2}"
        "}."
    )


def _task7_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 7 is quarterly headcount planning. Allocate visible hiring slots with budget_allocations and reason over delayed staffing effects over time. "
        "Headcount compounds slowly: in most quarters, back one primary staffing thesis and at most one enabling hire instead of spreading allocations across many programs. "
        "Growth or revenue hires often underperform if support, reliability, or delivery capacity is already strained and no enabling support or infrastructure hire is in place. "
        "Use the narrative, alerts, team state, and stakeholder notes to decide whether the company needs coverage, reliability, monetization support, or growth capacity right now. "
        "Do not exceed budget_remaining for the current quarter. Use integer allocations only. "
        "memory_focus, memory_writes, and belief_report are optional diagnostics: omit them unless they are short and materially useful. "
        "Do not include task_id. Omit optional fields you do not need. Prefer the minimal schema: "
        "{"
        "\"allocations\":{\"hire_sre_cluster\":2}"
        "}."
    )


def _task5_system_prompt(policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    return (
        f"{_operator_prefix(operator_profile) if policy_mode == 'synthetic_operator' else _shared_system_prefix()} "
        "Task 5 is a one-shot crisis response package. Choose a small set of visible initiative IDs and combine them with pricing, messaging, and support policy. "
        "Treat it like one executive operating decision: the best package should be coherent, defendable, and safe for high-touch accounts. "
        "Avoid aggressive pricing or automation-heavy support when the visible evidence suggests enterprise trust is already fragile. "
        "Allowed messaging_action values: growth_push, retention_campaign, revenue_upsell, cost_comms, none. "
        "Allowed support_policy values: premium_sla, balanced_triage, automation_first, incident_swarm. "
        "Do not include task_id or explanations. Output schema: "
        "{"
        "\"chosen\":[\"...\"],"
        "\"msg\":\"retention_campaign\","
        "\"price\":0.0,"
        "\"support\":\"balanced_triage\""
        "}."
    )


def system_prompt(task_id: TaskId, policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    """Return the system prompt for a given task."""
    if task_id == TaskId.TASK1:
        return _task1_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK2:
        return _task2_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK3:
        return _task3_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK6:
        return _task6_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK4:
        return _task4_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    if task_id == TaskId.TASK7:
        return _task7_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)
    return _task5_system_prompt(policy_mode=policy_mode, operator_profile=operator_profile)


def output_schema(task_id: TaskId) -> dict[str, Any]:
    """Return a compact JSON schema for strict generation."""
    if task_id == TaskId.TASK1:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "labels": {
                    "type": "object",
                    "additionalProperties": {"type": "string", "enum": [label.value for label in FeedbackLabel]},
                },
                "priorities": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "escalate_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["labels", "priorities", "escalate_ids"],
        }
    if task_id == TaskId.TASK2:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["selected"],
        }
    if task_id in {TaskId.TASK3, TaskId.TASK5, TaskId.TASK6}:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "chosen": {"type": "array", "items": {"type": "string"}},
                "msg": {"type": "string", "enum": [value.value for value in MessagingAction]},
                "price": {"type": "number", "minimum": -0.2, "maximum": 0.2},
                "support": {"type": "string", "enum": [value.value for value in SupportPolicy]},
            },
            "required": ["chosen", "msg", "price", "support"],
        }
    if task_id in {TaskId.TASK4, TaskId.TASK7}:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "allocations": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                }
            },
            "required": ["allocations"],
        }
    return {"type": "object", "additionalProperties": True}


def _allowed_ids(observation: dict) -> list[str]:
    task_id = TaskId(observation["task_id"])
    if task_id == TaskId.TASK1:
        return [item["item_id"] for item in observation.get("inbox", [])]
    return [item["item_id"] for item in observation.get("backlog", [])]


def _json_template(task_id: TaskId) -> dict:
    if task_id == TaskId.TASK1:
        return {
            "labels": {"fb_1": "bug"},
            "priorities": {"fb_1": 3},
            "escalate_ids": ["fb_1"],
        }
    if task_id == TaskId.TASK2:
        return {
            "selected": ["item_1"],
        }
    if task_id == TaskId.TASK3:
        return {
            "chosen": ["optimize_infra"],
            "msg": "cost_comms",
            "price": 0.0,
            "support": "automation_first",
        }
    if task_id == TaskId.TASK4:
        return {
            "allocations": {"renewal_rescue_pod": 3},
        }
    if task_id == TaskId.TASK6:
        return {
            "chosen": ["refactor_incident_tooling"],
            "msg": "retention_campaign",
            "price": 0.0,
            "support": "incident_swarm",
        }
    if task_id == TaskId.TASK7:
        return {
            "allocations": {"hire_sre_cluster": 2},
        }
    return {
        "chosen": ["fix_login_bug"],
        "msg": "retention_campaign",
        "price": 0.0,
        "support": "balanced_triage",
    }


def _dashboard_brief(observation: dict) -> dict:
    dashboard = observation.get("dashboard", {})
    return {
        "dau": dashboard.get("dau"),
        "d30_retention": dashboard.get("d30_retention"),
        "mrr": dashboard.get("mrr"),
        "ops_margin": dashboard.get("ops_margin"),
        "support_ticket_volume": dashboard.get("support_ticket_volume"),
    }


def _initiative_family(item_id: Any) -> str:
    value = str(item_id)
    match = re.match(r"(.+)_\d+$", value)
    if match:
        return match.group(1)
    return value


def _company_profile_brief(observation: dict) -> dict:
    profile = observation.get("company_profile") or {}
    return {
        "archetype_id": profile.get("archetype_id"),
        "product_type": profile.get("product_type"),
        "gtm_motion": profile.get("gtm_motion"),
        "pricing_model": profile.get("pricing_model"),
        "compliance_regime": profile.get("compliance_regime"),
        "core_user_workflow": profile.get("core_user_workflow"),
        "ideal_customer_profile": profile.get("ideal_customer_profile"),
        "board_narrative": profile.get("board_narrative"),
    }


def _memory_brief(observation: dict) -> dict:
    workspace = observation.get("memory_workspace") or {}
    return {
        "memory_budget_remaining": observation.get("memory_budget_remaining"),
        "memory_write_budget_remaining": observation.get("memory_write_budget_remaining"),
        "open_conflicts": workspace.get("open_conflicts", []),
        "entity_timelines": workspace.get("entity_timelines", {}),
        "records": [
            {
                "record_id": record.get("record_id"),
                "summary": record.get("summary"),
                "tags": record.get("tags", []),
                "stale": record.get("stale"),
                "conflict_group": record.get("conflict_group"),
            }
            for record in workspace.get("records", [])[:6]
        ],
    }


def _compact_task1_observation(observation: dict) -> dict:
    accounts_by_id = {account["account_id"]: account for account in observation.get("accounts", [])}
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "company_profile": _company_profile_brief(observation),
        "dashboard": _dashboard_brief(observation),
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "inbox": [
            {
                "item_id": item.get("item_id"),
                "sender": item.get("sender"),
                "text": item.get("text"),
                "severity": item.get("metadata", {}).get("severity"),
                "channel": item.get("metadata", {}).get("channel"),
                "account": {
                    "account_id": item.get("metadata", {}).get("account_id"),
                    "segment": item.get("metadata", {}).get("segment"),
                    "acv": item.get("metadata", {}).get("annual_contract_value"),
                    "renewal_window_days": item.get("metadata", {}).get("renewal_window_days"),
                    "support_tier": item.get("metadata", {}).get("support_tier"),
                    "relationship_health": item.get("metadata", {}).get("relationship_health"),
                    "strategic_importance": accounts_by_id.get(
                        item.get("metadata", {}).get("account_id"), {}
                    ).get("strategic_importance"),
                },
            }
            for item in observation.get("inbox", [])
        ],
    }


def _compact_task2_observation(observation: dict) -> dict:
    family_counts: dict[str, int] = {}
    for item in observation.get("backlog", []):
        family = _initiative_family(item.get("item_id"))
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "company_profile": _company_profile_brief(observation),
        "dashboard": _dashboard_brief(observation),
        "sprint_budget": observation.get("sprint_budget"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "gross_margin_target": (observation.get("market_context") or {}).get("gross_margin_target"),
        },
        "stakeholder_notes": observation.get("stakeholder_notes", []),
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])
        ],
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "initiative_family": _initiative_family(item.get("item_id")),
                "family_variant_count": family_counts.get(_initiative_family(item.get("item_id")), 1),
                "cost": item.get("cost"),
                "impact_summary": item.get("impact_summary"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "policy_tags": item.get("policy_tags", []),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task4_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "dashboard": _dashboard_brief(observation),
        "sprint_budget": observation.get("sprint_budget"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "gross_margin_target": (observation.get("market_context") or {}).get("gross_margin_target"),
        },
        "stakeholder_notes": observation.get("stakeholder_notes", []),
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])[:6]
        ],
        "memory_workspace": _memory_brief(observation) if observation.get("memory_workspace") else None,
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
                "impact_summary": item.get("impact_summary"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task7_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "company_profile": _company_profile_brief(observation),
        "step_index": observation.get("step_index"),
        "horizon": observation.get("horizon"),
        "sim_date": observation.get("sim_date"),
        "sim_day_label": observation.get("sim_day_label"),
        "narrative": observation.get("narrative"),
        "dashboard": _dashboard_brief(observation),
        "budget_remaining": observation.get("budget_remaining"),
        "sprint_budget": observation.get("sprint_budget"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "gross_margin_target": (observation.get("market_context") or {}).get("gross_margin_target"),
            "sales_pipeline_health": (observation.get("market_context") or {}).get("sales_pipeline_health"),
        },
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])[:6]
        ],
        "teams": [
            {
                "team_id": team.get("team_id"),
                "function": team.get("function"),
                "capacity": team.get("capacity"),
                "burnout_risk": team.get("burnout_risk"),
                "execution_reliability": team.get("execution_reliability"),
                "cross_team_friction": team.get("cross_team_friction"),
            }
            for team in observation.get("teams", [])
        ],
        "calendar_events": [
            {
                "name": event.get("name"),
                "sim_date": event.get("sim_date"),
                "status": event.get("status"),
                "summary": event.get("summary"),
            }
            for event in observation.get("calendar_events", [])
        ],
        "alerts": observation.get("alerts", []),
        "stakeholder_notes": observation.get("stakeholder_notes", []),
        "decision_ledger": [
            {
                "sim_date": entry.get("sim_date"),
                "chosen_initiatives": entry.get("chosen_initiatives", []),
            }
            for entry in observation.get("decision_ledger", [])[-3:]
        ],
        "pending_effects": [
            {
                "summary": effect.get("summary"),
                "scheduled_for_date": effect.get("scheduled_for_date"),
                "affected_account_ids": effect.get("affected_account_ids", []),
                "affected_team_ids": effect.get("affected_team_ids", []),
            }
            for effect in observation.get("pending_effects", [])[:5]
        ],
        "realized_effects": [
            {
                "summary": effect.get("summary"),
                "realized_date": effect.get("realized_date"),
                "affected_account_ids": effect.get("affected_account_ids", []),
                "affected_team_ids": effect.get("affected_team_ids", []),
            }
            for effect in observation.get("realized_effects", [])[:5]
        ],
        "memory_workspace": _memory_brief(observation),
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
                "impact_summary": item.get("impact_summary"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task3_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "company_profile": _company_profile_brief(observation),
        "step_index": observation.get("step_index"),
        "horizon": observation.get("horizon"),
        "sim_date": observation.get("sim_date"),
        "sim_day_label": observation.get("sim_day_label"),
        "narrative": observation.get("narrative"),
        "dashboard": _dashboard_brief(observation),
        "budget_remaining": observation.get("budget_remaining"),
        "capacity_remaining": observation.get("capacity_remaining"),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "sales_pipeline_health": (observation.get("market_context") or {}).get("sales_pipeline_health"),
            "competition_intensity": (observation.get("market_context") or {}).get("competition_intensity"),
        },
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "support_tier": account.get("support_tier"),
                "relationship_health": account.get("relationship_health"),
                "churn_propensity": account.get("churn_propensity"),
                "strategic_importance": account.get("strategic_importance"),
            }
            for account in observation.get("accounts", [])[:5]
        ],
        "stakeholders": [
            {
                "name": stakeholder.get("name"),
                "role": stakeholder.get("role"),
                "political_power": stakeholder.get("political_power"),
                "favorite_metrics": stakeholder.get("favorite_metrics", []),
            }
            for stakeholder in observation.get("stakeholders", [])[:4]
        ],
        "teams": [
            {
                "function": team.get("function"),
                "capacity": team.get("capacity"),
                "burnout_risk": team.get("burnout_risk"),
                "execution_reliability": team.get("execution_reliability"),
            }
            for team in observation.get("teams", [])
        ],
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
                "threshold": constraint.get("threshold"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "alerts": observation.get("alerts", []),
        "calendar_events": [
            {
                "name": event.get("name"),
                "sim_date": event.get("sim_date"),
                "status": event.get("status"),
                "summary": event.get("summary"),
            }
            for event in observation.get("calendar_events", [])
        ],
        "decision_ledger": [
            {
                "sim_date": entry.get("sim_date"),
                "chosen_initiatives": entry.get("chosen_initiatives", []),
                "messaging_action": entry.get("messaging_action"),
                "pricing_change_pct": entry.get("pricing_change_pct"),
                "support_policy": entry.get("support_policy"),
                "governance_flags": entry.get("governance_flags", []),
            }
            for entry in observation.get("decision_ledger", [])[-3:]
        ],
        "pending_effects": [
            {
                "summary": effect.get("summary"),
                "scheduled_for_date": effect.get("scheduled_for_date"),
                "affected_account_ids": effect.get("affected_account_ids", []),
                "affected_team_ids": effect.get("affected_team_ids", []),
            }
            for effect in observation.get("pending_effects", [])[:5]
        ],
        "realized_effects": [
            {
                "summary": effect.get("summary"),
                "realized_date": effect.get("realized_date"),
                "affected_account_ids": effect.get("affected_account_ids", []),
                "affected_team_ids": effect.get("affected_team_ids", []),
            }
            for effect in observation.get("realized_effects", [])[:5]
        ],
        "memory_workspace": _memory_brief(observation),
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "cost": item.get("cost"),
                "impact_summary": item.get("impact_summary"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "policy_tags": item.get("policy_tags", []),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
                "allocation_unit": item.get("allocation_unit"),
                "allocation_max": item.get("allocation_max"),
                "saturation_point": item.get("saturation_point"),
            }
            for item in observation.get("backlog", [])
        ],
    }


def _compact_task5_observation(observation: dict) -> dict:
    return {
        "task_id": observation["task_id"],
        "task_summary": observation.get("task_summary"),
        "company_profile": _company_profile_brief(observation),
        "narrative": observation.get("narrative"),
        "dashboard": _dashboard_brief(observation),
        "budget_remaining": observation.get("budget_remaining"),
        "capacity_remaining": observation.get("capacity_remaining"),
        "alerts": observation.get("alerts", []),
        "market_context": {
            "cash_runway_months": (observation.get("market_context") or {}).get("cash_runway_months"),
            "board_pressure_level": (observation.get("market_context") or {}).get("board_pressure_level"),
            "sales_pipeline_health": (observation.get("market_context") or {}).get("sales_pipeline_health"),
        },
        "inbox": [
            {
                "item_id": item.get("item_id"),
                "sender": item.get("sender"),
                "text": item.get("text"),
            }
            for item in observation.get("inbox", [])
        ],
        "accounts": [
            {
                "account_id": account.get("account_id"),
                "company_name": account.get("company_name"),
                "segment": account.get("segment"),
                "acv": account.get("annual_contract_value"),
                "renewal_window_days": account.get("renewal_window_days"),
                "support_tier": account.get("support_tier"),
                "relationship_health": account.get("relationship_health"),
            }
            for account in observation.get("accounts", [])[:6]
        ],
        "governance_constraints": [
            {
                "constraint_id": constraint.get("constraint_id"),
                "title": constraint.get("title"),
                "severity": constraint.get("severity"),
            }
            for constraint in observation.get("governance_constraints", [])
        ],
        "backlog": [
            {
                "item_id": item.get("item_id"),
                "cost": item.get("cost"),
                "impact_summary": item.get("impact_summary"),
                "beneficiary_segments": item.get("beneficiary_segments", []),
                "beneficiary_account_ids": item.get("beneficiary_account_ids", []),
                "implementation_risk": item.get("implementation_risk"),
                "policy_tags": item.get("policy_tags", []),
                "requires_item_ids": item.get("requires_item_ids", []),
                "conflicts_with_ids": item.get("conflicts_with_ids", []),
                "synergy_item_ids": item.get("synergy_item_ids", []),
                "risk_notes": item.get("risk_notes", []),
            }
            for item in observation.get("backlog", [])
        ],
    }


def user_prompt(observation: dict, policy_mode: str = "model", operator_profile: Any | None = None) -> str:
    """Serialize an observation as the user prompt."""
    task_id = TaskId(observation["task_id"])
    hints = [
        "Use only IDs that appear in the observation.",
        "Keep the policy internally coherent with one inferred objective.",
        "If evidence is ambiguous, prefer the strongest non-contradictory signal from dashboard + notes + alerts.",
        "Use the shortest valid JSON object that satisfies the compact schema and omit optional fields unless they matter.",
        "Return one JSON object only with no markdown fences.",
    ]
    if policy_mode == "synthetic_operator" and operator_profile is not None:
        hints.append("Behave like a realistic human operator: do not over-rotate, over-parallelize, or use extreme pricing without strong evidence.")
        hints.append("Prefer one defendable operating plan over a benchmark-gaming move that a real leadership team would not take.")
    if task_id == TaskId.TASK2:
        hints.append("Do not exceed sprint_budget.")
        hints.append("Use beneficiary_account_ids, beneficiary_segments, implementation_risk, dependencies, conflicts, and stakeholder_notes to form one coherent portfolio bet.")
        hints.append("If multiple IDs share the same initiative_family or the same prefix before the trailing numeric suffix, treat them as alternative variants and usually choose at most one.")
        hints.append("Unless visible conflicts force otherwise, spend close to sprint_budget and prefer a full portfolio over a tiny under-spent bundle.")
    if task_id == TaskId.TASK3:
        hints.append("Respect budget_remaining and capacity_remaining.")
        hints.append("If the narrative or alerts change sharply, consider that the hidden goal may have shifted.")
        hints.append("Use sim_date, calendar_events, decision_ledger, pending_effects, realized_effects, and visible initiative dependencies/conflicts to reason over delayed consequences.")
        hints.append("Use accounts, stakeholders, teams, market_context, and governance_constraints to understand who is affected and what is risky.")
        hints.append("Use memory_workspace to recover older evidence, but keep the action payload short unless extra memory/belief fields clearly help.")
        hints.append("belief_report is optional in strict mode; if you include it, keep it minimal.")
        hints.append("Only backlog initiative IDs are actionable. Inbox or calendar IDs are context, never valid chosen_initiatives.")
        hints.append("Prefer decisions whose delayed effects stay coherent with the same latent objective over the next few dates.")
    if task_id == TaskId.TASK6:
        hints.append("Respect budget_remaining and capacity_remaining.")
        hints.append("Use memory_workspace to track unresolved incident evidence and conflicting stakeholder claims.")
        hints.append("Prefer a short valid control payload; omit focus, writes, and belief unless they materially help this step.")
        hints.append("Treat the incident as partially observed: false leads are possible and delayed effects matter.")
    if task_id == TaskId.TASK4:
        hints.append("Use integer budget points only and do not exceed sprint_budget.")
        hints.append("Prefer funding levels near visible saturation_point before spreading tiny allocations across too many programs.")
        hints.append("Use dependencies, conflicts, synergies, and stakeholder_notes to form one coherent capital narrative.")
    if task_id == TaskId.TASK5:
        hints.append("Respect both budget_remaining and capacity_remaining.")
        hints.append("Treat pricing, messaging, support policy, and initiatives as one package, not four unrelated choices.")
        hints.append("Avoid policies that visibly clash with enterprise renewals, premium support obligations, or margin pressure.")
    if task_id == TaskId.TASK7:
        hints.append("Use integer budget_allocations only and do not exceed budget_remaining.")
        hints.append("Track delayed staffing effects through memory_workspace, but keep the output minimal unless extra memory/belief fields are truly needed.")
        hints.append("Favor coherent hiring narratives over scattered one-off hires.")

    compact_observation = observation
    if task_id == TaskId.TASK1:
        compact_observation = _compact_task1_observation(observation)
    elif task_id == TaskId.TASK2:
        compact_observation = _compact_task2_observation(observation)
    elif task_id == TaskId.TASK3:
        compact_observation = _compact_task3_observation(observation)
    elif task_id == TaskId.TASK6:
        compact_observation = _compact_task3_observation(observation)
    elif task_id == TaskId.TASK4:
        compact_observation = _compact_task4_observation(observation)
    elif task_id == TaskId.TASK7:
        compact_observation = _compact_task7_observation(observation)
    elif task_id == TaskId.TASK5:
        compact_observation = _compact_task5_observation(observation)

    prompt_payload = {
        "decision_hints": hints,
        "allowed_item_ids": _allowed_ids(observation),
        "output_template": _json_template(task_id),
        "observation": compact_observation,
    }
    if policy_mode == "synthetic_operator" and operator_profile is not None:
        prompt_payload["operator_profile"] = {
            "persona_id": operator_profile.persona_id,
            "label": operator_profile.label,
            "risk_posture": operator_profile.risk_posture,
            "change_tolerance": operator_profile.change_tolerance,
            "default_focus": operator_profile.default_focus,
            "max_parallel_bets": operator_profile.max_parallel_bets,
            "max_parallel_initiatives": operator_profile.max_parallel_initiatives,
            "max_escalations": operator_profile.max_escalations,
            "max_abs_pricing_change": operator_profile.max_abs_pricing_change,
        }
    return json.dumps(prompt_payload, sort_keys=True, separators=(",", ":"))
