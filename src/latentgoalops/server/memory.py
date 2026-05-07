"""Structured deterministic memory bank for multi-step tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from latentgoalops.models import (
    CompanyProfile,
    CustomerAccount,
    GovernanceConstraint,
    InternalTeamState,
    MemoryEntityRef,
    MemoryFocusRequest,
    MemoryRecord,
    MemoryWorkspace,
    MemoryWrite,
    StakeholderPersona,
)


DEFAULT_RETRIEVAL_BUDGET = 6
DEFAULT_WRITE_BUDGET = 2
DEFAULT_PRIVATE_NOTE_LIMIT = 40


def _record_id(prefix: str, step_index: int, counter: int) -> str:
    return f"{prefix}_{step_index}_{counter}"


def _entity_ids(record: MemoryRecord) -> set[str]:
    return {ref.entity_id for ref in record.entity_refs}


def _append_record(bank: dict, record: MemoryRecord) -> None:
    bank["records"].append(record)
    bank["records_by_id"][record.record_id] = record
    for ref in record.entity_refs:
        bank["records_by_entity"][ref.entity_id].append(record.record_id)
        bank["entity_timelines"][ref.entity_id].append(record.record_id)
    for tag in record.tags:
        bank["records_by_tag"][tag].append(record.record_id)
    if record.conflict_group:
        bank["conflict_groups"][record.conflict_group].append(record.record_id)


def _new_record(
    bank: dict,
    *,
    prefix: str,
    step_index: int,
    sim_date: str | None,
    source_type: str,
    summary: str,
    detail: str | None = None,
    entity_refs: list[MemoryEntityRef] | None = None,
    tags: list[str] | None = None,
    salience: float = 0.5,
    confidence: float | None = None,
    conflict_group: str | None = None,
    private_to_agent: bool = False,
) -> MemoryRecord:
    counter = bank["next_record_index"]
    bank["next_record_index"] += 1
    return MemoryRecord(
        record_id=_record_id(prefix, step_index, counter),
        step_index=step_index,
        sim_date=sim_date,
        source_type=source_type,
        summary=summary,
        detail=detail,
        entity_refs=entity_refs or [],
        tags=tags or [],
        salience=salience,
        confidence=confidence,
        conflict_group=conflict_group,
        private_to_agent=private_to_agent,
    )


def initialize_memory_bank(
    world: dict,
    task_id: str,
    sim_date: str | None,
    retrieval_budget: int = DEFAULT_RETRIEVAL_BUDGET,
    write_budget: int = DEFAULT_WRITE_BUDGET,
    private_note_limit: int = DEFAULT_PRIVATE_NOTE_LIMIT,
) -> dict:
    """Create a seeded memory bank for a world."""
    bank = {
        "task_id": task_id,
        "records": [],
        "records_by_id": {},
        "records_by_entity": defaultdict(list),
        "records_by_tag": defaultdict(list),
        "entity_timelines": defaultdict(list),
        "conflict_groups": defaultdict(list),
        "retrieval_budget_per_step": retrieval_budget,
        "write_budget_per_step": write_budget,
        "private_note_limit": private_note_limit,
        "retrieval_budget_remaining": retrieval_budget,
        "write_budget_remaining": write_budget,
        "next_record_index": 0,
        "last_focus": [],
    }
    profile: CompanyProfile = world["company_profile"]
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="world",
            step_index=0,
            sim_date=sim_date,
            source_type="world_seed",
            summary=(
                f"{profile.company_name} is a {profile.product_type} company selling through {profile.gtm_motion} "
                f"with {profile.pricing_model} pricing."
            ),
            detail=(
                f"Core workflow: {profile.core_user_workflow}. ICP: {profile.ideal_customer_profile}. "
                f"Compliance: {profile.compliance_regime}. Board narrative: {profile.board_narrative}."
            ),
            entity_refs=[MemoryEntityRef(entity_type="company", entity_id=profile.company_id)],
            tags=["company_profile", profile.seed_family, profile.sector],
            salience=0.95,
        ),
    )
    for account in world["accounts"][:6]:
        _append_record(
            bank,
            _new_record(
                bank,
                prefix="acct",
                step_index=0,
                sim_date=sim_date,
                source_type="world_seed",
                summary=(
                    f"{account.company_name} is a {account.segment} account with {account.annual_contract_value:,.0f} ACV, "
                    f"renewal window {account.renewal_window_days} days, and relationship health {account.relationship_health:.2f}."
                ),
                detail=(
                    f"Support tier {account.support_tier}, security sensitivity {account.security_sensitivity:.2f}, "
                    f"expansion potential {account.expansion_potential:.2f}."
                ),
                entity_refs=[
                    MemoryEntityRef(entity_type="company", entity_id=profile.company_id),
                    MemoryEntityRef(entity_type="account", entity_id=account.account_id),
                ],
                tags=["account_profile", account.segment, account.industry],
                salience=0.72 if account.renewal_window_days <= 30 else 0.58,
            ),
        )
    for stakeholder in world["stakeholders"]:
        conflict_group = None
        if stakeholder.role in {"CFO", "Head of CS"}:
            conflict_group = "pricing_vs_trust"
        elif stakeholder.role in {"CTO", "Growth Lead"}:
            conflict_group = "speed_vs_reliability"
        _append_record(
            bank,
            _new_record(
                bank,
                prefix="stakeholder",
                step_index=0,
                sim_date=sim_date,
                source_type="stakeholder_note",
                summary=f"{stakeholder.name} ({stakeholder.role}) keeps returning to {', '.join(stakeholder.favorite_metrics[:2])}.",
                detail=" ".join(stakeholder.soft_preferences + stakeholder.hard_constraints) or None,
                entity_refs=[
                    MemoryEntityRef(entity_type="company", entity_id=profile.company_id),
                    MemoryEntityRef(entity_type="stakeholder", entity_id=stakeholder.persona_id),
                ],
                tags=["stakeholder", stakeholder.role.lower().replace(" ", "_")],
                salience=0.60 + stakeholder.political_power * 0.20,
                conflict_group=conflict_group,
            ),
        )
    for team in world["teams"]:
        _append_record(
            bank,
            _new_record(
                bank,
                prefix="team",
                step_index=0,
                sim_date=sim_date,
                source_type="team_update",
                summary=(
                    f"{team.function.title()} team capacity {team.capacity:.2f}, burnout {team.burnout_risk:.2f}, "
                    f"execution reliability {team.execution_reliability:.2f}."
                ),
                entity_refs=[
                    MemoryEntityRef(entity_type="company", entity_id=profile.company_id),
                    MemoryEntityRef(entity_type="team", entity_id=team.team_id),
                ],
                tags=["team_state", team.function],
                salience=0.54,
            ),
        )
    for constraint in world["governance_constraints"]:
        _append_record(
            bank,
            _new_record(
                bank,
                prefix="constraint",
                step_index=0,
                sim_date=sim_date,
                source_type="governance",
                summary=f"{constraint.title}: {constraint.description}",
                detail=constraint.penalty_hint,
                entity_refs=[MemoryEntityRef(entity_type="constraint", entity_id=constraint.constraint_id)],
                tags=["governance", constraint.severity, *constraint.affected_channels],
                salience=0.76 if constraint.severity == "high" else 0.62,
            ),
        )
    return bank


def reset_step_budgets(bank: dict) -> None:
    """Refresh step-level retrieval and write budgets."""
    bank["retrieval_budget_remaining"] = int(bank["retrieval_budget_per_step"])
    bank["write_budget_remaining"] = int(bank["write_budget_per_step"])


def record_event(
    bank: dict,
    *,
    step_index: int,
    sim_date: str | None,
    event_id: str,
    summary: str,
    alerts: list[str],
) -> None:
    """Store a scheduled or surfaced event."""
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="event",
            step_index=step_index,
            sim_date=sim_date,
            source_type="event",
            summary=summary,
            detail=" ".join(alerts) or None,
            entity_refs=[MemoryEntityRef(entity_type="event", entity_id=event_id)],
            tags=["event", "external_shock"],
            salience=0.82,
        ),
    )


def record_decision(
    bank: dict,
    *,
    step_index: int,
    sim_date: str | None,
    decision_id: str,
    summary: str,
    chosen_initiatives: list[str],
    governance_flags: list[str],
    company_id: str | None = None,
) -> None:
    """Store a human-readable action summary."""
    entity_refs = [MemoryEntityRef(entity_type="task", entity_id=decision_id)]
    if company_id:
        entity_refs.append(MemoryEntityRef(entity_type="company", entity_id=company_id))
    entity_refs.extend(MemoryEntityRef(entity_type="initiative", entity_id=item_id) for item_id in chosen_initiatives)
    tags = ["decision"]
    if governance_flags:
        tags.append("governance_flagged")
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="decision",
            step_index=step_index,
            sim_date=sim_date,
            source_type="decision",
            summary=summary,
            detail="; ".join(governance_flags) or None,
            entity_refs=entity_refs,
            tags=tags,
            salience=0.78,
        ),
    )


def record_effect(
    bank: dict,
    *,
    step_index: int,
    sim_date: str | None,
    effect_id: str,
    summary: str,
    account_ids: list[str],
    team_ids: list[str],
    source_id: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Store a realized or pending effect."""
    entity_refs = [MemoryEntityRef(entity_type="event", entity_id=effect_id)]
    if source_id:
        entity_refs.append(MemoryEntityRef(entity_type="initiative", entity_id=source_id))
    entity_refs.extend(MemoryEntityRef(entity_type="account", entity_id=account_id) for account_id in account_ids)
    entity_refs.extend(MemoryEntityRef(entity_type="team", entity_id=team_id) for team_id in team_ids)
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="effect",
            step_index=step_index,
            sim_date=sim_date,
            source_type="effect",
            summary=summary,
            entity_refs=entity_refs,
            tags=["effect", *(tags or [])],
            salience=0.83,
        ),
    )


def record_account_update(
    bank: dict,
    *,
    step_index: int,
    sim_date: str | None,
    account: CustomerAccount,
    summary: str,
    tags: list[str] | None = None,
    conflict_group: str | None = None,
) -> None:
    """Store a point-in-time account state change."""
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="account_update",
            step_index=step_index,
            sim_date=sim_date,
            source_type="account_update",
            summary=summary,
            detail=(
                f"Renewal {account.renewal_window_days} days, relationship {account.relationship_health:.2f}, "
                f"churn propensity {account.churn_propensity:.2f}."
            ),
            entity_refs=[MemoryEntityRef(entity_type="account", entity_id=account.account_id)],
            tags=["account_update", account.segment, *(tags or [])],
            salience=0.80,
            conflict_group=conflict_group,
        ),
    )


def record_team_update(
    bank: dict,
    *,
    step_index: int,
    sim_date: str | None,
    team: InternalTeamState,
    summary: str,
    tags: list[str] | None = None,
) -> None:
    """Store a team-capacity or burnout update."""
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="team_update",
            step_index=step_index,
            sim_date=sim_date,
            source_type="team_update",
            summary=summary,
            entity_refs=[MemoryEntityRef(entity_type="team", entity_id=team.team_id)],
            tags=["team_update", team.function, *(tags or [])],
            salience=0.63,
        ),
    )


def mark_conflict_resolved(bank: dict, conflict_group: str, resolution_summary: str, step_index: int, sim_date: str | None) -> None:
    """Close a conflict group after visible evidence resolves it."""
    record_ids = bank["conflict_groups"].get(conflict_group, [])
    for record_id in record_ids:
        bank["records_by_id"][record_id].resolved = True
        if bank["records_by_id"][record_id].source_type == "stakeholder_note":
            bank["records_by_id"][record_id].stale = True
    _append_record(
        bank,
        _new_record(
            bank,
            prefix="resolution",
            step_index=step_index,
            sim_date=sim_date,
            source_type="company_update",
            summary=resolution_summary,
            tags=["conflict_resolution", conflict_group],
            salience=0.78,
            conflict_group=conflict_group,
        ),
    )


def apply_agent_writes(bank: dict, writes: list[MemoryWrite], step_index: int, sim_date: str | None) -> list[str]:
    """Persist a bounded number of agent-authored notes."""
    accepted: list[str] = []
    private_count = sum(1 for record in bank["records"] if record.private_to_agent)
    for write in writes[: bank["write_budget_remaining"]]:
        if private_count >= bank["private_note_limit"]:
            break
        record = _new_record(
            bank,
            prefix="agent_note",
            step_index=step_index,
            sim_date=sim_date,
            source_type="agent_note",
            summary=write.title,
            detail=write.body,
            entity_refs=write.entity_refs,
            tags=["agent_note", *write.tags],
            salience=0.55,
            confidence=write.confidence,
            private_to_agent=True,
        )
        _append_record(bank, record)
        accepted.append(record.record_id)
        bank["write_budget_remaining"] = max(0, int(bank["write_budget_remaining"]) - 1)
        private_count += 1
    return accepted


def _score_record(
    record: MemoryRecord,
    *,
    focus_entity_ids: set[str],
    focus_tags: set[str],
    current_step: int,
    include_private: bool,
    include_conflicts: bool,
) -> float:
    if record.private_to_agent and not include_private:
        return -1.0
    score = record.salience
    if focus_entity_ids:
        overlap = len(_entity_ids(record).intersection(focus_entity_ids))
        score += overlap * 1.8
    if focus_tags:
        score += len(set(record.tags).intersection(focus_tags)) * 1.2
    recency = max(0, current_step - record.step_index)
    score += max(0.0, 1.0 - recency / 8.0) * 0.8
    if record.private_to_agent:
        score += 0.45
    if record.conflict_group and not record.resolved:
        score += 0.6 if include_conflicts else -0.8
    if record.stale:
        score -= 0.35
    return score


def _entity_timeline(bank: dict, entity_id: str, limit: int = 3) -> list[str]:
    record_ids = bank["entity_timelines"].get(entity_id, [])
    return record_ids[-limit:]


def build_memory_workspace(
    bank: dict,
    *,
    current_step: int,
    focus_requests: list[MemoryFocusRequest] | None = None,
    visible_entity_ids: Iterable[str] | None = None,
    visible_tags: Iterable[str] | None = None,
) -> MemoryWorkspace:
    """Deterministically retrieve a bounded memory slice."""
    requests = focus_requests or []
    bank["last_focus"] = [request.model_dump(mode="json") for request in requests]
    focus_entity_ids = set(visible_entity_ids or [])
    focus_tags = set(visible_tags or [])
    include_private = True
    include_conflicts = True
    max_lookback = None
    for request in requests:
        focus_entity_ids.update(request.entity_ids)
        focus_tags.update(request.tags)
        include_private = include_private and request.include_private_notes
        include_conflicts = include_conflicts or request.include_conflicts
        if request.lookback_steps is not None:
            max_lookback = request.lookback_steps if max_lookback is None else max(max_lookback, request.lookback_steps)

    candidates: list[tuple[float, MemoryRecord]] = []
    for record in bank["records"]:
        if max_lookback is not None and current_step - record.step_index > max_lookback:
            continue
        score = _score_record(
            record,
            focus_entity_ids=focus_entity_ids,
            focus_tags=focus_tags,
            current_step=current_step,
            include_private=include_private,
            include_conflicts=include_conflicts,
        )
        if score >= 0.0:
            candidates.append((score, record))
    candidates.sort(key=lambda row: (-row[0], -row[1].step_index, row[1].record_id))
    limit = max(0, int(bank["retrieval_budget_remaining"]))
    selected = [record.model_copy(deep=True) for _, record in candidates[:limit]]
    consumed = len(selected)
    bank["retrieval_budget_remaining"] = max(0, int(bank["retrieval_budget_remaining"]) - consumed)
    timeline_entity_ids = list(focus_entity_ids)[:4]
    entity_timelines = {
        entity_id: _entity_timeline(bank, entity_id)
        for entity_id in timeline_entity_ids
    }
    open_conflicts = [
        group
        for group, record_ids in sorted(bank["conflict_groups"].items())
        if any(not bank["records_by_id"][record_id].resolved for record_id in record_ids)
    ]
    return MemoryWorkspace(
        records=selected,
        entity_timelines=entity_timelines,
        open_conflicts=open_conflicts,
        retrieval_budget_remaining=int(bank["retrieval_budget_remaining"]),
        write_budget_remaining=int(bank["write_budget_remaining"]),
    )


def auto_visible_entities(
    *,
    company_profile: CompanyProfile | None,
    accounts: list[CustomerAccount] | None = None,
    stakeholders: list[StakeholderPersona] | None = None,
    teams: list[InternalTeamState] | None = None,
    alerts: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Build a default retrieval context from the visible state."""
    entity_ids: list[str] = []
    if company_profile is not None:
        entity_ids.append(company_profile.company_id)
    entity_ids.extend(account.account_id for account in (accounts or [])[:4])
    entity_ids.extend(stakeholder.persona_id for stakeholder in (stakeholders or [])[:2])
    entity_ids.extend(team.team_id for team in (teams or [])[:2])
    tags = ["visible_context"]
    if alerts:
        lowered = " ".join(alerts).lower()
        if "renewal" in lowered or "trust" in lowered:
            tags.append("retention")
        if "margin" in lowered or "infra" in lowered or "cost" in lowered:
            tags.append("efficiency")
        if "board" in lowered or "pricing" in lowered or "commercial" in lowered:
            tags.append("revenue")
        if "competition" in lowered or "activation" in lowered or "pipeline" in lowered:
            tags.append("growth")
    return entity_ids, tags
