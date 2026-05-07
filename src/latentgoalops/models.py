"""Public Pydantic models for the LatentGoalOps OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


class GoalArchetype(str, Enum):
    """Hidden objective archetypes."""

    GROWTH = "growth"
    RETENTION = "retention"
    REVENUE = "revenue"
    EFFICIENCY = "efficiency"


class RiskPosture(str, Enum):
    """Latent risk tolerance modes."""

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class PlanningHorizon(str, Enum):
    """Latent planning horizon modes."""

    IMMEDIATE = "immediate"
    QUARTERLY = "quarterly"
    STRATEGIC = "strategic"


class GovernanceStrictness(str, Enum):
    """Latent governance tolerance modes."""

    FLEXIBLE = "flexible"
    MODERATE = "moderate"
    STRICT = "strict"


class FeedbackLabel(str, Enum):
    """Task 1 feedback categories."""

    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    CHURN_RISK = "churn_risk"
    PRAISE = "praise"
    BILLING_ISSUE = "billing_issue"
    LATENCY_COMPLAINT = "latency_complaint"


class MessagingAction(str, Enum):
    """Task 3 narrative messaging levers."""

    GROWTH_PUSH = "growth_push"
    RETENTION_CAMPAIGN = "retention_campaign"
    REVENUE_UPSELL = "revenue_upsell"
    COST_COMMS = "cost_comms"
    NONE = "none"


class SupportPolicy(str, Enum):
    """Task 3 support operating policies."""

    PREMIUM_SLA = "premium_sla"
    BALANCED_TRIAGE = "balanced_triage"
    AUTOMATION_FIRST = "automation_first"
    INCIDENT_SWARM = "incident_swarm"


class TaskId(str, Enum):
    """Available task identifiers."""

    TASK1 = "task1_feedback_triage"
    TASK2 = "task2_roadmap_priority"
    TASK3 = "task3_startup_week"
    TASK4 = "task4_capital_allocation"
    TASK5 = "task5_crisis_response"
    TASK6 = "task6_incident_response_week"
    TASK7 = "task7_quarterly_headcount_plan"


class DashboardState(BaseModel):
    """Visible KPI dashboard used across tasks."""

    dau: float = Field(default=0.0)
    mau: float = Field(default=0.0)
    d7_retention: float = Field(default=0.0)
    d30_retention: float = Field(default=0.0)
    mrr: float = Field(default=0.0)
    arpu: float = Field(default=0.0)
    cac: float = Field(default=0.0)
    churn_rate: float = Field(default=0.0)
    ops_margin: float = Field(default=0.0)
    infra_cost_per_unit: float = Field(default=0.0)
    support_ticket_volume: int = Field(default=0)

    def to_metric_vector(self) -> dict[str, float]:
        """Map raw dashboard values into the four latent utility channels."""
        growth = max(0.0, min(1.0, (self.dau / max(self.mau, 1.0))))
        retention = max(0.0, min(1.0, (self.d30_retention + self.d7_retention) / 2))
        revenue = max(0.0, min(1.0, (self.mrr / 100000.0) * 0.7 + self.arpu / 200.0 * 0.3))
        efficiency = max(
            0.0,
            min(1.0, (self.ops_margin + (1.0 - min(self.infra_cost_per_unit / 5.0, 1.0))) / 2),
        )
        return {
            "growth": growth,
            "retention": retention,
            "revenue": revenue,
            "efficiency": efficiency,
        }


class InboxItem(BaseModel):
    """Generic message item for task inboxes."""

    item_id: str
    text: str
    sender: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InitiativeItem(BaseModel):
    """Roadmap or operational initiative visible to the agent."""

    item_id: str
    title: str
    description: str
    cost: float
    kind: str
    kpi_deltas: dict[str, float]
    uncertainty_band: float = 0.0
    stakeholder_tag: str = "ops"
    lag_steps: int = 1
    effect_window: int = 1
    delivery_note: str | None = None
    beneficiary_segments: list[str] = Field(default_factory=list)
    beneficiary_account_ids: list[str] = Field(default_factory=list)
    implementation_risk: float = 0.0
    policy_tags: list[str] = Field(default_factory=list)
    requires_item_ids: list[str] = Field(default_factory=list)
    conflicts_with_ids: list[str] = Field(default_factory=list)
    synergy_item_ids: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    allocation_unit: float | None = None
    allocation_max: float | None = None
    saturation_point: float | None = None
    impact_summary: str | None = None


class CustomerAccount(BaseModel):
    """Visible synthetic customer profile used across tasks."""

    account_id: str
    company_name: str
    segment: str
    industry: str
    region: str
    seat_count: int
    annual_contract_value: float
    expansion_potential: float
    renewal_window_days: int
    payment_risk: float
    support_tier: str
    sla_level: str
    security_sensitivity: float
    integration_complexity: float
    adoption_stage: str
    relationship_health: float
    champion_strength: float
    decision_maker_involved: bool
    churn_propensity: float
    communication_style: str
    influence_weight: float
    strategic_importance: float


class StakeholderPersona(BaseModel):
    """Internal or board stakeholder with a recurring agenda."""

    persona_id: str
    name: str
    role: str
    agenda_bias: str
    risk_tolerance: float
    time_horizon: str
    communication_style: str
    political_power: float
    credibility: float
    patience: float
    favorite_metrics: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)


class InternalTeamState(BaseModel):
    """Visible operating state for a functional team."""

    team_id: str
    function: str
    capacity: float
    burnout_risk: float
    execution_reliability: float
    specialization: list[str] = Field(default_factory=list)
    cross_team_friction: float = 0.0
    delivery_latency_days: int = 1


class MarketContext(BaseModel):
    """Visible macro / company context for the startup world."""

    funding_stage: str
    cash_runway_months: int
    gross_margin_target: float
    board_pressure_level: float
    competition_intensity: float
    seasonality: str
    compliance_exposure: float
    strategic_accounts_share: float
    sales_pipeline_health: float


class CompanyProfile(BaseModel):
    """Visible company archetype instantiated for a sampled world."""

    company_id: str
    company_name: str
    archetype_id: str
    product_type: str
    gtm_motion: str
    pricing_model: str
    compliance_regime: str
    deployment_model: str
    core_user_workflow: str
    ideal_customer_profile: str
    primary_buyer: str
    board_narrative: str
    margin_profile: str
    sector: str
    seed_family: str
    risk_envelope: str


class GovernanceConstraint(BaseModel):
    """Visible governance or policy rule that constrains decisions."""

    constraint_id: str
    title: str
    description: str
    severity: str
    affected_channels: list[str] = Field(default_factory=list)
    threshold: float | None = None
    penalty_hint: str | None = None


class TemporalEffectRecord(BaseModel):
    """Scheduled or realized delayed consequence in the startup simulator."""

    effect_id: str
    decision_id: str | None = None
    source_type: str
    source_id: str
    summary: str
    channel_deltas: dict[str, float] = Field(default_factory=dict)
    dashboard_deltas: dict[str, float] = Field(default_factory=dict)
    affected_account_ids: list[str] = Field(default_factory=list)
    affected_team_ids: list[str] = Field(default_factory=list)
    scheduled_for_step: int
    scheduled_for_date: str
    realized_step: int | None = None
    realized_date: str | None = None


class DecisionLedgerEntry(BaseModel):
    """Human-readable decision record for Task 3 timeline analysis."""

    decision_id: str
    step_index: int
    sim_date: str
    chosen_initiatives: list[str] = Field(default_factory=list)
    messaging_action: MessagingAction | None = None
    pricing_change_pct: float | None = None
    support_policy: SupportPolicy | None = None
    rationale: str | None = None
    expected_channels: dict[str, float] = Field(default_factory=dict)
    scheduled_effect_ids: list[str] = Field(default_factory=list)
    realized_effect_ids: list[str] = Field(default_factory=list)
    observed_alerts: list[str] = Field(default_factory=list)
    governance_flags: list[str] = Field(default_factory=list)


class PublicInitiativeItem(BaseModel):
    """Redacted initiative surface shown to agents and human evaluators."""

    item_id: str
    title: str
    description: str
    cost: float
    uncertainty_band: float = 0.0
    stakeholder_tag: str = "ops"
    lag_steps: int = 1
    effect_window: int = 1
    delivery_note: str | None = None
    beneficiary_segments: list[str] = Field(default_factory=list)
    beneficiary_account_ids: list[str] = Field(default_factory=list)
    implementation_risk: float = 0.0
    policy_tags: list[str] = Field(default_factory=list)
    requires_item_ids: list[str] = Field(default_factory=list)
    conflicts_with_ids: list[str] = Field(default_factory=list)
    synergy_item_ids: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    allocation_unit: float | None = None
    allocation_max: float | None = None
    saturation_point: float | None = None
    impact_summary: str | None = None


class PublicTemporalEffectRecord(BaseModel):
    """Redacted temporal consequence record for public observations."""

    effect_id: str
    decision_id: str | None = None
    source_type: str
    source_id: str
    summary: str
    affected_account_ids: list[str] = Field(default_factory=list)
    affected_team_ids: list[str] = Field(default_factory=list)
    scheduled_for_step: int
    scheduled_for_date: str
    realized_step: int | None = None
    realized_date: str | None = None


class PublicDecisionLedgerEntry(BaseModel):
    """Redacted decision record exposed in public observations."""

    decision_id: str
    step_index: int
    sim_date: str
    chosen_initiatives: list[str] = Field(default_factory=list)
    messaging_action: MessagingAction | None = None
    pricing_change_pct: float | None = None
    support_policy: SupportPolicy | None = None
    rationale: str | None = None
    scheduled_effect_ids: list[str] = Field(default_factory=list)
    realized_effect_ids: list[str] = Field(default_factory=list)
    observed_alerts: list[str] = Field(default_factory=list)
    governance_flags: list[str] = Field(default_factory=list)


class SimCalendarEvent(BaseModel):
    """Visible calendar marker for the startup simulator."""

    event_id: str
    name: str
    sim_date: str
    status: Literal["today", "upcoming"]
    summary: str
    alerts: list[str] = Field(default_factory=list)


class MemoryEntityRef(BaseModel):
    """Typed entity handle referenced by memory records."""

    entity_type: Literal[
        "company",
        "account",
        "stakeholder",
        "team",
        "initiative",
        "event",
        "constraint",
        "task",
    ]
    entity_id: str
    role: str | None = None


class MemoryRelationRef(BaseModel):
    """Optional relation edge between referenced entities."""

    subject_id: str
    predicate: str
    object_id: str


class MemoryRecord(BaseModel):
    """Append-only memory record exposed through the memory bank."""

    record_id: str
    step_index: int
    sim_date: str | None = None
    source_type: Literal[
        "world_seed",
        "decision",
        "effect",
        "event",
        "stakeholder_note",
        "governance",
        "account_update",
        "team_update",
        "company_update",
        "agent_note",
    ]
    summary: str
    detail: str | None = None
    entity_refs: list[MemoryEntityRef] = Field(default_factory=list)
    relation_refs: list[MemoryRelationRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    salience: float = 0.5
    confidence: float | None = None
    stale: bool = False
    conflict_group: str | None = None
    resolved: bool = False
    private_to_agent: bool = False


class MemoryFocusRequest(BaseModel):
    """Agent request for the next memory retrieval bundle."""

    entity_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    lookback_steps: int | None = None
    include_conflicts: bool = True
    include_private_notes: bool = True


class MemoryWrite(BaseModel):
    """Short agent-authored note stored in the private memory bank."""

    title: str
    body: str
    entity_refs: list[MemoryEntityRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float | None = None


class MemoryWorkspace(BaseModel):
    """Retrieved memory slice returned to the agent for the current step."""

    records: list[MemoryRecord] = Field(default_factory=list)
    entity_timelines: dict[str, list[str]] = Field(default_factory=dict)
    open_conflicts: list[str] = Field(default_factory=list)
    retrieval_budget_remaining: int = 0
    write_budget_remaining: int = 0


class BeliefReport(BaseModel):
    """Optional factorized belief report emitted by the agent."""

    archetype_probs: dict[str, float] = Field(default_factory=dict)
    risk_posture_probs: dict[str, float] = Field(default_factory=dict)
    planning_horizon_probs: dict[str, float] = Field(default_factory=dict)
    segment_focus_probs: dict[str, float] = Field(default_factory=dict)
    governance_strictness_probs: dict[str, float] = Field(default_factory=dict)
    shift_detected_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None


class ItemLabelAssignment(BaseModel):
    """Task 1 label assignment."""

    item_id: str
    label: FeedbackLabel


class ItemPriorityAssignment(BaseModel):
    """Task 1 priority assignment."""

    item_id: str
    priority: int = Field(ge=1, le=5)


class TaskDescriptor(BaseModel):
    """Human-readable task card."""

    task_id: TaskId
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    horizon: int


class GraderResult(BaseModel):
    """Programmatic task score with sub-metrics."""

    task_id: TaskId
    score: float = Field(ge=0.0, le=1.0)
    sub_scores: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class LatentGoalOpsAction(Action):
    """Unified action model covering all benchmark tasks."""

    task_id: TaskId
    labels: list[ItemLabelAssignment] = Field(default_factory=list)
    priorities: list[ItemPriorityAssignment] = Field(default_factory=list)
    escalate_ids: list[str] = Field(default_factory=list)
    selected_item_ids: list[str] = Field(default_factory=list)
    chosen_initiatives: list[str] = Field(default_factory=list)
    budget_allocations: dict[str, float] = Field(default_factory=dict)
    messaging_action: MessagingAction | None = Field(default=None)
    pricing_change_pct: float | None = Field(default=None, ge=-0.20, le=0.20)
    support_policy: SupportPolicy | None = Field(default=None)
    rationale_summary: str | None = None
    rationale: str | None = None
    memory_focus: list[MemoryFocusRequest] = Field(default_factory=list)
    memory_writes: list[MemoryWrite] = Field(default_factory=list)
    belief_report: BeliefReport | None = None

    @model_validator(mode="after")
    def validate_task_payload(self) -> "LatentGoalOpsAction":
        """Prevent cross-task field leakage."""
        if self.task_id == TaskId.TASK1:
            if self.selected_item_ids or self.chosen_initiatives or self.budget_allocations:
                raise ValueError("Task 1 actions cannot include roadmap or startup-week selections.")
        elif self.task_id == TaskId.TASK2:
            if self.labels or self.priorities or self.escalate_ids or self.budget_allocations:
                raise ValueError("Task 2 actions cannot include inbox triage fields.")
        elif self.task_id == TaskId.TASK3:
            if self.labels or self.priorities:
                raise ValueError("Task 3 actions cannot include Task 1 labeling fields.")
        elif self.task_id == TaskId.TASK4:
            if (
                self.labels
                or self.priorities
                or self.escalate_ids
                or self.selected_item_ids
                or self.chosen_initiatives
                or self.messaging_action is not None
                or self.pricing_change_pct is not None
                or self.support_policy is not None
            ):
                raise ValueError("Task 4 actions only support budget_allocations and an optional rationale_summary.")
        elif self.task_id == TaskId.TASK6:
            if self.labels or self.priorities or self.selected_item_ids or self.budget_allocations:
                raise ValueError("Task 6 actions only support crisis package controls plus optional memory/belief fields.")
        elif self.task_id == TaskId.TASK7:
            if (
                self.labels
                or self.priorities
                or self.escalate_ids
                or self.selected_item_ids
                or self.chosen_initiatives
                or self.messaging_action is not None
                or self.pricing_change_pct is not None
                or self.support_policy is not None
            ):
                raise ValueError("Task 7 actions only support budget_allocations and optional rationale/memory/belief fields.")
        elif self.task_id == TaskId.TASK5:
            if self.labels or self.priorities or self.escalate_ids or self.selected_item_ids or self.budget_allocations:
                raise ValueError("Task 5 actions cannot include Task 1, Task 2, or allocation-only fields.")
        return self


class LatentGoalOpsObservation(Observation):
    """Unified observation model covering all benchmark tasks."""

    task_id: TaskId
    step_index: int = 0
    horizon: int = 1
    sim_date: str | None = None
    sim_day_label: str | None = None
    task_summary: str = ""
    narrative: str | None = None
    dashboard: DashboardState = Field(default_factory=DashboardState)
    inbox: list[InboxItem] = Field(default_factory=list)
    backlog: list[PublicInitiativeItem] = Field(default_factory=list)
    accounts: list[CustomerAccount] = Field(default_factory=list)
    stakeholders: list[StakeholderPersona] = Field(default_factory=list)
    teams: list[InternalTeamState] = Field(default_factory=list)
    company_profile: CompanyProfile | None = None
    market_context: MarketContext | None = None
    governance_constraints: list[GovernanceConstraint] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    calendar_events: list[SimCalendarEvent] = Field(default_factory=list)
    decision_ledger: list[PublicDecisionLedgerEntry] = Field(default_factory=list)
    pending_effects: list[PublicTemporalEffectRecord] = Field(default_factory=list)
    realized_effects: list[PublicTemporalEffectRecord] = Field(default_factory=list)
    budget_remaining: float = 0.0
    capacity_remaining: float = 0.0
    sprint_budget: float | None = None
    stakeholder_notes: list[str] = Field(default_factory=list)
    memory_summary: str | None = None
    memory_workspace: MemoryWorkspace | None = None
    memory_budget_remaining: int = 0
    memory_write_budget_remaining: int = 0
    available_actions: list[str] = Field(default_factory=list)


class LatentGoalOpsState(State):
    """Public, non-leaking environment state for the /state endpoint."""

    task_id: TaskId | None = None
    max_steps: int = 0
    sim_date: str | None = None
    sim_day_label: str | None = None
    budget_remaining: float = 0.0
    capacity_remaining: float = 0.0
    decision_count: int = 0
    pending_effect_count: int = 0
    completed: bool = False
    cumulative_reward: float = 0.0
    last_score: float | None = None
