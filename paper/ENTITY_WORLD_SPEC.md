# Entity World Spec

This document summarizes the persistent entity layer added to LatentGoalOps to make the benchmark more realistic and more suitable for paper-level trajectory analysis.

## Design Goal

The benchmark should not feel like a sequence of unrelated prompts. It should feel like one startup world with recurring customers, recurring stakeholders, dated decisions, delayed consequences, and economic tradeoffs.

## Entity Families

### CustomerAccount

Visible fields include:

- `account_id`
- `company_name`
- `segment`
- `industry`
- `region`
- `seat_count`
- `annual_contract_value`
- `expansion_potential`
- `renewal_window_days`
- `support_tier`
- `sla_level`
- `security_sensitivity`
- `integration_complexity`
- `adoption_stage`
- `relationship_health`
- `champion_strength`
- `decision_maker_involved`
- `churn_propensity`
- `communication_style`
- `influence_weight`
- `strategic_importance`

Why it matters:

- Task 1 becomes economically grounded feedback triage rather than text-only labeling.
- Task 2 can connect initiatives to specific beneficiary accounts and segments.
- Task 3 can model renewals, trust erosion, and account-level delayed effects.

### StakeholderPersona

Visible fields include:

- `persona_id`
- `name`
- `role`
- `agenda_bias`
- `risk_tolerance`
- `time_horizon`
- `communication_style`
- `political_power`
- `credibility`
- `patience`
- `favorite_metrics`

Why it matters:

- The agent must reason about conflicting internal agendas instead of a single synthetic narrator.
- Stakeholder notes can provide noisy but structured evidence about the latent objective.

### InternalTeamState

Visible fields include:

- `team_id`
- `function`
- `capacity`
- `burnout_risk`
- `execution_reliability`
- `specialization`
- `cross_team_friction`
- `delivery_latency_days`

Why it matters:

- Actions now have organizational consequences.
- The environment can represent execution bottlenecks instead of assuming strategy is free.

### MarketContext

Visible fields include:

- `funding_stage`
- `cash_runway_months`
- `gross_margin_target`
- `board_pressure_level`
- `competition_intensity`
- `seasonality`
- `compliance_exposure`
- `strategic_accounts_share`
- `sales_pipeline_health`

Why it matters:

- The latent objective is embedded inside a plausible business regime.
- The agent can use board pressure and runway signals as strategic context.

### GovernanceConstraint

Visible fields include:

- `constraint_id`
- `title`
- `description`
- `severity`
- `affected_channels`
- `threshold`
- `penalty_hint`

Current guardrails:

- pricing near strategic renewals
- SLA handling for premium / strategic accounts
- growth spend under margin stress

## Task-Specific Use

### Task 1

- feedback messages are emitted from concrete customer accounts
- message tone is persona-conditioned
- escalation is influenced by ACV, renewals, support tier, and strategic importance

### Task 2

- backlog items expose beneficiary segments and linked accounts
- stakeholder notes come from persistent named personas
- initiatives expose policy tags and implementation risk

### Task 3

- dated observations expose accounts, stakeholders, teams, market context, and governance constraints
- decisions update account health, churn risk, team burnout, and board pressure
- renewal windows decrement over time
- renewals and renewal scares create realized dated business outcomes
- governance violations schedule future damage rather than only immediate penalties
- temporal effect records can point to affected account IDs and team IDs for auditability

### Task 4

- budget allocations are made over visible programs rather than binary pick / skip choices
- beneficiary accounts, saturation hints, and portfolio conflicts make capital deployment path-dependent
- stakeholder notes create indirect pressure without directly naming the objective

### Task 5

- one-shot crisis packages combine initiatives with pricing, messaging, and support policy
- high-touch accounts and governance constraints make mixed policy actions visibly risky
- the same visible crisis can support different defensible packages depending on the latent objective

## Held-out Scenario Split

- strategic tasks now support a held-out scenario family split with alternate initiative and event banks
- this split is meant to test transfer across unseen operating situations rather than pure prompt memorization
- it is not a full distribution shift benchmark, but it gives the paper a stronger OOD story than a single fixed scenario family

## Research Value

This layer improves the benchmark in four ways:

1. Stronger realism: decisions affect customers, teams, and board context rather than only abstract KPIs.
2. Better causal analysis: delayed effects can be tied back to entities and dates.
3. Better interpretability: case studies can reference concrete accounts and stakeholders.
4. Better novelty: the environment is no longer just an instruction-following simulator, but a synthetic operating world with latent business objectives and non-stationary dynamics.
