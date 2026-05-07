---
title: LatentGoalOps
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - benchmark
  - agents
---

# LatentGoalOps

LatentGoalOps is a research-grade OpenEnv benchmark for latent-goal inference and adaptation under non-stationary operational objectives. Instead of telling an agent what to optimize, the environment gives indirect clues through KPI dashboards, stakeholder messages, backlog options, and event noise. The agent must infer the business objective, choose coherent actions, and in the long-horizon tasks recover from a silent mid-episode goal shift.

This repo implements:

- Full OpenEnv-style `reset()`, `step()`, and `state()` support
- Typed Pydantic `Action`, `Observation`, and `State` models
- Seven tasks with deterministic programmatic graders
- Leakage-safe shaped rewards with partial progress signals
- A resumable baseline harness with OpenAI-client calls against OpenAI-compatible endpoints such as DigitalOcean Gradient
- A strict paper-eval mode that disables heuristic rescue and parse repair for clean model comparisons
- A root [`inference.py`](inference.py) submission runner for the required hackathon baseline flow
- Hugging Face Space metadata and Docker packaging

## Why This Is Real-World

The benchmark models tasks that real startup operators and product teams actually do:

- Task 1: triage customer feedback and escalations
- Task 2: prioritize a sprint roadmap under budget
- Task 3: run a startup for a week under noisy and shifting business constraints
- Task 4: allocate capital across nonlinear programs with diminishing returns
- Task 5: assemble a crisis-response package from initiatives plus operating policies
- Task 6: run an incident-response week with noisy evidence and delayed recovery effects
- Task 7: plan quarterly headcount under shifting operational priorities

This is not a toy game. The hidden goal changes how the same visible evidence should be interpreted, which makes the benchmark useful for agent evaluation rather than just schema following.

The latest version also adds a persistent synthetic startup world with:

- visible customer accounts, including ACV, renewal windows, support tier, and relationship health
- recurring stakeholder personas with distinct agenda biases
- team-state signals such as capacity, burnout, and execution reliability
- market context and governance constraints that make some actions strategically unsafe

## Benchmark Slices

The environment now has two evaluation slices:

- One-shot decision tasks: `task1`, `task2`, `task4`, `task5`
- Long-horizon adaptation tasks: `task3`, `task6`, `task7`

The legacy hackathon runner intentionally defaults to the 3 original tasks required by that submission flow:

- `task1_feedback_triage`
- `task2_roadmap_priority`
- `task3_startup_week`

This keeps the mandatory [`inference.py`](inference.py) runtime comfortably inside the submission budget while preserving the easy / medium / hard progression. Full paper-style evaluations should use the dedicated baseline runners in strict paper-eval mode instead of the hackathon path.

## Environment API

The public OpenEnv surface is:

- `reset(seed: int | None = None, task_id: str | None = None) -> LatentGoalOpsObservation`
- `step(action: LatentGoalOpsAction) -> LatentGoalOpsObservation`
- `state -> LatentGoalOpsState`

The root manifest is in [`openenv.yaml`](openenv.yaml), and the FastAPI entrypoint is in [`server/app.py`](server/app.py).

## Observation Space

The unified observation includes these typed fields:

- `task_id`
- `step_index`
- `horizon`
- `sim_date`
- `sim_day_label`
- `task_summary`
- `narrative`
- `dashboard`
- `inbox`
- `backlog`
- `accounts`
- `stakeholders`
- `teams`
- `market_context`
- `governance_constraints`
- `alerts`
- `calendar_events`
- `decision_ledger`
- `pending_effects`
- `realized_effects`
- `budget_remaining`
- `capacity_remaining`
- `sprint_budget`
- `stakeholder_notes`
- `available_actions`
- `done`
- `reward`
- `metadata`

The dashboard contains realistic metrics such as `dau`, `mrr`, `d30_retention`, `ops_margin`, and `support_ticket_volume`.

## Action Space

The unified action model contains task-specific fields:

- Task 1: `labels`, `priorities`, `escalate_ids`
- Task 2: `selected_item_ids`, `rationale_summary`
- Task 3: `chosen_initiatives`, `messaging_action`, `pricing_change_pct`, `support_policy`, `rationale`
- Task 4: `budget_allocations`, `rationale_summary`
- Task 5: `chosen_initiatives`, `messaging_action`, `pricing_change_pct`, `support_policy`, `rationale`

Validation forbids irrelevant cross-task fields, so agents cannot submit malformed hybrid actions without penalty.

## Tasks

### Task 1: Weighted Feedback Triage

Difficulty: easy

The agent receives 8-12 customer messages and must:

- label each item correctly
- assign urgency from 1-5
- escalate up to 3 items

Each message is now tied to a visible customer account profile, so the agent must reason over content plus economics: ACV, renewal timing, support expectations, and relationship health. The hidden goal changes the category weighting of errors, churn risk, revenue issues, and efficiency problems, but it does not change the ground-truth label.

Grader:

- `0.50 * label_accuracy`
- `0.30 * priority_alignment`
- `0.20 * escalation_overlap`

### Task 2: Roadmap Prioritization Under Budget

Difficulty: medium

The agent receives a visible backlog with costs, KPI deltas, beneficiary segments, linked accounts, implementation risk, and policy tags. Recurring stakeholder notes only indirectly reveal the true objective. The hidden goal controls which KPI improvements matter most.

Grader:

- normalized hidden utility against a reproducible random baseline
- budget waste penalty for leaving too much unused capacity

### Task 3: Startup Week

Difficulty: hard

The agent operates a startup over a dated operating calendar. Each day includes a narrative briefing, inbox messages, a backlog, alerts, explicit upcoming calendar events, a decision ledger from prior days, delayed effects still in flight, customer accounts with renewal windows and contract value, internal team state, market context, and visible governance constraints. In some episodes the hidden goal silently shifts in the middle, forcing the agent to adapt without being told.

Grader:

- terminal latent utility
- adaptation score after goal shift
- trajectory coherence
- constraint adherence

### Task 4: Capital Allocation Under Uncertainty

Difficulty: medium

The agent receives a visible menu of operating programs and must allocate discrete budget points across them. Programs expose visible allocation caps, saturation hints, beneficiary segments, dependencies, conflicts, and implementation risk. The hidden goal determines which returns matter most, but the visible structure makes over-allocation and scattered portfolios strategically costly.

Grader:

- normalized hidden utility against a reproducible random-allocation baseline
- budget use quality under a visible budget cap

### Task 5: Crisis Response Package

Difficulty: hard

The agent must choose one executive response package under pressure: a small set of initiatives plus pricing, messaging, and support policy. The observation includes a crisis narrative, inbox pressure, high-touch accounts, governance constraints, and resource limits. The challenge is to infer the latent objective while still avoiding obviously unsafe policy combinations.

Grader:

- normalized package value against a reproducible random baseline
- explicit constraint adherence

## Reward Design

The reward is intentionally leakage-safe. It only depends on observable KPI movement and action coherence, not on the hidden weight vector directly.

Step reward combines:

- KPI improvement reward
- strategy coherence reward
- invalid action penalty
- budget waste penalty
- KPI damage penalty
- governance-violation penalty

This gives useful partial progress feedback without letting the agent recover the latent objective by probing the reward surface.

Task 3 also exposes simulation-time structure for agents and analysis:

- dated observations via `sim_date`
- explicit `decision_ledger`
- `pending_effects` and `realized_effects`
- visible `calendar_events` that let agents reason about delayed consequences
- account-level renewals and contract-risk signals that land later in the episode
- market and team state that can worsen when the wrong strategy keeps compounding

## Persistent Entity World

Every episode is built inside a synthetic but stateful startup world rather than a one-off prompt bundle.

The world currently contains:

- `CustomerAccount` entities with segment, seat count, ACV, renewal timing, support tier, security sensitivity, churn propensity, and strategic importance
- `StakeholderPersona` entities for roles such as CEO, CFO, CTO, Head of CS, and Growth Lead
- `InternalTeamState` entities that expose capacity, burnout risk, reliability, specialization, and cross-team friction
- `MarketContext` state such as runway, board pressure, compliance exposure, and pipeline health
- `GovernanceConstraint` rules for pricing, SLA handling, and margin preservation

Task 3 mutates this world over time:

- pricing and support actions can trigger governance failures
- initiatives can target specific accounts and land with delayed effects
- renewal windows tick down each simulated day
- renewals and churn scares create dated business outcomes
- team state and board pressure evolve as consequences accumulate

The more detailed design note for this layer lives in [paper/ENTITY_WORLD_SPEC.md](paper/ENTITY_WORLD_SPEC.md).

## Paper Freeze and Release Docs

The strengthened frozen paper artifact bundle lives under `outputs/paper-freeze-2026-04-18`.
Release-facing documentation for that freeze lives under `paper/release/`, with the reviewer entry point at [`paper/release/README.md`](paper/release/README.md):

- `DATASHEET.md`
- `REPRODUCTION.md`
- `latentgoalops_croissant.json`

## Credentials

The submission baseline uses the required hackathon env vars and the official OpenAI Python client.

```env
API_BASE_URL=https://inference.do-ai.run/v1
MODEL_NAME=openai-gpt-oss-20b
HF_TOKEN=...
```

Notes:

- `HF_TOKEN` is the required submission variable name for the baseline script. If you are using DigitalOcean Gradient, point `API_BASE_URL` at the Gradient endpoint and provide that token through `HF_TOKEN`.
- `DIGITALOCEAN_API_TOKEN` and `OPENAI_BASE_URL` are still accepted as backward-compatible fallbacks for local runs.

## Local Setup

### 1. Create and sync the `uv` environment

```bash
uv sync
```

### 2. Run tests

```bash
uv run pytest
```

### 3. Start the server locally

```bash
uv run server --port 8000
```

### 4. Validate the environment structure

```bash
uv run python -m openenv.cli validate .
```

## Baselines

### Submission baseline

This is the required hackathon entrypoint. It uses the OpenAI Python client against an OpenAI-compatible endpoint and defaults to the 3 core submission tasks.

```bash
export API_BASE_URL=https://inference.do-ai.run/v1
export MODEL_NAME=openai-gpt-oss-20b
export HF_TOKEN=...
uv run python inference.py
```

Optional local Gradient compatibility:

```bash
set -a
source .env
export API_BASE_URL="${API_BASE_URL:-https://inference.do-ai.run/v1}"
export MODEL_NAME="${MODEL_NAME:-openai-gpt-oss-20b}"
export HF_TOKEN="${HF_TOKEN:-$DIGITALOCEAN_API_TOKEN}"
set +a
uv run python inference.py --tasks task1_feedback_triage,task2_roadmap_priority,task3_startup_week
```

### Reference baselines (local evaluation)

These policies help sanity-check the environment (for example oracle above random):

```bash
uv run baseline --policy heuristic --tasks all --seeds 100:3
uv run baseline --policy random --tasks all --seeds 100:3
uv run baseline --policy oracle --tasks all --seeds 100:3
```

Model-driven baseline (all tasks, OpenAI-compatible endpoint):

```bash
uv run baseline --policy model --model openai-gpt-oss-20b --tasks all --seeds 100:1
```

Synthetic operator baseline (persona-style proxy; uses the same endpoint configuration as the model baseline):

```bash
uv run baseline --policy synthetic_operator --persona-model openai-gpt-oss-20b --operator-style auto --tasks all --seeds 100:3
```

## Expected baseline behavior

The exact numbers depend on seeds, but the intended ordering is oracle above heuristic above random, with Task 3 as the hardest temporal task and goal shifts producing visible degradation when objectives are misread.

## Outputs

Runs write JSONL traces, `checkpoint.json`, and `summary.json` under the chosen output directory (for example `outputs/<run>/`). The summary separates overall mean scores from strict versus parse-rescued scores where applicable.

## Docker and Hugging Face Space

This repo includes a root [`Dockerfile`](Dockerfile) for Docker-based HF Spaces.

Suggested HF runtime secrets / variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

If you deploy with DigitalOcean Gradient as the backing model endpoint, set:

- `API_BASE_URL=https://inference.do-ai.run/v1`
- `MODEL_NAME=<your Gradient model id>`
- `HF_TOKEN=<your Gradient token>`

## Repository layout

- [`src/latentgoalops/models.py`](src/latentgoalops/models.py): typed public models
- [`src/latentgoalops/server/environment.py`](src/latentgoalops/server/environment.py): main environment
- [`src/latentgoalops/server/tasks`](src/latentgoalops/server/tasks): task generators and dynamics
- [`src/latentgoalops/server/grader.py`](src/latentgoalops/server/grader.py): deterministic graders
- [`src/latentgoalops/baseline/run_baseline.py`](src/latentgoalops/baseline/run_baseline.py): baseline runner
- [`src/latentgoalops/logging_`](src/latentgoalops/logging_): JSONL logging schemas and writer
- [`src/latentgoalops/analysis`](src/latentgoalops/analysis): aggregation, stats, and plotting (optional)
