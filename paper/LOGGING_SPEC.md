# Logging Spec

This document defines the minimum logging contract for LatentGoalOps experiments and how to use timestamps in paper figures, tables, and analyses.

## Principles

- All timestamps are stored in UTC ISO-8601 format.
- Every record is attached to a reproducible experiment identity.
- Time can be analyzed at three levels: step, episode, and full run.
- Local JSONL logs and W&B use the same semantic fields whenever possible.

## Primary Identity Keys

Use this composite key to join records across files and plots:

- `run_id`
- `task_id`
- `seed`
- `policy`
- `model_name`

For step-level ordering within an episode, also use:

- `step_index`

For batch progress within a sweep, use:

- `episodes_completed`
- `episodes_total`

## Record Types

### StepLog

Each action taken by the agent produces one step record.

Fields:

- `timestamp`: record creation time
- `started_at`: step start time
- `finished_at`: step end time
- `elapsed_seconds`: wall-clock step runtime
- `run_id`
- `task_id`
- `seed`
- `policy`
- `model_name`
- `step_index`
- `action`
- `reward`
- `done`
- `observation`
- `provider_usage`
- `metadata`

Important observation fields for temporal analysis:

- `sim_date`
- `sim_day_label`
- `accounts`
- `stakeholders`
- `teams`
- `market_context`
- `governance_constraints`
- `calendar_events`
- `decision_ledger`
- `pending_effects`
- `realized_effects`

Provider usage fields:

- `input_tokens`
- `output_tokens`
- `cost_usd`
- `parse_fallback`

### EpisodeSummary

Each completed environment episode produces one summary record.

Fields:

- `timestamp`
- `started_at`
- `finished_at`
- `elapsed_seconds`
- `run_id`
- `task_id`
- `seed`
- `policy`
- `model_name`
- `score`
- `total_reward`
- `total_steps`
- `provider_usage`
- `grade`
- `metadata`

Important episode metadata:

- `progress_fraction`
- `episodes_completed`
- `episodes_total`
- `eta_seconds`
- `run_elapsed_seconds`

Important grader fields:

- `score`
- `sub_scores`
- `details`

### CheckpointState

The resumability file stores sweep-level state.

Fields:

- `run_id`
- `model_name`
- `policy`
- `tasks`
- `seeds`
- `started_at`
- `next_index`
- `cumulative_cost_usd`
- `cumulative_input_tokens`
- `cumulative_output_tokens`
- `failure_mode`
- `metadata`

## Where Logs Live

- Per-run JSONL: `outputs/.../runs.jsonl`
- Per-run checkpoint: `outputs/.../checkpoint.json`
- Per-run summary: `outputs/.../summary.json`
- Multi-model sweep summary: `outputs/.../aggregate_summary.json`

## How Timestamps Connect To Paper Analyses

### Runtime and Throughput

Use:

- `elapsed_seconds`
- `total_steps`
- `input_tokens`
- `output_tokens`

Derived metrics:

- seconds per episode
- seconds per step
- tokens per second
- dollars per episode

### Progress Curves

Use:

- `started_at`
- `finished_at`
- `progress_fraction`
- `eta_seconds`
- `run_elapsed_seconds`

These support:

- run progression plots
- cumulative score-over-time plots
- cumulative cost-over-time plots
- expected completion time comparisons across models

### Joining Time To Quality

To connect timestamps to benchmark quality in the paper:

1. Join step or episode rows on `run_id`, `task_id`, `seed`, `policy`, and `model_name`.
2. Use `step_index` to reconstruct within-episode trajectories.
3. Use `started_at` and `finished_at` to align runtime with score, cost, and grader sub-scores.
4. Aggregate by `model_name` or `task_id` for paper tables and figures.

Example questions this supports:

- Does a slower model buy better latent-goal inference?
- Does Task 3 require more wall-clock time than Task 1 and Task 2?
- Are parse fallbacks concentrated at specific times or tasks?
- Does adaptation after a goal shift increase episode runtime?
- Which dated decisions produced delayed positive or negative outcomes later in the episode?
- How long do models keep following an outdated strategy after the objective shifts?
- Which account segments were protected or harmed by a given policy sequence?
- Do governance violations precede later KPI damage or renewal losses?
- How do team burnout and board pressure evolve across model strategies?

## W&B Mapping

The following key metrics are mirrored to W&B:

- `step/reward`
- `step/elapsed_seconds`
- `step/input_tokens`
- `step/output_tokens`
- `step/cost_usd`
- `step/parse_fallback`
- `episode/score`
- `episode/elapsed_seconds`
- `episode/total_reward`
- `episode/total_steps`
- `episode/cost_usd`
- `episode/progress_fraction`
- `episode/eta_seconds`
- `episode/subscore/*`
- `summary/run_elapsed_seconds`
- `summary/cumulative_cost_usd`

## Reporting Guidance

For the paper, treat timestamps as operational metadata rather than the primary benchmark target.

Recommended uses:

- include runtime and cost tables next to score tables
- show cumulative progress plots for long sweeps
- report median episode runtime by task and model
- analyze whether hard-task adaptation costs extra time

Avoid overinterpreting raw wall-clock time across different machines unless all experiments were run on the same hardware and network setup.
