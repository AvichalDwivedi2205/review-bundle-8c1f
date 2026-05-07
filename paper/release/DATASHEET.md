# LatentGoalOps Frozen Bundle Datasheet

## Overview

LatentGoalOps is a synthetic benchmark for latent objective inference and adaptation under silent non-stationarity.
The strengthened frozen paper bundle corresponds to the artifact root:

- `outputs/paper-freeze-2026-04-18`

The bundle is intended to support reproducible benchmark reporting rather than online model serving, and it is the reviewer-facing release snapshot for the NeurIPS 2026 Evaluations & Datasets submission.

## Motivation

Most agent benchmarks make the objective explicit.
LatentGoalOps instead measures whether an agent can infer a hidden operational objective from public evidence and adapt when that objective changes without announcement.

The frozen paper release is designed to answer three concrete questions:

1. Do current models show simple monotone improvement on latent-objective adaptation?
2. How much of failure is latent-policy weakness versus structured-action unreliability?
3. Does the public observation surface leak the hidden objective through trivial shortcuts?

## Composition

The frozen paper aggregate uses three sequential environments:

- P1 Startup Week (`task3_startup_week`)
- P2 Incident Response Week (`task6_incident_response_week`)
- P3 Quarterly Headcount Plan (`task7_quarterly_headcount_plan`)

The repository also includes auxiliary one-shot tasks for smoke tests,
ablation checks, and future extensions. They are not headline paper claims.

The bundle includes:

- generated CSV/JSON tables
- generated PNG/PDF figures
- curated case studies
- resolved protocol metadata and run manifests
- release-facing docs and Croissant metadata

The anonymous reviewer archive additionally includes the exact raw run
directories referenced by the resolved manifest, so the frozen paper assets can
be regenerated without rerunning the benchmark.

## Instance Structure

Each episode exposes only public state:

- KPI dashboard
- inbox/stakeholder messages
- backlog or planning options
- accounts
- stakeholder personas
- teams
- market context
- governance constraints
- dated decision ledger
- pending and realized delayed effects

The hidden objective is never directly revealed.

## Collection Process

The benchmark instances are procedurally generated from deterministic seeds.
The frozen paper protocol uses:

- `10` seeds per task
- both `core` and `heldout` scenario families
- hidden shifts enabled
- strict paper-eval mode with parse repair and heuristic rescue disabled

## Primary Aggregate

The primary aggregate pools three tasks across ten seeds each, yielding:

- `30` episodes per model on `core`
- `30` episodes per model on `heldout`

This aggregate is the paper's headline score view.
Native-only diagnostics may use fewer effective episodes because shell-fragile models can fail to remain native to the action interface.

## Hidden Shift Mechanism

In the frozen configuration:

- hidden shifts occur with probability `0.6`
- shift steps are sampled from `{3, 4, 5}` when the horizon allows it
- shifts may be abrupt or drift-like
- shifts can alter latent weights and structured objective attributes such as risk posture, planning horizon, segment focus, and governance strictness

## Action Interface

The sequential tasks require structured, executable actions.

P1 / P2 fields include:

- `chosen_initiatives`
- `messaging_action`
- `pricing_change_pct`
- `support_policy`
- optional rationale, memory, and belief-report fields

P3 uses:

- `budget_allocations`
- optional rationale, memory, and belief-report fields

Malformed actions trigger fallback transitions and therefore affect future state.

## Intended Uses

Appropriate uses:

- benchmark evaluation of latent-objective adaptation
- analysis of interface failure versus policy failure
- heldout generalization and leakage studies
- qualitative case-study inspection of dated trajectories

Inappropriate uses:

- treating the benchmark as a real-world deployment simulator
- interpreting the hidden objective as a normative business recommendation
- making legal, medical, or financial decisions

## Known Limitations

- One served family point, GPT-OSS-120B, is API-served and should be treated as an external anchor rather than a clean local within-family comparison.
- The hosted API probes for Gemini 3 Flash preview, GPT-5.4 mini, and Claude Sonnet 4.6 are exploratory OpenRouter runs added after the frozen local matrix. They are useful for calibration, but are not exhaustive frontier coverage and should not be mixed into local within-family scaling claims.
- Qwen and Gemma model identifiers in the frozen bundle are local serving aliases; the resolved protocol keeps those exact raw tags for reproducibility.
- The benchmark is synthetic and should be interpreted as a capability isolator, not as a direct mirror of any specific organization.

## Leakage and Validity Notes

Restricted-view centroid probes remain near four-way chance on the primary tasks, and explicit-goal-token leakage is `0.0` throughout the frozen audit.
The expected baseline ordering (`random < visible heuristic < myopic oracle`) also holds on both splits.

## Release Metadata

Release-facing metadata lives alongside this datasheet:

- `paper/release/README.md`
- `paper/release/REPRODUCTION.md`
- `paper/release/REVIEW_BUNDLE.md`
- `paper/release/COMPUTE.md`
- `paper/release/THIRD_PARTY_ASSETS.md`
- `paper/release/latentgoalops_croissant.json`

## Maintenance

This datasheet describes the frozen bundle dated `2026-04-18`.
Future reruns should either preserve this bundle unchanged or publish a new dated freeze with its own manifest and datasheet.
