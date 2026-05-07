# Paper Freeze 2026-04-18

This directory is the strengthened source-of-truth bundle for the reviewer-facing benchmark release and paper analysis.

## Contents

- `tables/`: generated CSV and JSON tables
- `figures/`: generated PNG and PDF figures
- `case_studies/`: curated trajectory narratives
- `metadata/`: resolved protocol and run manifest

## Scope

- Main paper aggregate: `task3_startup_week`, `task6_incident_response_week`, `task7_quarterly_headcount_plan`
- Full public benchmark release: all seven tasks in the benchmark suite
- Benchmark-health and leakage metadata: captured in `metadata/run_manifest.json` and the generated tables

The `2026-04-18` freeze is the frozen paper-analysis root. Some entries in the
resolved manifest intentionally point to canonical source runs that were
validated earlier or added as targeted strengthening top-ups, so reviewers
should follow `metadata/run_manifest.json` rather than assume every upstream
input physically originated under this directory name. Those referenced raw run
directories are included in the anonymous reviewer archive under their original
relative paths rather than duplicated inside this analysis root.

## Reproduction

Start from [`paper/release/README.md`](../../paper/release/README.md) and [`paper/release/REPRODUCTION.md`](../../paper/release/REPRODUCTION.md).

The release bundle is intended to stay frozen. If you need a new benchmark snapshot, publish it as a new dated freeze rather than mutating this directory in place.
