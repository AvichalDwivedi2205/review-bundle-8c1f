# Reproducing the Frozen Paper Assets

This note describes how to regenerate the analysis assets tied to the frozen bundle at:

- `outputs/paper-freeze-2026-04-18`

## Preconditions

- Python environment managed by `uv`
- repository checkout rooted at the current workspace
- either the anonymous reviewer archive or a local checkout that includes the
  exact raw run directories referenced by
  `outputs/paper-freeze-2026-04-18/metadata/run_manifest.json`

The paper asset pipeline does not require rerunning models if the frozen review
archive already includes the referenced raw run directories.

## 1. Regenerate tables, figures, case studies, and LaTeX fragments

```bash
uv run paper-assets --config configs/paper_protocol_strengthened.yaml
```

This command validates the frozen summaries, rebuilds CSV/JSON tables, regenerates PNG/PDF figures, refreshes case studies, and rewrites the LaTeX table fragments under:

- `paper/latex/generated`

## 2. Inspect the resolved protocol and manifest

The asset builder writes:

- `outputs/paper-freeze-2026-04-18/metadata/paper_protocol_resolved.json`
- `outputs/paper-freeze-2026-04-18/metadata/run_manifest.json`

These files are the source of truth for:

- exact run directories
- provider mode
- task scope
- figure/table plan
- diagnostic runs

The strengthened `2026-04-18` freeze is the frozen analysis root for the paper
assets, but the manifest intentionally reuses some canonical underlying run
directories from earlier validated reruns and targeted top-ups. Reviewers
should treat the resolved manifest, not directory naming alone, as the
reproducible source of truth. The anonymous review archive packages those
referenced directories under the same relative paths expected by the manifest.

## 3. Compile the manuscripts

If a TeX toolchain is available, compile from `paper/latex`.
The repository includes a prebuilt `tectonic` binary under `tools/tectonic-prebuilt/tectonic`.

Example:

```bash
cd paper/latex
../../tools/tectonic-prebuilt/tectonic main.tex
../../tools/tectonic-prebuilt/tectonic supplement.tex
```

## 4. Main aggregate definition

The paper's primary aggregate uses:

- `task3_startup_week`
- `task6_incident_response_week`
- `task7_quarterly_headcount_plan`

with:

- `10` seeds per task
- `core` and `heldout` splits
- hidden shifts enabled
- strict paper-eval mode

This yields `30` episodes per model/split in the headline aggregate.

## 5. Auxiliary smoke-test tasks

The reviewer-facing release also exposes auxiliary one-shot tasks. The main
paper aggregate remains the three sequential environments above; these
auxiliary tasks are useful for smoke tests and extensibility checks:

- `task1_feedback_triage`
- `task2_roadmap_priority`
- `task4_capital_allocation`
- `task5_crisis_response`

For a quick end-to-end benchmark sanity check, run:

```bash
uv run baseline --policy heuristic --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-heuristic-all
uv run baseline --policy random --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-random-all
uv run baseline --policy oracle --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-oracle-all
```

## 6. Interpreting model identifiers

The resolved protocol preserves the exact raw serving identifiers used in the frozen runs.
Some paper-facing tables/figures map these to shorter labels for readability, but the raw aliases remain the reproducible source of truth.

The paper pipeline also respects each model's `benchmark_role`:

- `primary` models appear in the benchmark-facing main tables and figures
- `external_anchor` models are preserved in the artifact bundle but reported separately

## 7. Release-facing documents

The frozen release description lives in:

- `paper/release/README.md`
- `paper/release/DATASHEET.md`
- `paper/release/REVIEW_BUNDLE.md`
- `paper/release/COMPUTE.md`
- `paper/release/THIRD_PARTY_ASSETS.md`
- `paper/release/latentgoalops_croissant.json`

If a future rerun changes the freeze root or task scope, publish a new dated release note instead of overwriting the existing frozen metadata.
