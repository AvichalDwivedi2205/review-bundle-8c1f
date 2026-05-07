# LatentGoalOps Release Index

This directory is the reviewer-facing entry point for the strengthened frozen analysis root at `outputs/paper-freeze-2026-04-18` and the anonymous reviewer archive built from this repository.

## What to read first

- [`DATASHEET.md`](DATASHEET.md): benchmark scope, composition, intended uses, and limitations.
- [`REPRODUCTION.md`](REPRODUCTION.md): exact commands to regenerate the paper assets and sanity-check the benchmark.
- [`REVIEW_BUNDLE.md`](REVIEW_BUNDLE.md): what the anonymous reviewer archive contains and how it is sanitized.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md): exact NeurIPS E\&D upload checklist for the paper PDF and anonymous code artifact.
- [`COMPUTE.md`](COMPUTE.md): preserved timing and provider metadata for the frozen runs.
- [`THIRD_PARTY_ASSETS.md`](THIRD_PARTY_ASSETS.md): external model and runtime asset inventory with license and terms pointers.
- [`latentgoalops_croissant.json`](latentgoalops_croissant.json): machine-readable dataset metadata for the frozen bundle.
- [`../../outputs/paper-freeze-2026-04-18/README.md`](../../outputs/paper-freeze-2026-04-18/README.md): bundle-root index for the frozen artifact directory.

## Scope at a glance

- Paper-facing environments: P1 `task3_startup_week`, P2 `task6_incident_response_week`, P3 `task7_quarterly_headcount_plan`
- Auxiliary one-shot tasks: included for smoke tests and extensibility, not headline paper claims
- Frozen analysis root: generated tables, figures, case studies, resolved protocol metadata, and release docs
- Anonymous reviewer archive: runnable code plus the exact raw run directories referenced by the resolved manifest

## Offline reviewer path

1. `uv sync`
2. `uv run python -m openenv.cli validate .`
3. `uv run baseline --policy heuristic --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-heuristic-all`
4. `uv run paper-assets --config configs/paper_protocol_strengthened.yaml`
5. `cd paper/latex && ../../tools/tectonic-prebuilt/tectonic main.tex && ../../tools/tectonic-prebuilt/tectonic supplement.tex`

This path verifies the frozen benchmark assets without requiring API
credentials or rerunning the full model matrix.

## Full rerun path

If you want to execute the interactive model runner rather than only verify the
frozen artifact:

1. set the required local or API model credentials
2. set `MODEL_NAME`
3. run `uv run python inference.py`

The credentialed path is optional for reviewer verification. The primary review
artifact is the frozen bundle plus the executable paper-asset pipeline above.

## Anonymous reviewer archive

Build the review-ready archive with:

```bash
uv run python scripts/build_neurips_review_bundle.py
```

The builder creates a sanitized directory tree and a zip archive under:

- `artifacts/neurips-review-bundle/`

The archive contains:

- the runnable benchmark code for the paper-facing environments plus auxiliary smoke-test tasks
- the `outputs/paper-freeze-2026-04-18` analysis root
- the exact raw run directories referenced by `metadata/run_manifest.json`
- the LaTeX sources and compiled PDFs if present
- the release docs in this directory

The builder intentionally excludes `.git/`, local dotfiles, citation metadata,
and other author-identifying repository artifacts.

## Submission note

The Croissant file is written against repository-relative bundle paths so that
it works both in a local checkout and in the anonymous reviewer archive. If you
later mirror the benchmark to a public hosting page, update the `url` and
`contentUrl` entries to that public location before camera-ready release.
