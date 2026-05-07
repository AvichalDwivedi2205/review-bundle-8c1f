# NeurIPS E&D Final Submission Checklist

This note is the final upload checklist for the LatentGoalOps NeurIPS 2026
Evaluations and Datasets submission.

## 1. OpenReview paper submission

Upload one single PDF containing:

- 9-page main paper content
- references
- optional appendix
- mandatory NeurIPS checklist

Do not upload the appendix as a separate paper PDF.

## 2. Recommended PDF choice

Use the canonical paper build from:

- `paper/latex/main.pdf`

This file is the submission PDF. If you also keep a copy at
`paper/latex/submission_single.pdf`, ensure the two files are identical before
upload.

## 3. Code artifact URL

Because LatentGoalOps is a reusable benchmark / evaluation environment, the
full-paper submission must include an anonymous reviewer-accessible code URL.

Required properties:

- anonymous
- accessible at submission time
- executable
- documented

Recommended hosted artifact:

- the zip archive produced by `scripts/build_neurips_review_bundle.py`

Local output path:

- `artifacts/neurips-review-bundle/review-bundle.zip`

## 4. Review-bundle build

Build the anonymous review bundle with:

```bash
uv run python scripts/build_neurips_review_bundle.py
```

This bundle should contain:

- runnable benchmark code
- `paper/release/`
- `paper/latex/`
- frozen artifact root `outputs/paper-freeze-2026-04-18/`
- exploratory hosted API logs under `outputs/api-probe-2026-05-07/`
- exact manifest-referenced run directories
- analysis inputs required by `paper-assets`
- bundled `tectonic`

## 5. Minimum verification before upload

From a clean extracted review bundle:

```bash
uv sync
uv run python -m openenv.cli validate .
uv run baseline --policy heuristic --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-heuristic-all
uv run baseline --policy random --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-random-all
uv run baseline --policy oracle --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-oracle-all
uv run paper-assets --config configs/paper_protocol_strengthened.yaml
cd paper/latex
../../tools/tectonic-prebuilt/tectonic main.tex
```

## 6. OpenReview field choices

For this submission:

- Review mode: `Double-blind`
- Dataset submission: `checked` if OpenReview treats the benchmark artifact as
  a dataset; upload `paper/release/latentgoalops_croissant.json` and use the
  same anonymous reviewer-accessible bundle URL as the dataset URL.
- Contribution type: `Benchmark design and benchmark analysis`
- Code URL: anonymous hosted review-bundle URL

## 7. Final sanity check

Before uploading:

- PDF opens and is anonymized
- code URL works in an incognito browser
- review bundle extracts without errors
- smoke tests run
- `paper-assets` runs
- `main.pdf` rebuilds from the bundle
