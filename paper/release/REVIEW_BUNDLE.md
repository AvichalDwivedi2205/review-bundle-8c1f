# Anonymous Review Bundle

This note describes the archive built for anonymous NeurIPS review.

## Build command

```bash
uv run python scripts/build_neurips_review_bundle.py
```

The builder writes a sanitized directory tree and zip archive under:

- `artifacts/neurips-review-bundle/`

## What the archive contains

- runnable benchmark code for the three paper-facing environments plus auxiliary smoke-test tasks
- release documentation under `paper/release/`
- paper LaTeX sources and compiled PDFs if they already exist locally
- the frozen analysis root at `outputs/paper-freeze-2026-04-18/`
- the exact raw run directories referenced by
  `outputs/paper-freeze-2026-04-18/metadata/run_manifest.json`
- the bundled `tectonic` binary used for LaTeX compilation

## What the builder sanitizes

- excludes `.git/`, local dotfiles, logs, and other workstation artifacts
- excludes `CITATION.cff`
- rewrites package/runtime metadata that would reveal author identity in the
  anonymous archive
- rewrites the non-anonymous author branches in the LaTeX sources
- rewrites generated metadata that would otherwise contain absolute local paths
- performs a final grep-based anonymity scan for common identifying strings

## Reviewer workflow

After extracting the archive:

```bash
uv sync
uv run python -m openenv.cli validate .
uv run baseline --policy heuristic --tasks all --seeds 100:3 --output-dir outputs/reviewer-smoke-heuristic-all
uv run paper-assets --config configs/paper_protocol_strengthened.yaml
cd paper/latex
../../tools/tectonic-prebuilt/tectonic main.tex
../../tools/tectonic-prebuilt/tectonic supplement.tex
```

The archive is designed to be sufficient for reviewer verification without
requiring access to the original author checkout.
