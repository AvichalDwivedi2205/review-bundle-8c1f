# LaTeX Paper Workspace

This directory contains the manuscript draft for the LatentGoalOps paper.

Files:
- `main.tex`: main paper draft
- `supplement.tex`: supplement / appendix draft
- `references.bib`: BibTeX entries used by both

Notes:
- The workspace now includes the official `neurips_2026.sty` file and the
  accompanying `checklist.tex` template from the NeurIPS 2026 paper bundle.
- The manuscript preamble still falls back to a standard `article` setup if
  the official NeurIPS style file is missing.
- The papers can be compiled locally with the bundled `tectonic` binary under
  `tools/tectonic-prebuilt/tectonic`.
- `main.tex` now includes the filled checklist from `checklist_filled.tex` for
  submission-style builds, while the raw NeurIPS template remains available in
  `checklist.tex`.

Expected compile command once a TeX toolchain is present:

```bash
cd paper/latex
../../tools/tectonic-prebuilt/tectonic main.tex
../../tools/tectonic-prebuilt/tectonic supplement.tex
```
