# Compute Disclosure

This note summarizes the compute metadata that are preserved in the frozen
LatentGoalOps benchmark release and the anonymous reviewer archive.

## What is preserved

- The primary benchmark matrix uses `10` locally served model configurations
  through Ollama and one external anchor (`gpt-oss-120b`) through OpenRouter.
- Step and episode logs preserve `started_at`, `finished_at`,
  `elapsed_seconds`, `input_tokens`, `output_tokens`, and `cost_usd` when the
  provider exposes billing information.
- The frozen release preserves the exact run directories referenced by
  `outputs/paper-freeze-2026-04-18/metadata/run_manifest.json`.

## Validation host used for this release pass

The current release validation pass, including manuscript rebuild and archive
verification, was run on:

- `Apple M4`
- `16 GB` memory

This host note is useful for bundle validation only. It should not be confused
with a complete hardware inventory for every original frozen benchmark run.

## Frozen runtime summary

The values below are recomputed from the preserved episode summaries in the
referenced run directories.

| Category | Episodes | Wall-clock total | Notes |
| --- | ---: | ---: | --- |
| Primary local benchmark matrix | 630 | 10.87 h | Ten locally served primary configurations across `core` and `heldout` |
| External API anchor | 60 | 3.50 h | `gpt-oss-120b` via OpenRouter, reported separately from the primary matrix |
| Exploratory hosted API probes | 180 | 0.83 h | Gemini 3 Flash preview, GPT-5.4 mini, and Claude Sonnet 4.6 via OpenRouter; not part of the primary local matrix |
| Deterministic baselines | 360 | 0.04 h | Random, heuristic, and myopic oracle |
| Shell-controlled appendix diagnostic | 80 | 11.76 h | Parse-repair diagnostic for `qwen3.5:27b` and `gemma4:26b` |

The exploratory hosted API probes consumed approximately `8.57M` input tokens,
`0.150M` output tokens, and `$12.89` in provider-reported cost. These probes
were run after the frozen local matrix to calibrate hosted-model behavior under
the same action shell; they are reported as exploratory external probes rather
than as primary benchmark evidence.

Notable slow slices from the preserved logs:

- `qwen3.5:27b` heldout: `30` episodes, median `573.5 s`, total `15743.7 s`
- `gpt-oss:20b` core: `30` episodes, median `113.8 s`, total `3652.4 s`
- `openai-gpt-oss-120b` heldout: `30` episodes, median `117.6 s`, total `8077.9 s`

## What is not fully preserved

- The exact original worker inventory for every frozen run is not stored in the
  manifest as a complete accelerator-and-memory table.
- Several local runs were served through different Ollama ports during the
  strengthening cycle, but those ports are not, by themselves, a reliable
  description of the underlying worker hardware.
- For that reason, the paper does not make hardware-normalized throughput
  claims, and the NeurIPS checklist answer for compute resources remains
  conservative.

## What would be required for a checklist "Yes"

Under the NeurIPS 2026 checklist guidance, a stronger "Yes" answer would
require a complete compute description for the result-bearing experiments, not
just timing summaries. In practice that means preserving one of the following:

- the exact worker type for each frozen run, such as CPU/GPU/accelerator class
  or cloud instance family
- relevant memory and storage information for those workers
- per-run or per-cell compute requirements together with total reported compute
- disclosure of whether the full project used materially more compute than the
  experiments reported in the paper

There are two honest ways to reach that standard in a future freeze:

1. Recover the original worker inventory for all frozen result-bearing runs and
   attach it to the manifest.
2. Re-run the full submission-critical matrix on a single documented host (or a
   small documented host set), then freeze that rerun as the new source of
   truth for the paper.

Until one of those is done, answering "No" is more accurate than over-claiming
hardware-level reproducibility.

## How to inspect compute metadata

The preserved logs can be audited directly from the anonymous reviewer archive.
Representative commands:

```bash
uv run paper-assets --config configs/paper_protocol_strengthened.yaml
uv run python -m openenv.cli validate .
```

For direct inspection, the timing and token fields live in each referenced
`runs.jsonl` file and are described in `paper/LOGGING_SPEC.md`.
