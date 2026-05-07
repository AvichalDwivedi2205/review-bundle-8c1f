# Third-Party Assets and Terms

This note consolidates the external model and runtime assets used in the
LatentGoalOps benchmark release. It is written for the anonymous review bundle,
so it references official upstream sources and explains how local serving
aliases should be interpreted.

## Evaluated model families

| Local benchmark identifiers | Upstream creator | Official source | Terms / license notes |
| --- | --- | --- | --- |
| `qwen3.5:2b`, `qwen3.5:9b`, `qwen3.5:27b` | Qwen Team | `Qwen/Qwen3-*` official model cards on Hugging Face | Apache 2.0. The frozen manifest preserves the exact local serving aliases used in the runs. When a local Ollama installation is available, `scripts/verify_ollama_model_identity.sh` can capture the exact pulled modelfile behind an alias. |
| `deepseek-r1:1.5b`, `deepseek-r1:8b`, `deepseek-r1:14b` | DeepSeek-AI | Official DeepSeek-R1 model card / Ollama `deepseek-r1` library page | MIT license for DeepSeek-R1 weights. The Ollama publisher notes that Qwen-distilled variants inherit upstream Qwen Apache 2.0 ancestry, and the `8b` alias currently points to the Qwen3-based DeepSeek-R1-0528 8B variant. |
| `gemma4:e2b`, `gemma4:e4b`, `gemma4:26b` | Google DeepMind | Official Gemma model cards and Gemma Terms of Use | Subject to Google Gemma Terms of Use rather than a plain OSS license. The release bundle treats these models as externally governed assets and does not redistribute their weights. |
| `gpt-oss:20b`, `openai-gpt-oss-120b` | OpenAI | Official OpenAI `gpt-oss` release page and model card | Apache 2.0 license for the open-weight models, plus the `gpt-oss` usage policy. The local `20b` configuration is Ollama-served; `120b` is reported only as an external OpenRouter-served anchor. |
| `google/gemini-3-flash-preview` | Google DeepMind | Google Gemini model documentation / OpenRouter model catalog | Hosted API probe served through OpenRouter. Reported only as exploratory external calibration; LatentGoalOps does not redistribute model weights. |
| `openai/gpt-5.4-mini` | OpenAI | OpenAI model documentation / OpenRouter model catalog | Hosted API probe served through OpenRouter under provider terms. Reported only as exploratory external calibration. |
| `anthropic/claude-sonnet-4.6` | Anthropic | Anthropic Claude documentation / OpenRouter model catalog | Hosted API probe served through OpenRouter under provider terms. Core-only exploratory calibration; no weights redistributed. |

## Serving dependencies

| Asset | Role in benchmark | Official source | Terms / license notes |
| --- | --- | --- | --- |
| Ollama | Local serving runtime for the primary benchmark models | Official Ollama GitHub repository / library pages | MIT license for the open-source runtime. Pulled model weights remain governed by their upstream model licenses and terms. |
| OpenRouter | API routing layer for the external anchor and hosted-model probes | Official OpenRouter Terms of Service | External hosted service used for `gpt-oss-120b` and the exploratory Gemini/GPT/Claude probes. These are reported separately from the primary local matrix. |

## Attribution and redistribution boundary

- LatentGoalOps redistributes benchmark code, synthetic benchmark instances,
  generated paper assets, and release metadata.
- LatentGoalOps does **not** redistribute third-party model weights inside the
  review archive.
- Reviewers who want to rerun locally served model slices should pull the
  relevant upstream model through their own Ollama installation and remain
  subject to the upstream model's license or terms.

## Alias handling

The paper intentionally reports readable family labels such as `Qwen 27B` and
`Gemma 26B`, while the reproducible manifest preserves the raw local serving
identifiers used in the frozen runs. If a future public release wants to attach
exact upstream modelfiles or quantization fingerprints, the intended mechanism
is:

```bash
scripts/verify_ollama_model_identity.sh <model-name>
```

That script captures `ollama show` output for a locally installed alias without
changing the frozen benchmark results.
