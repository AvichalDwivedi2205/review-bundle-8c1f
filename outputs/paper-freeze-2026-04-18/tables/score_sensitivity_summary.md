# Score Sensitivity Analysis

Offline recomputation from frozen final-episode grader outputs. No model inference is run.

The `original` row uses the official stored `grader.score`; other rows recompute scores from saved final-episode subscores under alternate weights.

| Variant | Split | Best primary | Best score | Oracle gap | Median shift gap | Baseline order |
|---|---:|---|---:|---:|---:|---:|
| original | core | qwen3.5:27b | 0.443 | 0.156 | 0.154 | True |
| original | heldout | deepseek-r1:8b | 0.446 | 0.152 | 0.183 | True |
| utility_only | core | qwen3.5:27b | 0.392 | 0.012 | 0.072 | True |
| utility_only | heldout | gpt-oss:20b | 0.393 | 0.016 | 0.083 | True |
| balanced | core | deepseek-r1:14b | 0.494 | 0.087 | 0.185 | True |
| balanced | heldout | deepseek-r1:8b | 0.499 | 0.081 | 0.209 | True |
| adaptation_heavy | core | qwen3.5:27b | 0.463 | 0.089 | 0.214 | True |
| adaptation_heavy | heldout | deepseek-r1:8b | 0.462 | 0.088 | 0.249 | True |
| adaptation_light | core | deepseek-r1:1.5b | 0.519 | 0.057 | 0.109 | True |
| adaptation_light | heldout | qwen3.5:2b | 0.523 | 0.056 | 0.118 | True |
| no_coherence | core | qwen3.5:27b | 0.430 | 0.061 | 0.169 | False |
| no_coherence | heldout | deepseek-r1:8b | 0.429 | 0.060 | 0.207 | True |
