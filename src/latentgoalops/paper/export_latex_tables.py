"""Export frozen paper tables as LaTeX fragments for the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from latentgoalops.analysis.make_paper_tables import generate_tables
from latentgoalops.paper.run_paper_suite import generate_paper_suite

MODEL_PAPER_LABELS = {
    "qwen3.5:2b": "Qwen 2B",
    "qwen3.5:9b": "Qwen 9B",
    "qwen3.5:27b": "Qwen 27B",
    "deepseek-r1:1.5b": "DeepSeek 1.5B",
    "deepseek-r1:8b": "DeepSeek 8B",
    "deepseek-r1:14b": "DeepSeek 14B",
    "gemma4:e2b": "Gemma 2B",
    "gemma4:e4b": "Gemma 4B",
    "gemma4:26b": "Gemma 26B",
    "gpt-oss:20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "openai-gpt-oss-20b": "GPT-OSS 20B",
    "openai-gpt-oss-120b": "GPT-OSS 120B",
}

FAMILY_PAPER_LABELS = {
    "qwen": "Qwen3.5",
    "deepseek-r1": "DeepSeek-R1",
    "gemma4": "Gemma4",
    "gpt-oss": "GPT-OSS",
}

PROVIDER_PAPER_LABELS = {
    "ollama_local": "Ollama local",
    "openrouter_api": "OpenRouter API",
}

API_PROBE_SPECS = [
    ("Gemini 3 Flash preview", "core", "outputs/api-probe-2026-05-07/gemini-3-flash-preview/core/summary.json"),
    ("Gemini 3 Flash preview", "heldout", "outputs/api-probe-2026-05-07/gemini-3-flash-preview/heldout/summary.json"),
    ("GPT-5.4 mini", "core", "outputs/api-probe-2026-05-07/gpt-5.4-mini/core/summary.json"),
    ("GPT-5.4 mini", "heldout", "outputs/api-probe-2026-05-07/gpt-5.4-mini/heldout/summary.json"),
    ("Claude Sonnet 4.6", "core", "outputs/api-probe-2026-05-07/claude-sonnet-4.6/core-30-combined-summary.json"),
    ("Claude Sonnet 4.6", "heldout", "outputs/api-probe-2026-05-07/claude-sonnet-4.6/heldout/summary.json"),
]

BENCHMARK_HEALTH_LABELS = {
    "core_baseline_ordering": "Core baseline ordering",
    "heldout_baseline_ordering": "Heldout baseline ordering",
    "core_oracle_gap": "Core oracle gap",
    "heldout_oracle_gap": "Heldout oracle gap",
    "silent_shift_penalty_present": "Silent-shift penalty",
}

SCALING_DIAGNOSIS_LABELS = {
    "larger_model_native_policy_regression": "native policy regression",
    "larger_model_interface_regression": "interface regression",
    "larger_model_gain": "gain",
    "approximately_flat": "flat",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_protocol(config_path: str | Path) -> dict[str, Any]:
    suite = generate_paper_suite(config_path)
    metadata_path = Path(suite["metadata_dir"]) / "paper_protocol_resolved.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return _latex_escape(value)


def _fmt_ci(low: Any, high: Any, digits: int = 3) -> str:
    if low is None or high is None or (isinstance(low, float) and pd.isna(low)) or (isinstance(high, float) and pd.isna(high)):
        return "--"
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"


def _paper_model_label(value: Any) -> str:
    return MODEL_PAPER_LABELS.get(str(value), str(value))


def _paper_family_label(value: Any) -> str:
    return FAMILY_PAPER_LABELS.get(str(value), str(value))


def _paper_provider_label(value: Any) -> str:
    return PROVIDER_PAPER_LABELS.get(str(value), str(value))


def _paper_health_label(value: Any) -> str:
    return BENCHMARK_HEALTH_LABELS.get(str(value), str(value))


def _paper_scaling_diagnosis(value: Any) -> str:
    return SCALING_DIAGNOSIS_LABELS.get(str(value), str(value).replace("_", " "))


def _ordered_paper_model_labels() -> list[str]:
    # Some raw serving aliases collapse to the same paper-facing label.
    return list(dict.fromkeys(_paper_model_label(model) for model in MODEL_PAPER_LABELS))


def _format_health_value(check: str, raw_value: Any) -> str:
    if check in {"core_baseline_ordering", "heldout_baseline_ordering"}:
        try:
            parsed = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return _latex_escape(raw_value)
        preferred_order = ["random", "visible heuristic", "myopic oracle"]
        present_keys = [key for key in preferred_order if key in parsed]
        if present_keys:
            return " < ".join(f"{float(parsed[key]):.3f}" for key in present_keys)
        ordered_keys = sorted(parsed)
        return " < ".join(f"{float(parsed[key]):.3f}" for key in ordered_keys)
    try:
        return f"{float(raw_value):.3f}"
    except (TypeError, ValueError):
        return _latex_escape(raw_value)


def _generated_dir() -> Path:
    path = _repo_root() / "paper" / "latex" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _main_results_table(main_results: pd.DataFrame) -> str:
    df = main_results.copy()
    df["model"] = df["model"].map(_paper_model_label)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["model"]),
                    _latex_escape(row["split"]),
                    _fmt(int(row["episode_count"]), digits=0),
                    _fmt(row["overall_mean_score"]),
                    _fmt_ci(row["overall_ci_low"], row["overall_ci_high"]),
                    _fmt(row["strict_native_overall_mean_score"]),
                    _fmt(int(row["strict_native_episode_count"]), digits=0),
                    _fmt(row["native_action_episode_rate"], digits=2),
                    _fmt(row["empty_fallback_episode_rate"], digits=2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{llrrrrrrr}}
    \toprule
    Model & Split & Episodes & Overall & 95\% CI & Native-only & Native n & Native rate & Empty fallback \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Frozen primary benchmark matrix over Tasks 3, 6, and 7. This table
  includes only locally served models with benchmark role `primary`. Exact raw
  serving tags remain in the resolved protocol metadata. API-served comparison
  points are reported separately as external anchors.}}
  \label{{tab:mainresults}}
\end{{table}}
"""


def _external_anchor_table(anchor_results: pd.DataFrame) -> str:
    if anchor_results.empty:
        return "% No external-anchor table available.\n"
    df = anchor_results.copy()
    df["model"] = df["model"].map(_paper_model_label)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["model"]),
                    _latex_escape(row["split"]),
                    _latex_escape(_paper_provider_label(row["provider_mode"])),
                    _fmt(int(row["episode_count"]), digits=0),
                    _fmt(row["overall_mean_score"]),
                    _fmt_ci(row["overall_ci_low"], row["overall_ci_high"]),
                    _fmt(row["native_action_episode_rate"], digits=2),
                    _fmt(row["empty_fallback_episode_rate"], digits=2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{lllrrrrr}}
    \toprule
    Model & Split & Provider & Episodes & Overall & 95\% CI & Native rate & Empty fallback \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{External anchors excluded from the primary benchmark matrix because
  they were not run under the same local serving path. They are retained for
  transparency but are not used as load-bearing within-family scaling evidence.}}
  \label{{tab:externalanchors}}
\end{{table}}
"""


def _baseline_table(baseline_results: pd.DataFrame) -> str:
    rows = []
    for _, row in baseline_results.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["baseline"]),
                    _latex_escape(row["split"]),
                    _fmt(int(row["episode_count"]), digits=0),
                    _fmt(row["overall_mean_score"]),
                    _fmt_ci(row["overall_ci_low"], row["overall_ci_high"]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \small
  \begin{{tabular}}{{llrrr}}
    \toprule
    Baseline & Split & Episodes & Mean score & 95\% CI \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Baseline separation on the frozen primary-task aggregate.}}
  \label{{tab:baselines}}
\end{{table}}
"""


def _benchmark_health_table(benchmark_health: pd.DataFrame) -> str:
    keep = benchmark_health[
        benchmark_health["check"].isin(
            [
                "core_baseline_ordering",
                "heldout_baseline_ordering",
                "core_oracle_gap",
                "heldout_oracle_gap",
                "silent_shift_penalty_present",
            ]
        )
    ].copy()
    rows = []
    for _, row in keep.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(_paper_health_label(row["check"])),
                    _format_health_value(str(row["check"]), row["value"]),
                    "Yes" if bool(row["passes"]) else "No",
                    _latex_escape(row["threshold"]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \small
  \begin{{tabular}}{{p{{0.24\linewidth}}p{{0.22\linewidth}}cp{{0.26\linewidth}}}}
    \toprule
    Check & Value & Passes & Threshold \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Selected benchmark health checks used in the paper.}}
  \label{{tab:health}}
\end{{table}}
"""


def _adaptation_table(adaptation_audit: pd.DataFrame) -> str:
    main_df = adaptation_audit[
        adaptation_audit["task_id"].isin(["task3_startup_week", "task6_incident_response_week"])
    ].copy()
    main_df["model"] = main_df["model"].map(_paper_model_label)
    main_df["task_key"] = main_df["split"].str.capitalize() + " " + main_df["task_id"].map(
        {
            "task3_startup_week": "T3",
            "task6_incident_response_week": "T6",
        }
    )
    mean_pivot = main_df.pivot(index="model", columns="task_key", values="mean_adaptation_score")
    zero_pivot = main_df.pivot(index="model", columns="task_key", values="zero_adaptation_episode_rate")
    column_order = ["Core T3", "Core T6", "Heldout T3", "Heldout T6"]
    rows = []
    ordered_models = [label for label in _ordered_paper_model_labels() if label in mean_pivot.index]
    ordered_models.extend(sorted(set(mean_pivot.index) - set(ordered_models)))
    for model in ordered_models:
        values = []
        for column in column_order:
            values.append(_fmt(mean_pivot.loc[model, column]))
            values.append(_fmt(zero_pivot.loc[model, column], digits=2))
        rows.append("    " + " & ".join([_latex_escape(model), *values]) + r" \\")
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{lrrrrrrrr}}
    \toprule
    \multirow{{2}}{{*}}{{Model}} & \multicolumn{{2}}{{c}}{{Core T3}} & \multicolumn{{2}}{{c}}{{Core T6}} & \multicolumn{{2}}{{c}}{{Heldout T3}} & \multicolumn{{2}}{{c}}{{Heldout T6}} \\
    & Adapt. & Zero rate & Adapt. & Zero rate & Adapt. & Zero rate & Adapt. & Zero rate \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Supplementary adaptation audit. ``Adapt.'' is mean adaptation score
  on shifted episodes; ``Zero rate'' is the fraction of shifted episodes with
  zero adaptation score.}}
  \label{{tab:adaptationaudit}}
\end{{table}}
"""


def _shell_controlled_diagnostic_table(shell_controlled: pd.DataFrame) -> str:
    if shell_controlled.empty:
        return "% No shell-controlled diagnostic table available.\n"
    df = shell_controlled.copy()
    df["model"] = df["model"].map(_paper_model_label)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["model"]),
                    _latex_escape(row["split"]),
                    _fmt(row["frozen_overall_mean_score"]),
                    _fmt(row["diagnostic_overall_mean_score"]),
                    _fmt(row["diagnostic_assisted_overall_mean_score"]),
                    _fmt(row["diagnostic_native_action_episode_rate"], digits=2),
                    _fmt(row["diagnostic_parse_repaired_episode_rate"], digits=2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{llrrrrr}}
    \toprule
    Model & Split & Frozen overall & Shell overall & Shell assisted & Native rate & Repair rate \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Appendix shell-controlled diagnostic with parse repair enabled and heuristic rescue disabled. ``Shell assisted'' is the assisted aggregate reported by the diagnostic runner.}}
  \label{{tab:shelldiagnostic}}
\end{{table}}
"""


def _scaling_transition_table(scaling_deltas: pd.DataFrame) -> str:
    if scaling_deltas.empty:
        return "% No scaling transition table available.\n"
    rows = []
    for _, row in scaling_deltas.iterrows():
        transition = f"{_paper_model_label(row['from_model'])} $\\rightarrow$ {_paper_model_label(row['to_model'])}"
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(_paper_family_label(row["family"])),
                    _latex_escape(row["split"]),
                    transition,
                    _fmt(row["delta_overall_mean_score"]),
                    _fmt(row["delta_strict_native_overall_mean_score"]),
                    _fmt(row["delta_native_action_episode_rate"], digits=2),
                    _latex_escape(_paper_scaling_diagnosis(row["scaling_label"])),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{llp{{0.26\linewidth}}rrrp{{0.20\linewidth}}}}
    \toprule
    Family & Split & Transition & $\Delta$ overall & $\Delta$ native-only & $\Delta$ native rate & Diagnosis \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Scaling-transition summary. Native-only deltas help separate clean policy regressions from interface-mediated regressions.}}
  \label{{tab:scalingtransitions}}
\end{{table}}
"""


def _runtime_table(main_results: pd.DataFrame) -> str:
    df = main_results.copy()
    df["model"] = df["model"].map(_paper_model_label)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["model"]),
                    _latex_escape(row["split"]),
                    _latex_escape(_paper_provider_label(row["provider_mode"])),
                    _fmt(float(row["total_input_tokens"]) / 1000.0, digits=1),
                    _fmt(float(row["total_output_tokens"]) / 1000.0, digits=1),
                    _fmt(row["total_cost_usd"], digits=3),
                    _fmt(row["native_action_episode_rate"], digits=2),
                    _fmt(row["empty_fallback_episode_rate"], digits=2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{lllrrrrr}}
    \toprule
    Model & Split & Provider & In tok (K) & Out tok (K) & Cost (\$) & Native & Empty fb \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Provider/runtime summary for the frozen primary-task aggregate. This table is included because benchmark interpretation depends on serving path, token pressure, and interface reliability.}}
  \label{{tab:runtime}}
\end{{table}}
"""


def _per_task_results_table(task_breakdown: pd.DataFrame) -> str:
    if task_breakdown.empty:
        return "% No per-task results table available.\n"
    task_labels = {
        "task3_startup_week": "T3",
        "task6_incident_response_week": "T6",
        "task7_quarterly_headcount_plan": "T7",
    }
    df = task_breakdown.copy()
    df["model"] = df["model"].map(_paper_model_label)
    rows = []
    for _, row in df.sort_values(["task_id", "family", "size_b", "split"]).iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(task_labels.get(str(row["task_id"]), str(row["task_id"]))),
                    _latex_escape(row["model"]),
                    _latex_escape(row["split"]),
                    _fmt(row["overall_mean_score"]),
                    _fmt(row["strict_native_overall_mean_score"]),
                    _fmt(row["native_action_episode_rate"], digits=2),
                    _fmt(row["empty_fallback_episode_rate"], digits=2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{longtable}}{{llllrrr}}
\caption{{Per-task results for all frozen model runs on Tasks 3, 6, and 7.}} \label{{tab:pertaskresults}} \\
\toprule
Task & Model & Split & Overall & Native-only & Native rate & Empty fb \\
\midrule
\endfirsthead
\toprule
Task & Model & Split & Overall & Native-only & Native rate & Empty fb \\
\midrule
\endhead
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
"""


def _subscore_longtable(subscores: pd.DataFrame) -> str:
    if subscores.empty:
        return "% No subscore table available.\n"
    task_labels = {
        "task3_startup_week": "T3",
        "task6_incident_response_week": "T6",
        "task7_quarterly_headcount_plan": "T7",
    }
    df = subscores.copy()
    df["model"] = df["model"].map(_paper_model_label)
    rows = []
    for _, row in df.sort_values(["task_id", "family", "size_b", "split"]).iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(task_labels.get(str(row["task_id"]), str(row["task_id"]))),
                    _latex_escape(row["model"]),
                    _latex_escape(row["split"]),
                    _fmt(row["final_utility"]),
                    _fmt(row["adaptation"]),
                    _fmt(row["coherence"]),
                    _fmt(row["constraints"]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{longtable}}{{llllrrr}}
\caption{{Per-task subscore decomposition for all frozen model runs on Tasks 3, 6, and 7.}} \label{{tab:subscorelong}} \\
\toprule
Task & Model & Split & Utility & Adapt. & Coherence & Constraints \\
\midrule
\endfirsthead
\toprule
Task & Model & Split & Utility & Adapt. & Coherence & Constraints \\
\midrule
\endhead
{chr(10).join(rows)}
\bottomrule
\end{{longtable}}
"""


def _api_probe_table() -> str:
    rows = []
    for model, split, relative_path in API_PROBE_SPECS:
        summary_path = _repo_root() / relative_path
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_scores = summary.get("mean_scores", {})
        anchor_scores = [
            float(task_scores["task3_startup_week"]),
            float(task_scores["task6_incident_response_week"]),
        ]
        p1_p2 = sum(anchor_scores) / len(anchor_scores)
        p1_p3 = float(summary["overall_mean_score"])
        native_only = float(summary.get("strict_native_overall_mean_score", p1_p3))
        native_n = int(summary.get("strict_episode_count", summary["episode_count"]))
        rows.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(model),
                    _latex_escape(split),
                    _fmt(int(summary["episode_count"]), digits=0),
                    _fmt(p1_p2),
                    _fmt(p1_p3),
                    _fmt(native_only),
                    _fmt(native_n, digits=0),
                    _fmt(float(summary["native_action_episode_rate"]), digits=2),
                    _fmt(float(summary["empty_fallback_episode_rate"]), digits=2),
                ]
            )
            + r" \\"
        )
    if not rows:
        return "% No hosted API probe table available.\n"
    return rf"""
\begin{{table}}[t]
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{3pt}}
  \resizebox{{\linewidth}}{{!}}{{%
  \begin{{tabular}}{{llrrrrrrr}}
    \toprule
    Model & Split & P1--P3 eps. & P1/P2 & P1--P3 & Native-only & Native n & Native rate & Empty fallback \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
  }}
  \caption{{Exploratory hosted-model probes run through OpenRouter on the
  P1--P3 aggregate. The P1/P2 column averages the two week-long anchor tasks
  within each 30-episode P1--P3 probe. These API probes are excluded from the
  primary local-model matrix and are used only to test whether readily
  accessible hosted models close the gap to the myopic oracle. Entries are
  descriptive point estimates.}}
  \label{{tab:apiprobes}}
\end{{table}}
"""


def export_latex_tables(config_path: str | Path) -> dict[str, Any]:
    generate_tables(config_path)
    protocol = _load_protocol(config_path)
    tables_dir = _repo_root() / protocol["paper"]["tables_dir"]
    out_dir = _generated_dir()

    main_results = pd.read_csv(tables_dir / "main_results.csv")
    external_anchor_results = pd.read_csv(tables_dir / "external_anchor_results.csv")
    baseline_results = pd.read_csv(tables_dir / "baseline_results.csv")
    benchmark_health = pd.read_csv(tables_dir / "benchmark_health.csv")
    adaptation_audit = pd.read_csv(tables_dir / "adaptation_audit.csv")
    shell_controlled = pd.read_csv(tables_dir / "shell_controlled_diagnostic.csv")
    scaling_deltas = pd.read_csv(tables_dir / "scaling_deltas.csv")
    task_breakdown = pd.read_csv(tables_dir / "task_breakdown.csv")
    subscore_breakdown = pd.read_csv(tables_dir / "subscore_breakdown.csv")

    outputs = {
        "main_results_table.tex": _main_results_table(main_results),
        "external_anchor_table.tex": _external_anchor_table(external_anchor_results),
        "baseline_table.tex": _baseline_table(baseline_results),
        "benchmark_health_table.tex": _benchmark_health_table(benchmark_health),
        "adaptation_audit_table.tex": _adaptation_table(adaptation_audit),
        "shell_controlled_diagnostic_table.tex": _shell_controlled_diagnostic_table(shell_controlled),
        "scaling_transition_table.tex": _scaling_transition_table(scaling_deltas),
        "runtime_table.tex": _runtime_table(main_results),
        "per_task_results_table.tex": _per_task_results_table(task_breakdown),
        "subscore_longtable.tex": _subscore_longtable(subscore_breakdown),
        "api_probe_table.tex": _api_probe_table(),
    }
    for name, content in outputs.items():
        _write(out_dir / name, content)

    summary = {
        "generated_dir": str(out_dir.relative_to(_repo_root())),
        "file_count": len(outputs),
        "files": sorted(outputs),
    }
    _write(out_dir / "latex_tables_metadata.json", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_protocol.yaml")
    args = parser.parse_args()
    print(json.dumps(export_latex_tables(args.config), indent=2))


if __name__ == "__main__":
    main()
