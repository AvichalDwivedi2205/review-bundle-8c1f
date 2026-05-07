"""Generate paper-oriented figures from frozen paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from latentgoalops.analysis.make_paper_tables import generate_tables


MODEL_ORDER = [
    "qwen3.5:2b",
    "qwen3.5:9b",
    "qwen3.5:27b",
    "deepseek-r1:1.5b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "gemma4:e2b",
    "gemma4:e4b",
    "gemma4:26b",
    "gpt-oss:20b",
    "gpt-oss-120b",
]

FAMILY_COLORS = {
    "qwen": "#2f6f73",
    "deepseek-r1": "#c65f2e",
    "gemma4": "#7a5c9e",
    "gpt-oss": "#2f4f7f",
}

FAMILY_DISPLAY_NAMES = {
    "qwen": "Qwen3.5",
    "deepseek-r1": "DeepSeek-R1",
    "gemma4": "Gemma4",
    "gpt-oss": "GPT-OSS",
}

TASK_LABELS = {
    "task3_startup_week": "Task 3\nStartup week",
    "task6_incident_response_week": "Task 6\nIncident week",
    "task7_quarterly_headcount_plan": "Task 7\nHeadcount",
}

SUBSCORE_LABELS = {
    "final_utility": "Utility",
    "adaptation": "Adaptation",
    "coherence": "Coherence",
    "constraints": "Constraints",
}

MODEL_LABEL_OVERRIDES = {
    "openai-gpt-oss-20b": "GPT-OSS 20B",
    "openai-gpt-oss-120b": "GPT-OSS 120B",
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
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_resolved_protocol(config_path: str | Path) -> dict:
    from latentgoalops.paper.run_paper_suite import generate_paper_suite

    suite = generate_paper_suite(config_path)
    path = Path(suite["metadata_dir"]) / "paper_protocol_resolved.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _table_path(protocol: dict, stem: str) -> Path:
    return _repo_root() / protocol["paper"]["tables_dir"] / f"{stem}.csv"


def _figures_dir(protocol: dict) -> Path:
    path = _repo_root() / protocol["paper"]["figures_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_order(models: pd.Series) -> list[str]:
    present = set(models.dropna().astype(str))
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _display_model_label(label: str) -> str:
    return MODEL_LABEL_OVERRIDES.get(str(label), str(label))


def _family_key_from_model(label: str) -> str:
    text = str(label)
    if text.startswith("qwen"):
        return "qwen"
    if text.startswith("deepseek-r1"):
        return "deepseek-r1"
    if text.startswith("gemma4"):
        return "gemma4"
    if "gpt-oss" in text:
        return "gpt-oss"
    return text


def _transition_display_label(from_label: str, to_label: str) -> str:
    family_key = _family_key_from_model(from_label)
    family_prefix = {
        "qwen": "Qwen",
        "deepseek-r1": "DeepSeek",
        "gemma4": "Gemma",
        "gpt-oss": "GPT-OSS",
    }.get(family_key, _display_model_label(from_label).split()[0])
    from_size = _display_model_label(from_label).split()[-1]
    to_size = _display_model_label(to_label).split()[-1]
    return f"{family_prefix} {from_size}->{to_size}"


def _yerr_from_ci(frame: pd.DataFrame, mean_col: str, low_col: str, high_col: str) -> np.ndarray:
    lower = []
    upper = []
    for _, row in frame.iterrows():
        mean_value = row.get(mean_col)
        low_value = row.get(low_col)
        high_value = row.get(high_col)
        if pd.isna(mean_value) or pd.isna(low_value) or pd.isna(high_value):
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(float(mean_value) - float(low_value))
            upper.append(float(high_value) - float(mean_value))
    return np.array([lower, upper], dtype=float)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _box(axis: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.22) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.13,
        boxstyle="round,pad=0.025,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=color,
        alpha=0.95,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + 0.065, text, ha="center", va="center", fontsize=10, color="#1f1f1f")


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#333333") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def _plot_benchmark_overview(output_dir: Path) -> list[str]:
    fig, axis = plt.subplots(figsize=(10.8, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(axis, (0.04, 0.62), "Hidden latent\nobjective", "#d9c5a5")
    _box(axis, (0.04, 0.25), "Operating world\nstate", "#b8d2c7")
    _box(axis, (0.34, 0.43), "Public observation\nredacted schema", "#c7d5e8", width=0.24)
    _box(axis, (0.65, 0.43), "Agent decision\nstructured action", "#e3c3b7", width=0.23)
    _box(axis, (0.66, 0.12), "Delayed effects\nand silent shift", "#dbc7e5", width=0.23)
    _box(axis, (0.66, 0.74), "Deterministic grader\nutility/adaptation", "#cfd7b6", width=0.23)
    _arrow(axis, (0.26, 0.685), (0.34, 0.51))
    _arrow(axis, (0.26, 0.315), (0.34, 0.48))
    _arrow(axis, (0.58, 0.50), (0.65, 0.50))
    _arrow(axis, (0.76, 0.43), (0.76, 0.25))
    _arrow(axis, (0.66, 0.19), (0.26, 0.30), "#6a4c7c")
    _arrow(axis, (0.78, 0.56), (0.78, 0.74))
    axis.text(
        0.5,
        0.94,
        "LatentGoalOps evaluates whether agents infer and adapt to hidden business objectives",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    axis.text(
        0.5,
        0.04,
        "The agent only sees public observations. The latent objective and delayed consequences shape future rewards.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    return _save(fig, output_dir, "benchmark_overview")


def _plot_task_timelines(output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 5.8), sharex=True)
    tasks = [
        ("Task 3: Startup week", "Prioritize initiatives", "Silent strategy reprioritization", "Renewals / account consequences"),
        ("Task 6: Incident response week", "Stabilize incident", "Silent recovery-goal shift", "Trust / governance fallout"),
    ]
    days = np.arange(1, 8)
    for axis, (title, early, shift, delayed) in zip(axes, tasks, strict=True):
        axis.set_xlim(0.6, 7.4)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.hlines(0.5, 1, 7, color="#555555", linewidth=2)
        axis.scatter(days, [0.5] * len(days), s=90, color="#547d87", zorder=3)
        for day in days:
            axis.text(day, 0.34, f"Day {day}", ha="center", va="top", fontsize=9)
        axis.axvline(4, ymin=0.24, ymax=0.86, color="#c76f54", linestyle="--", linewidth=1.5)
        axis.text(2.1, 0.75, early, ha="center", va="center", fontsize=10)
        axis.text(4.15, 0.91, shift, ha="left", va="center", fontsize=10, color="#9a442f")
        axis.annotate(
            delayed,
            xy=(6.3, 0.5),
            xytext=(5.15, 0.68),
            arrowprops={"arrowstyle": "-|>", "color": "#6a4c7c"},
            fontsize=10,
            color="#4b3658",
        )
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
    axes[-1].set_xlabel("Episode horizon")
    fig.suptitle("Main benchmark tasks require sequential decisions under delayed consequences", y=1.02)
    return _save(fig, output_dir, "task_timelines")


def _setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.1,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        },
    )


def _plot_family_scaling(family_scaling: pd.DataFrame, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = family_scaling[family_scaling["split"] == split]
        for family, group in split_df.groupby("family"):
            ordered = group.sort_values("size_b")
            axis.errorbar(
                ordered["size_b"],
                ordered["overall_mean_score"],
                yerr=_yerr_from_ci(ordered, "overall_mean_score", "overall_ci_low", "overall_ci_high"),
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=FAMILY_DISPLAY_NAMES.get(family, family),
                color=FAMILY_COLORS.get(family),
                capsize=3,
            )
            for _, row in ordered.iterrows():
                axis.annotate(
                    f"{int(row['size_b'])}B",
                    (row["size_b"], row["overall_mean_score"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
        axis.set_xscale("log")
        axis.set_title(split.capitalize())
        axis.set_xlabel("Model size (B parameters, log scale)")
        axis.set_ylabel("Mean score on primary tasks")
        axis.set_ylim(0.30, 0.45)
    axes[0].legend(title="Family", loc="lower left", frameon=True)
    fig.suptitle("Frozen scaling profile on the primary latent-adaptation tasks (95% bootstrap CIs)", y=1.03)
    return _save(fig, output_dir, "family_scaling")


def _plot_baseline_separation(baseline_results: pd.DataFrame, output_dir: Path) -> list[str]:
    order = ["random", "visible heuristic", "myopic oracle"]
    present = [item for item in order if item in set(baseline_results["baseline"])]
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    split_colors = {"core": "#547d87", "heldout": "#d1864a"}
    width = 0.38
    x = np.arange(len(present))
    for offset, split in [(-width / 2, "core"), (width / 2, "heldout")]:
        split_df = (
            baseline_results[baseline_results["split"] == split]
            .set_index("baseline")
            .reindex(present)
            .reset_index()
        )
        axis.bar(
            x + offset,
            split_df["overall_mean_score"],
            width,
            label=split.capitalize(),
            color=split_colors[split],
            yerr=_yerr_from_ci(split_df, "overall_mean_score", "overall_ci_low", "overall_ci_high"),
            capsize=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(present)
    axis.set_title("Primary-task aggregate preserves baseline ordering")
    axis.set_xlabel("")
    axis.set_ylabel("Mean score on primary tasks")
    axis.set_ylim(0.25, 0.65)
    axis.legend(title="Split", frameon=True)
    return _save(fig, output_dir, "baseline_separation")


def _plot_overall_vs_native(family_scaling: pd.DataFrame, output_dir: Path) -> list[str]:
    order = _model_order(family_scaling["model"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True, sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = family_scaling[family_scaling["split"] == split].set_index("model").loc[order].reset_index()
        x = np.arange(len(split_df))
        width = 0.38
        axis.bar(x - width / 2, split_df["overall_mean_score"], width, label="Overall", color="#547d87")
        axis.bar(
            x + width / 2,
            split_df["strict_native_overall_mean_score"],
            width,
            label="Native-only",
            color="#d1864a",
            yerr=_yerr_from_ci(split_df, "strict_native_overall_mean_score", "strict_native_ci_low", "strict_native_ci_high"),
            capsize=3,
        )
        axis.errorbar(
            x - width / 2,
            split_df["overall_mean_score"],
            yerr=_yerr_from_ci(split_df, "overall_mean_score", "overall_ci_low", "overall_ci_high"),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1.0,
        )
        for index, row in split_df.iterrows():
            if row["strict_native_episode_count"] < 10:
                axis.text(
                    index + width / 2,
                    row["strict_native_overall_mean_score"] + 0.012,
                    f"n={int(row['strict_native_episode_count'])}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#6b3f1f",
                )
        axis.set_title(split.capitalize())
        axis.set_ylabel("Mean score")
        axis.set_ylim(0.28, 0.68)
        axis.legend(loc="upper right", frameon=True)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels([_display_model_label(item) for item in order], rotation=35, ha="right")
    fig.suptitle("Observed score vs. native-only score (native-only estimates marked when native n is small)", y=1.02)
    return _save(fig, output_dir, "overall_vs_native")


def _plot_action_reliability(family_scaling: pd.DataFrame, output_dir: Path) -> list[str]:
    order = _model_order(family_scaling["model"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = family_scaling[family_scaling["split"] == split].set_index("model").loc[order].reset_index()
        native = split_df["native_action_episode_rate"].fillna(0)
        empty = split_df["empty_fallback_episode_rate"].fillna(0)
        cap = split_df["response_cap_hit_episode_rate"].fillna(0)
        x = np.arange(len(split_df))
        width = 0.26
        axis.bar(x - width, native, width, label="Native action", color="#3d7a68")
        axis.bar(x, empty, width, label="Empty fallback", color="#b85d4b")
        axis.bar(x + width, cap, width, label="Cap hit", color="#d9a441")
        axis.set_title(split.capitalize())
        axis.set_ylabel("Episode rate")
        axis.set_ylim(0, 1.02)
        axis.legend(loc="upper right", frameon=True, ncol=3)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels([_display_model_label(item) for item in order], rotation=35, ha="right")
    fig.suptitle("Structured-action reliability remains a major failure mode in the frozen ladder", y=1.02)
    return _save(fig, output_dir, "action_reliability")


def _plot_token_pressure_vs_native_rate(family_scaling: pd.DataFrame, output_dir: Path) -> list[str]:
    df = family_scaling.copy()
    df["tokens_per_episode"] = df["total_output_tokens"] / df["episode_count"].clip(lower=1)
    df["display_label"] = df["model"].map(_display_model_label)
    marker_map = {"core": "o", "heldout": "s"}
    present_families: set[str] = set()
    fig, axis = plt.subplots(figsize=(9.8, 5.4))
    for split in ["core", "heldout"]:
        split_df = df[df["split"] == split]
        for family, group in split_df.groupby("family"):
            present_families.add(str(family))
            axis.scatter(
                group["tokens_per_episode"],
                group["native_action_episode_rate"],
                s=95,
                marker=marker_map[split],
                color=FAMILY_COLORS.get(family),
                edgecolors="#222222",
                linewidths=0.5,
                alpha=0.9,
            )
            for _, row in group.iterrows():
                axis.annotate(
                    row["display_label"],
                    (row["tokens_per_episode"], row["native_action_episode_rate"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                )
    axis.set_xscale("log")
    axis.set_xlabel("Output tokens per episode (log scale)")
    axis.set_ylabel("Native action episode rate")
    axis.set_ylim(-0.02, 1.05)
    axis.set_title("Longer outputs strongly track interface collapse in the frozen runs")
    handles = []
    labels = []
    for split, marker in marker_map.items():
        handles.append(Line2D([], [], color="#444444", marker=marker, linestyle="", markersize=8))
        labels.append(split.capitalize())
    legend1 = axis.legend(handles, labels, title="Split", loc="lower left", frameon=True)
    axis.add_artist(legend1)
    family_order = [family for family in FAMILY_COLORS if family in present_families]
    family_handles = [
        Line2D([], [], color=FAMILY_COLORS[family], marker="o", linestyle="", markersize=8)
        for family in family_order
    ]
    family_labels = [FAMILY_DISPLAY_NAMES.get(family, family) for family in family_order]
    axis.legend(family_handles, family_labels, title="Family", loc="upper right", frameon=True)
    return _save(fig, output_dir, "token_pressure_vs_native_rate")


def _plot_shifted_vs_unshifted(family_scaling: pd.DataFrame, output_dir: Path) -> list[str]:
    order = _model_order(family_scaling["model"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True, sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = family_scaling[family_scaling["split"] == split].set_index("model").loc[order].reset_index()
        x = np.arange(len(split_df))
        width = 0.38
        axis.bar(x - width / 2, split_df["unshifted_overall_mean_score"], width, label="Unshifted", color="#61789a")
        axis.bar(
            x + width / 2,
            split_df["shifted_overall_mean_score"],
            width,
            label="Shifted",
            color="#c76f54",
        )
        axis.errorbar(
            x - width / 2,
            split_df["unshifted_overall_mean_score"],
            yerr=_yerr_from_ci(split_df, "unshifted_overall_mean_score", "unshifted_ci_low", "unshifted_ci_high"),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1.0,
        )
        axis.errorbar(
            x + width / 2,
            split_df["shifted_overall_mean_score"],
            yerr=_yerr_from_ci(split_df, "shifted_overall_mean_score", "shifted_ci_low", "shifted_ci_high"),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1.0,
        )
        axis.set_title(split.capitalize())
        axis.set_ylabel("Mean score")
        axis.set_ylim(0.28, 0.66)
        axis.legend(loc="upper right", frameon=True)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels([_display_model_label(item) for item in order], rotation=35, ha="right")
    fig.suptitle("Silent objective shifts remain materially harder than no-shift episodes", y=1.02)
    return _save(fig, output_dir, "shifted_vs_unshifted")


def _plot_scaling_transition_map(scaling_deltas: pd.DataFrame, output_dir: Path) -> list[str]:
    if scaling_deltas.empty:
        return []
    marker_map = {"core": "o", "heldout": "s"}
    present_families: set[str] = set()
    fig, axis = plt.subplots(figsize=(9.6, 5.6))
    axis.axhline(0, color="#555555", linewidth=1.0, linestyle="--")
    axis.axvline(0, color="#555555", linewidth=1.0, linestyle="--")
    for split in ["core", "heldout"]:
        split_df = scaling_deltas[scaling_deltas["split"] == split]
        for family, group in split_df.groupby("family"):
            present_families.add(str(family))
            axis.scatter(
                group["delta_native_action_episode_rate"],
                group["delta_strict_native_overall_mean_score"],
                s=110,
                marker=marker_map[split],
                color=FAMILY_COLORS.get(family),
                edgecolors="#222222",
                linewidths=0.6,
                alpha=0.92,
            )
            for _, row in group.iterrows():
                label = _transition_display_label(str(row["from_model"]), str(row["to_model"]))
                axis.annotate(
                    label,
                    (row["delta_native_action_episode_rate"], row["delta_strict_native_overall_mean_score"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                )
    axis.set_xlabel(r"$\Delta$ native action rate (larger model - smaller model)")
    axis.set_ylabel(r"$\Delta$ native-only score")
    axis.set_title("Family-size jumps separate native-policy regressions from interface regressions")
    axis.text(-0.92, 0.24, "Interface loss,\nnative-only gain", fontsize=9, color="#444444")
    axis.text(-0.25, -0.10, "Interface and/or\npolicy regression", fontsize=9, color="#444444")
    axis.text(0.03, 0.02, "Native-policy gain", fontsize=9, color="#444444")
    split_handles = [
        Line2D([], [], color="#444444", marker=marker, linestyle="", markersize=8)
        for marker in marker_map.values()
    ]
    split_labels = [label.capitalize() for label in marker_map]
    legend1 = axis.legend(split_handles, split_labels, title="Split", loc="lower right", frameon=True)
    axis.add_artist(legend1)
    family_order = [family for family in FAMILY_COLORS if family in present_families]
    family_handles = [
        Line2D([], [], color=FAMILY_COLORS[family], marker="o", linestyle="", markersize=8)
        for family in family_order
    ]
    axis.legend(
        family_handles,
        [FAMILY_DISPLAY_NAMES.get(family, family) for family in family_order],
        title="Family",
        loc="upper right",
        frameon=True,
    )
    return _save(fig, output_dir, "scaling_transition_map")


def _plot_task_heatmap(task_breakdown: pd.DataFrame, output_dir: Path) -> list[str]:
    main_df = task_breakdown[task_breakdown["task_bucket"] == "main"].copy()
    order = _model_order(main_df["model"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 6.2), sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = main_df[main_df["split"] == split]
        pivot = split_df.pivot(index="model", columns="task_id", values="overall_mean_score").reindex(order)
        pivot = pivot.rename(columns=TASK_LABELS)
        pivot.index = [_display_model_label(item) for item in pivot.index]
        sns.heatmap(
            pivot,
            ax=axis,
            vmin=0.25,
            vmax=0.55,
            cmap="crest",
            annot=True,
            fmt=".3f",
            cbar=split == "heldout",
            linewidths=0.5,
            linecolor="white",
        )
        axis.set_title(split.capitalize())
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.suptitle("Task-level scores on the frozen main benchmark core", y=1.02)
    return _save(fig, output_dir, "task_heatmap")


def _plot_subscore_heatmap(subscores: pd.DataFrame, output_dir: Path) -> list[str]:
    main_df = subscores[subscores["task_bucket"] == "main"].copy()
    metrics = [column for column in SUBSCORE_LABELS if column in main_df.columns]
    averaged = main_df.groupby(["model", "split"], as_index=False)[metrics].mean()
    order = _model_order(averaged["model"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.2), sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = averaged[averaged["split"] == split]
        pivot = split_df.set_index("model")[metrics].reindex(order).rename(columns=SUBSCORE_LABELS)
        pivot.index = [_display_model_label(item) for item in pivot.index]
        sns.heatmap(
            pivot,
            ax=axis,
            vmin=0.0,
            vmax=1.0,
            cmap="rocket_r",
            annot=True,
            fmt=".2f",
            cbar=split == "heldout",
            linewidths=0.5,
            linecolor="white",
        )
        axis.set_title(split.capitalize())
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.suptitle("Subscore decomposition across utility, adaptation, coherence, and constraints", y=1.02)
    return _save(fig, output_dir, "subscore_heatmap")


def _plot_heldout_gap(heldout_gap: pd.DataFrame, output_dir: Path) -> list[str]:
    order = _model_order(heldout_gap["model"])
    df = heldout_gap.set_index("model").loc[order].reset_index()
    colors = ["#3d7a68" if value >= 0 else "#b85d4b" for value in df["heldout_gap"]]
    fig, axis = plt.subplots(figsize=(11.2, 4.4))
    axis.axhline(0, color="#333333", linewidth=1)
    axis.bar([_display_model_label(item) for item in df["model"]], df["heldout_gap"], color=colors)
    axis.set_ylabel("Heldout - core score")
    axis.set_title("Heldout generalization gap on main tasks")
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    return _save(fig, output_dir, "heldout_gap")


def _plot_leakage_audit(leakage_results: pd.DataFrame, output_dir: Path) -> list[str]:
    main_df = leakage_results[leakage_results["task_bucket"] == "main"].copy()
    view_order = [
        "full_observation",
        "sender_only",
        "metadata_only",
        "impact_summary_only",
        "numeric_only",
        "stakeholder_order_only",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = main_df[main_df["split"] == split]
        sns.barplot(
            split_df,
            x="view",
            y="centroid_accuracy",
            hue="canonical_task_id",
            order=view_order,
            palette="crest",
            ax=axis,
        )
        axis.axhline(0.25, color="#8b4c39", linestyle="--", linewidth=1.3, label="Chance (4-way)")
        axis.set_title(split.capitalize())
        axis.set_xlabel("")
        axis.set_ylabel("Latent-goal probe accuracy")
        axis.set_ylim(0, 0.42)
        axis.tick_params(axis="x", rotation=35)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=True, fontsize=9)
    fig.suptitle("Restricted-view leakage probes stay near 4-way chance on the primary tasks", y=1.02)
    return _save(fig, output_dir, "leakage_audit")


def _plot_reliability_ci(reliability_results: pd.DataFrame, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = reliability_results[reliability_results["split"] == split].copy().reset_index(drop=True)
        x = np.arange(len(split_df))
        means = split_df["mean_repeat_score"].fillna(0.0)
        lower = means - split_df["repeat_ci_low"].fillna(means)
        upper = split_df["repeat_ci_high"].fillna(means) - means
        axis.bar(x, means, color="#547d87", alpha=0.85)
        axis.errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor="#1f1f1f", capsize=4, linewidth=1.2)
        axis.set_xticks(x)
        axis.set_xticklabels([_display_model_label(item) for item in split_df["model_name"]], rotation=25, ha="right")
        axis.set_title(split.capitalize())
        axis.set_xlabel("")
        axis.set_ylabel("Repeat-mean score on main tasks")
        axis.set_ylim(0.34, 0.47)
        for idx, row in enumerate(split_df.to_dict(orient="records")):
            axis.text(idx, row["mean_repeat_score"] + 0.004, f"n={int(row['repeat_count'])}", ha="center", fontsize=8)
    fig.suptitle("Repeat-based reliability is currently well estimated only for GPT-OSS 20B", y=1.03)
    return _save(fig, output_dir, "reliability_ci")


def _plot_adaptation_audit(adaptation_audit: pd.DataFrame, output_dir: Path) -> list[str]:
    main_df = adaptation_audit[adaptation_audit["task_id"].isin(["task3_startup_week", "task6_incident_response_week"])].copy()
    if main_df.empty:
        return []
    main_df["task_split"] = main_df["split"].str.capitalize() + "\n" + main_df["task_id"].map(TASK_LABELS)
    order = _model_order(main_df["model"])
    column_order = [
        "Core\nTask 3\nStartup week",
        "Core\nTask 6\nIncident week",
        "Heldout\nTask 3\nStartup week",
        "Heldout\nTask 6\nIncident week",
    ]
    mean_pivot = (
        main_df.pivot(index="model", columns="task_split", values="mean_adaptation_score")
        .reindex(order)
        .reindex(columns=column_order)
    )
    zero_pivot = (
        main_df.pivot(index="model", columns="task_split", values="zero_adaptation_episode_rate")
        .reindex(order)
        .reindex(columns=column_order)
    )
    mean_pivot.index = [_display_model_label(item) for item in mean_pivot.index]
    zero_pivot.index = [_display_model_label(item) for item in zero_pivot.index]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), sharey=True)
    sns.heatmap(
        mean_pivot,
        ax=axes[0],
        vmin=0.0,
        vmax=0.35,
        cmap="mako",
        annot=True,
        fmt=".2f",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    axes[0].set_title("Mean adaptation score")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.heatmap(
        zero_pivot,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
        cmap="rocket_r",
        annot=True,
        fmt=".2f",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    axes[1].set_title("Zero-adaptation episode rate")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle(
        "Adaptation failures are concentrated in specific task/split slices rather than uniformly across the benchmark",
        y=1.02,
    )
    return _save(fig, output_dir, "adaptation_audit")


def _plot_shell_controlled_diagnostic(shell_controlled: pd.DataFrame, output_dir: Path) -> list[str]:
    if shell_controlled.empty:
        return []
    order = _model_order(shell_controlled["model"])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for axis, split in zip(axes, ["core", "heldout"], strict=True):
        split_df = shell_controlled[shell_controlled["split"] == split].copy()
        if split_df.empty:
            axis.axis("off")
            continue
        split_df["model"] = pd.Categorical(split_df["model"], categories=order, ordered=True)
        split_df = split_df.sort_values("model")
        x = np.arange(len(split_df))
        width = 0.24
        axis.bar(x - width, split_df["frozen_overall_mean_score"], width, label="Frozen overall", color="#547d87")
        axis.bar(x, split_df["diagnostic_overall_mean_score"], width, label="Shell overall", color="#a86a54")
        axis.bar(
            x + width,
            split_df["diagnostic_assisted_overall_mean_score"],
            width,
            label="Shell assisted",
            color="#d8a34c",
        )
        for idx, row in split_df.reset_index(drop=True).iterrows():
            assisted = row["diagnostic_assisted_overall_mean_score"]
            axis.text(
                idx,
                max(row["diagnostic_overall_mean_score"], assisted if pd.notna(assisted) else 0.0) + 0.012,
                f"native={row['diagnostic_native_action_episode_rate']:.2f}\nrepair={int(row['diagnostic_parse_repaired_episode_count'] or 0)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4a3227",
            )
        axis.set_title(split.capitalize())
        axis.set_xticks(x)
        axis.set_xticklabels([_display_model_label(item) for item in split_df["model"]], rotation=20, ha="right")
        axis.set_ylabel("Mean score")
        axis.set_ylim(0.20, 0.70)
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle(
        "Shell-controlled appendix diagnostic: lightweight parse repair only partially rescues Qwen-27B and does not rescue Gemma-26B",
        y=1.03,
    )
    return _save(fig, output_dir, "shell_controlled_diagnostic")


def generate_figures(config_path: str | Path) -> dict[str, object]:
    generate_tables(config_path)
    protocol = _load_resolved_protocol(config_path)
    output_dir = _figures_dir(protocol)
    _setup_style()
    family_scaling = pd.read_csv(_table_path(protocol, "family_scaling"))
    baseline_results = pd.read_csv(_table_path(protocol, "baseline_results"))
    task_breakdown = pd.read_csv(_table_path(protocol, "task_breakdown"))
    subscores = pd.read_csv(_table_path(protocol, "subscore_breakdown"))
    heldout_gap = pd.read_csv(_table_path(protocol, "heldout_gap"))
    leakage_results = pd.read_csv(_table_path(protocol, "leakage_results"))
    reliability_results = pd.read_csv(_table_path(protocol, "reliability_results"))
    adaptation_audit = pd.read_csv(_table_path(protocol, "adaptation_audit"))
    shell_controlled = pd.read_csv(_table_path(protocol, "shell_controlled_diagnostic"))
    scaling_deltas = pd.read_csv(_table_path(protocol, "scaling_deltas"))

    figure_builders: dict[str, Callable[[], list[str]]] = {
        "benchmark_overview": lambda: _plot_benchmark_overview(output_dir),
        "task_timelines": lambda: _plot_task_timelines(output_dir),
        "baseline_separation": lambda: _plot_baseline_separation(baseline_results, output_dir),
        "family_scaling": lambda: _plot_family_scaling(family_scaling, output_dir),
        "overall_vs_native": lambda: _plot_overall_vs_native(family_scaling, output_dir),
        "action_reliability": lambda: _plot_action_reliability(family_scaling, output_dir),
        "token_pressure_vs_native_rate": lambda: _plot_token_pressure_vs_native_rate(family_scaling, output_dir),
        "shifted_vs_unshifted": lambda: _plot_shifted_vs_unshifted(family_scaling, output_dir),
        "scaling_transition_map": lambda: _plot_scaling_transition_map(scaling_deltas, output_dir),
        "task_heatmap": lambda: _plot_task_heatmap(task_breakdown, output_dir),
        "subscore_heatmap": lambda: _plot_subscore_heatmap(subscores, output_dir),
        "heldout_gap": lambda: _plot_heldout_gap(heldout_gap, output_dir),
        "leakage_audit": lambda: _plot_leakage_audit(leakage_results, output_dir),
        "reliability_ci": lambda: _plot_reliability_ci(reliability_results, output_dir),
        "adaptation_audit": lambda: _plot_adaptation_audit(adaptation_audit, output_dir),
        "shell_controlled_diagnostic": lambda: _plot_shell_controlled_diagnostic(shell_controlled, output_dir),
    }
    generated: dict[str, list[str]] = {}
    for name, builder in figure_builders.items():
        generated[name] = [str(Path(path).relative_to(_repo_root())) for path in builder()]
    summary = {
        "figures_dir": str(output_dir.relative_to(_repo_root())),
        "figure_count": len(generated),
        "figures": generated,
    }
    (output_dir / "figures_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_protocol.yaml")
    args = parser.parse_args()
    print(json.dumps(generate_figures(args.config), indent=2))


if __name__ == "__main__":
    main()
