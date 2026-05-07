"""Plot helpers for benchmark summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save(fig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_task_scores(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Grouped bar chart of mean scores by task and model."""
    if episodes.empty:
        return
    summary = episodes.groupby(["policy", "model_name", "task_id"], as_index=False)["score"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=summary, x="model_name", y="score", hue="task_id", ax=ax)
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Model / Approach")
    ax.set_title("LatentGoalOps Mean Task Score by Model")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_path)


def plot_score_distributions(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Box plot of score distributions by task."""
    if episodes.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=episodes, x="task_id", y="score", hue="model_name", ax=ax)
    ax.set_title("Score Distribution by Task")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, output_path)


def plot_heatmap(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Heatmap of mean scores."""
    if episodes.empty:
        return
    heatmap = (
        episodes.groupby(["model_name", "task_id"], as_index=False)["score"]
        .mean()
        .pivot(index="model_name", columns="task_id", values="score")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data=heatmap, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title("Mean Score Heatmap")
    _save(fig, output_path)


def plot_cost_vs_score(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot of average cost vs score."""
    if episodes.empty:
        return
    summary = (
        episodes.groupby(["policy", "model_name"], as_index=False)[["score", "cost_usd"]]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=summary, x="cost_usd", y="score", hue="model_name", style="policy", s=140, ax=ax)
    ax.set_title("Average Cost vs Average Score")
    _save(fig, output_path)


def plot_step_reward_trajectories(steps: pd.DataFrame, output_path: str | Path) -> None:
    """Mean reward trajectory over step index."""
    if steps.empty:
        return
    summary = (
        steps.groupby(["policy", "model_name", "task_id", "step_index"], as_index=False)["reward"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=summary,
        x="step_index",
        y="reward",
        hue="model_name",
        style="task_id",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Mean Reward Trajectory by Step")
    _save(fig, output_path)


def plot_token_usage(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Token usage bar chart."""
    if episodes.empty:
        return
    summary = (
        episodes.groupby(["policy", "model_name"], as_index=False)[["input_tokens", "output_tokens"]]
        .mean()
    )
    melted = summary.melt(id_vars=["policy", "model_name"], value_vars=["input_tokens", "output_tokens"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="model_name", y="value", hue="variable", ax=ax)
    ax.set_title("Average Token Usage per Episode")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_path)


def plot_parse_fallbacks(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Fallback frequency chart for model-output parsing."""
    if episodes.empty or "parse_fallback" not in episodes.columns:
        return
    columns = ["parse_fallback"]
    if "parse_repaired" in episodes.columns:
        columns.append("parse_repaired")
    if "empty_fallback_episode" in episodes.columns:
        columns.append("empty_fallback_episode")
    if "response_cap_hit" in episodes.columns:
        columns.append("response_cap_hit")
    summary = episodes.groupby(["policy", "model_name"], as_index=False)[columns].mean()
    summary = summary.melt(id_vars=["policy", "model_name"], value_vars=columns, var_name="outcome", value_name="rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="model_name", y="rate", hue="outcome", ax=ax)
    ax.set_title("Output Validity / Cap-Hit Rate")
    ax.set_ylabel("Episode fraction")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_path)


def plot_adaptation_distribution(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Task 3 adaptation-score distribution."""
    if episodes.empty or "subscore_adaptation" not in episodes.columns:
        return
    task3 = episodes[
        (episodes["task_id"] == "task3_startup_week")
        & episodes["subscore_adaptation"].notna()
    ]
    if "adaptation_scored" in task3.columns:
        task3 = task3[task3["adaptation_scored"] == 1]
    if task3.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=task3, x="model_name", y="subscore_adaptation", hue="policy", ax=ax)
    ax.set_title("Task 3 Adaptation Score Distribution")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_path)


def plot_coherence_vs_score(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Task 3 coherence-quality scatter."""
    if episodes.empty or "subscore_coherence" not in episodes.columns:
        return
    task3 = episodes[episodes["task_id"] == "task3_startup_week"]
    if task3.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=task3,
        x="subscore_coherence",
        y="score",
        hue="model_name",
        style="policy",
        s=120,
        ax=ax,
    )
    ax.set_title("Task 3 Coherence vs Final Score")
    _save(fig, output_path)


def plot_oracle_gap(episodes: pd.DataFrame, output_path: str | Path) -> None:
    """Gap to oracle by task and model."""
    if episodes.empty:
        return
    oracle = (
        episodes[episodes["policy"] == "oracle"]
        .groupby("task_id", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "oracle_score"})
    )
    if oracle.empty:
        return
    merged = episodes.groupby(["policy", "model_name", "task_id"], as_index=False)["score"].mean().merge(
        oracle, on="task_id", how="left"
    )
    merged["oracle_gap"] = merged["oracle_score"] - merged["score"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=merged, x="model_name", y="oracle_gap", hue="task_id", ax=ax)
    ax.set_title("Gap to Oracle by Task")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_path)
