"""Main environment implementation for LatentGoalOps."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.analysis.belief import expected_calibration_error, score_belief_report
from latentgoalops.models import (
    FeedbackLabel,
    BeliefReport,
    GoalArchetype,
    GraderResult,
    ItemLabelAssignment,
    ItemPriorityAssignment,
    LatentGoalOpsAction,
    LatentGoalOpsObservation,
    LatentGoalOpsState,
    MessagingAction,
    PublicDecisionLedgerEntry,
    PublicInitiativeItem,
    PublicTemporalEffectRecord,
    SupportPolicy,
    TaskDescriptor,
    TaskId,
)
from latentgoalops.server.public_reasoning import (
    build_public_impact_summary,
    dominant_visible_focus,
    infer_goal_hint_from_evidence,
    visible_item_proxy,
)
from latentgoalops.server.config import load_config
from latentgoalops.server.grader import (
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
    grade_task5,
    grade_task6,
    grade_task7,
    latent_score_from_dashboard,
    trajectory_coherence,
)
from latentgoalops.server.hidden_goals import (
    HiddenGoal,
    active_goal_name,
    belief_target,
    compute_utility,
    sample_hidden_goal,
    snapshot_hidden_goal,
)
from latentgoalops.server.rewards import CHANNELS, compute_proxy_reward, strategy_embedding
from latentgoalops.server.tasks.task1_feedback import build_task1_episode
from latentgoalops.server.tasks.task2_prioritization import (
    build_task2_episode,
    selection_value,
)
from latentgoalops.server.tasks.task4_capital_allocation import (
    allocation_value,
    build_task4_episode,
)
from latentgoalops.server.tasks.task3_startup_week import (
    apply_task3_action,
    build_task3_episode,
    build_task3_view,
    evaluate_task3_action_value,
    solve_task3_oracle_action,
)
from latentgoalops.server.tasks.task6_incident_response_week import (
    build_task6_episode,
    build_task6_view,
)
from latentgoalops.server.tasks.task7_quarterly_headcount_plan import (
    apply_task7_action,
    build_task7_episode,
    build_task7_view,
    evaluate_task7_action_value,
    solve_task7_oracle_action,
)
from latentgoalops.server.tasks.task5_crisis_response import (
    build_task5_episode,
    evaluate_task5_action_value,
    solve_task5_oracle_action,
)
from latentgoalops.server.world import build_world


class LatentGoalOpsEnvironment(
    Environment[LatentGoalOpsAction, LatentGoalOpsObservation, LatentGoalOpsState]
):
    """OpenEnv environment for latent-goal startup operations."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, experiment_config: ExperimentConfig | None = None) -> None:
        super().__init__()
        self._config = experiment_config or ExperimentConfig()
        self._rng = random.Random(0)
        self._seed = 0
        self._state = LatentGoalOpsState(episode_id=str(uuid4()), step_count=0)
        self._hidden_goal: HiddenGoal | None = None
        self._task_id: TaskId | None = None
        self._episode: dict | None = None
        self._last_grade: GraderResult | None = None
        self._previous_action: LatentGoalOpsAction | None = None
        self._previous_strategy_signature: dict[str, float] | None = None

    @staticmethod
    def describe_tasks() -> list[TaskDescriptor]:
        """Static task catalog for docs and API exposure."""
        return [
            TaskDescriptor(
                task_id=TaskId.TASK1,
                difficulty="easy",
                objective="Label, prioritize, and escalate customer feedback while inferring the hidden business objective.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK2,
                difficulty="medium",
                objective="Select the best roadmap slice under budget when stakeholder clues only indirectly reveal the objective.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK3,
                difficulty="hard",
                objective="Operate a startup across a dated calendar, reason about delayed effects, detect a silent goal shift, and re-align policy without direct supervision.",
                horizon=10,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK4,
                difficulty="medium",
                objective="Allocate capital across visible programs with diminishing returns, interactions, and indirect stakeholder clues about what outcome matters most.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK5,
                difficulty="hard",
                objective="Assemble one crisis-response package from initiatives plus pricing, messaging, and support policy while inferring the hidden business objective.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK6,
                difficulty="hard",
                objective="Respond across multiple incident days, track recovery evidence in memory, and adapt as latent leadership priorities shift under delayed effects.",
                horizon=7,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK7,
                difficulty="hard",
                objective="Allocate quarterly headcount under delayed staffing effects, team bottlenecks, and changing latent objectives over a multi-step planning horizon.",
                horizon=4,
            ),
        ]

    def get_metadata(self) -> EnvironmentMetadata:
        """Expose environment metadata and README content."""
        readme_path = Path(__file__).resolve().parents[3] / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
        return EnvironmentMetadata(
            name="LatentGoalOps",
            description="A realistic startup-operations benchmark for latent objective inference and non-stationary goal adaptation.",
            readme_content=readme_content,
            version="0.1.0",
            author="Anonymous Authors",
        )

    @staticmethod
    def _infer_goal_hint_from_evidence(evidence: str) -> str:
        """Infer a rough visible objective from narrative evidence only."""
        return infer_goal_hint_from_evidence(evidence)

    def _impact_summary(self, item) -> str:
        account_names = []
        if self._episode is not None:
            accounts_by_id = {account.account_id: account for account in self._episode.get("accounts", [])}
            account_names = [
                accounts_by_id[account_id].company_name
                for account_id in item.beneficiary_account_ids
                if account_id in accounts_by_id
            ][:2]
        return build_public_impact_summary(
            item_id=item.item_id,
            title=item.title,
            kpi_deltas=item.kpi_deltas,
            beneficiary_segments=item.beneficiary_segments,
            beneficiary_account_names=account_names,
            lag_steps=item.lag_steps,
            implementation_risk=item.implementation_risk,
            policy_tags=item.policy_tags,
            delivery_note=item.delivery_note,
        )

    def _public_initiative_view(self, item) -> object:
        impact_summary = item.impact_summary if self._task_id == TaskId.TASK2 and item.impact_summary else self._impact_summary(item)
        return PublicInitiativeItem(
            item_id=item.item_id,
            title=item.title,
            description=item.description,
            cost=item.cost,
            uncertainty_band=item.uncertainty_band,
            stakeholder_tag=item.stakeholder_tag,
            lag_steps=item.lag_steps,
            effect_window=item.effect_window,
            delivery_note=item.delivery_note,
            beneficiary_segments=list(item.beneficiary_segments),
            beneficiary_account_ids=list(item.beneficiary_account_ids),
            implementation_risk=item.implementation_risk,
            policy_tags=list(item.policy_tags),
            requires_item_ids=list(item.requires_item_ids),
            conflicts_with_ids=list(item.conflicts_with_ids),
            synergy_item_ids=list(item.synergy_item_ids),
            risk_notes=list(item.risk_notes),
            allocation_unit=item.allocation_unit,
            allocation_max=item.allocation_max,
            saturation_point=item.saturation_point,
            impact_summary=impact_summary,
        )

    @staticmethod
    def _public_effect_view(effect) -> object:
        return PublicTemporalEffectRecord(
            effect_id=effect.effect_id,
            decision_id=effect.decision_id,
            source_type=effect.source_type,
            source_id=effect.source_id,
            summary=effect.summary,
            affected_account_ids=list(effect.affected_account_ids),
            affected_team_ids=list(effect.affected_team_ids),
            scheduled_for_step=effect.scheduled_for_step,
            scheduled_for_date=effect.scheduled_for_date,
            realized_step=effect.realized_step,
            realized_date=effect.realized_date,
        )

    @staticmethod
    def _public_decision_view(entry) -> object:
        return PublicDecisionLedgerEntry(
            decision_id=entry.decision_id,
            step_index=entry.step_index,
            sim_date=entry.sim_date,
            chosen_initiatives=list(entry.chosen_initiatives),
            messaging_action=entry.messaging_action,
            pricing_change_pct=entry.pricing_change_pct,
            support_policy=entry.support_policy,
            rationale=entry.rationale,
            scheduled_effect_ids=list(entry.scheduled_effect_ids),
            realized_effect_ids=list(entry.realized_effect_ids),
            observed_alerts=list(entry.observed_alerts),
            governance_flags=list(entry.governance_flags),
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: object,
    ) -> LatentGoalOpsObservation:
        """Reset the environment to a clean episode."""
        self._seed = 0 if seed is None else int(seed)
        self._rng = random.Random(self._seed)
        self._task_id = TaskId(task_id or TaskId.TASK1.value)
        task_config = load_config("tasks.yaml")
        max_shift_step: int | None = None
        task7_reaction_shift_step: int | None = None
        if self._task_id == TaskId.TASK3:
            horizon = int(self._config.task3_horizon_override or task_config["task3"]["horizon"])
            max_shift_step = max(1, horizon - 2)
        elif self._task_id == TaskId.TASK6:
            horizon = int(task_config["task6"]["horizon"])
            max_shift_step = max(1, horizon - 2)
        elif self._task_id == TaskId.TASK7:
            horizon = int(task_config["task7"]["horizon"])
            max_shift_step = max(1, horizon - 2)
            task7_reaction_shift_step = 1 if horizon >= 3 else max_shift_step
        self._hidden_goal = sample_hidden_goal(
            self._rng,
            allow_shift=self._task_id in {TaskId.TASK3, TaskId.TASK6, TaskId.TASK7} and self._config.enable_hidden_shift,
            max_shift_step=max_shift_step,
        )
        if (
            self._task_id == TaskId.TASK7
            and self._hidden_goal.shift_step is not None
            and task7_reaction_shift_step is not None
        ):
            # Task 7 only has a small number of quarterly decisions, so late or drifting
            # shifts collapse the adaptation window into a single ambiguous step.
            self._hidden_goal.shift_step = min(self._hidden_goal.shift_step, task7_reaction_shift_step)
            self._hidden_goal.shift_type = "abrupt"
            self._hidden_goal.shift_duration_steps = 1
        self._last_grade = None
        self._previous_action = None
        world = build_world(random.Random(self._seed + 10_003), split=self._config.scenario_split)
        if self._task_id == TaskId.TASK1:
            low, high = task_config["task1"]["item_count_range"]
            self._episode = build_task1_episode(
                self._rng,
                self._hidden_goal,
                self._rng.randint(low, high),
                world,
            )
            max_steps = 1
            budget_remaining = 0.0
            capacity_remaining = 0.0
        elif self._task_id == TaskId.TASK2:
            budget = int(task_config["task2"]["sprint_budget"])
            self._episode = build_task2_episode(
                self._rng,
                self._hidden_goal,
                budget,
                world,
                split=self._config.scenario_split,
            )
            max_steps = 1
            budget_remaining = float(budget)
            capacity_remaining = 0.0
        elif self._task_id == TaskId.TASK3:
            horizon = int(self._config.task3_horizon_override or task_config["task3"]["horizon"])
            budget = float(task_config["task3"]["budget_per_episode"])
            capacity = float(task_config["task3"]["capacity_per_episode"])
            self._episode = build_task3_episode(
                self._rng,
                self._hidden_goal,
                horizon,
                budget,
                capacity,
                world,
                split=self._config.scenario_split,
            )
            self._episode["initial_budget"] = budget
            self._episode["initial_capacity"] = capacity
            self._episode["seed"] = self._seed
            self._episode["enable_delayed_effects"] = self._config.enable_delayed_effects
            self._episode["expose_decision_ledger"] = self._config.expose_decision_ledger
            max_steps = horizon
            budget_remaining = budget
            capacity_remaining = capacity
        elif self._task_id == TaskId.TASK6:
            horizon = int(task_config["task6"]["horizon"])
            budget = float(task_config["task6"]["budget_per_episode"])
            capacity = float(task_config["task6"]["capacity_per_episode"])
            self._episode = build_task6_episode(
                self._rng,
                self._hidden_goal,
                horizon,
                budget,
                capacity,
                world,
                split=self._config.scenario_split,
            )
            self._episode["initial_budget"] = budget
            self._episode["initial_capacity"] = capacity
            self._episode["seed"] = self._seed
            self._episode["enable_delayed_effects"] = self._config.enable_delayed_effects
            self._episode["expose_decision_ledger"] = self._config.expose_decision_ledger
            max_steps = horizon
            budget_remaining = budget
            capacity_remaining = capacity
        elif self._task_id == TaskId.TASK4:
            budget = int(task_config["task4"]["allocation_budget"])
            self._episode = build_task4_episode(
                self._rng,
                self._hidden_goal,
                budget,
                world,
                split=self._config.scenario_split,
            )
            max_steps = 1
            budget_remaining = float(budget)
            capacity_remaining = 0.0
        elif self._task_id == TaskId.TASK7:
            horizon = int(task_config["task7"]["horizon"])
            self._episode = build_task7_episode(
                self._rng,
                self._hidden_goal,
                horizon,
                world,
            )
            self._episode["enable_delayed_effects"] = self._config.enable_delayed_effects
            self._episode["expose_decision_ledger"] = self._config.expose_decision_ledger
            max_steps = horizon
            budget_remaining = float(self._episode["budget_remaining"])
            capacity_remaining = 0.0
        else:
            budget = float(task_config["task5"]["budget"])
            capacity = int(task_config["task5"]["capacity"])
            self._episode = build_task5_episode(
                self._rng,
                self._hidden_goal,
                budget,
                capacity,
                world,
                split=self._config.scenario_split,
            )
            max_steps = 1
            budget_remaining = budget
            capacity_remaining = float(capacity)

        self._state = LatentGoalOpsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task_id,
            max_steps=max_steps,
            sim_date=self._episode.get("start_date") if self._task_id in {TaskId.TASK3, TaskId.TASK6, TaskId.TASK7} else None,
            sim_day_label=(
                "Quarter 1"
                if self._task_id == TaskId.TASK7
                else ("Day 1" if self._task_id in {TaskId.TASK3, TaskId.TASK6} else None)
            ),
            budget_remaining=budget_remaining,
            capacity_remaining=capacity_remaining,
            decision_count=0,
            pending_effect_count=0,
            completed=False,
            cumulative_reward=0.0,
            last_score=None,
        )
        self._previous_strategy_signature = None
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: LatentGoalOpsAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> LatentGoalOpsObservation:
        """Take a step in the environment."""
        del timeout_s
        if self._episode is None or self._task_id is None or self._hidden_goal is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if action.task_id != self._task_id:
            raise ValueError(f"Action task_id={action.task_id.value} does not match active task {self._task_id.value}.")
        if self._state.completed:
            return self._build_observation(reward=0.0, done=True)

        if self._task_id == TaskId.TASK1:
            observation = self._step_task1(action)
        elif self._task_id == TaskId.TASK2:
            observation = self._step_task2(action)
        elif self._task_id == TaskId.TASK3:
            observation = self._step_task3(action)
        elif self._task_id == TaskId.TASK6:
            observation = self._step_task6(action)
        elif self._task_id == TaskId.TASK4:
            observation = self._step_task4(action)
        elif self._task_id == TaskId.TASK7:
            observation = self._step_task7(action)
        else:
            observation = self._step_task5(action)

        self._previous_action = action
        return observation

    def _step_task1(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        grade = grade_task1(
            true_labels=self._episode["true_labels"],
            true_priorities=self._episode["true_priorities"],
            oracle_escalations=self._episode["oracle_escalations"],
            action_payload=action.model_dump(mode="json"),
            hidden_goal=self._hidden_goal,
        )
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = grade.score
        return self._build_observation(
            reward=grade.score,
            done=True,
            metadata={"grader": grade.model_dump(mode="json")},
        )

    def _step_task2(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        backlog = self._episode["backlog"]
        valid_ids = {item.item_id for item in backlog}
        selected_ids = [item_id for item_id in action.selected_item_ids if item_id in valid_ids]
        spent_budget = sum(item.cost for item in backlog if item.item_id in selected_ids)
        budget = float(self._episode["sprint_budget"])
        invalid = spent_budget > budget + 1e-9
        if invalid:
            selected_ids = []
            spent_budget = 0.0
        agent_value = selection_value(
            selected_ids,
            backlog,
            int(round(budget)),
            self._hidden_goal.weights,
            self._hidden_goal,
            self._episode.get("company_profile"),
            step_index=0,
        )
        unused_budget_ratio = (budget - spent_budget) / max(budget, 1.0)
        grade = grade_task2(
            agent_value=agent_value,
            random_baseline=float(self._episode["random_value"]),
            oracle_value=float(self._episode["oracle_value"]),
            unused_budget_ratio=unused_budget_ratio,
        )
        reward = max(0.0, grade.score - (0.15 if invalid else 0.0))
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = reward
        self._state.budget_remaining = max(0.0, budget - spent_budget)
        return self._build_observation(
            reward=reward,
            done=True,
            metadata={
                "grader": grade.model_dump(mode="json"),
                "spent_budget": spent_budget,
                "invalid_budget": invalid,
            },
        )

    def _record_belief_metrics(self, action: LatentGoalOpsAction, step_index: int) -> dict[str, float]:
        assert self._episode is not None
        assert self._hidden_goal is not None
        target = belief_target(self._hidden_goal, step_index)
        row = score_belief_report(action.belief_report, target)
        field_names = [
            "archetype",
            "risk_posture",
            "planning_horizon",
            "segment_focus",
            "governance_strictness",
        ]
        belief_quality = sum(max(0.0, 1.0 - float(row[f"{field}_brier"])) for field in field_names) / len(field_names)
        self._episode.setdefault("belief_history", []).append(
            {
                "step_index": step_index,
                "belief_quality": round(float(belief_quality), 4),
                "shift_detected_confidence": float(
                    action.belief_report.shift_detected_confidence if action.belief_report else 0.0
                ),
                **row,
            }
        )
        return {"belief_quality": belief_quality, **row}

    def _belief_summary(self) -> dict[str, float]:
        assert self._episode is not None
        rows = list(self._episode.get("belief_history", []))
        if not rows:
            return {
                "belief_tracking": 0.0,
                "belief_ece": 0.0,
                "belief_nll": 0.0,
                "shift_detection_delay_steps": float(self._episode.get("horizon", 0)),
            }
        confidence_rows = []
        nll_values = []
        belief_quality = []
        for row in rows:
            belief_quality.append(float(row["belief_quality"]))
            for field in [
                "archetype",
                "risk_posture",
                "planning_horizon",
                "segment_focus",
                "governance_strictness",
            ]:
                confidence_rows.append((float(row[f"{field}_confidence"]), bool(row[f"{field}_correct"])))
                nll_values.append(float(row[f"{field}_nll"]))
        if self._hidden_goal is None or self._hidden_goal.shift_step is None:
            detection_delay = 0.0
        else:
            detection_delay = float(self._episode.get("horizon", 0))
            for row in rows:
                if row["step_index"] >= self._hidden_goal.shift_step and row["shift_detected_confidence"] >= 0.60:
                    detection_delay = float(row["step_index"] - self._hidden_goal.shift_step)
                    break
        return {
            "belief_tracking": sum(belief_quality) / len(belief_quality),
            "belief_ece": expected_calibration_error(confidence_rows),
            "belief_nll": sum(nll_values) / len(nll_values),
            "shift_detection_delay_steps": detection_delay,
        }

    def _base_objective(self) -> HiddenGoal:
        assert self._hidden_goal is not None
        return HiddenGoal(
            archetype=self._hidden_goal.archetype,
            weights=dict(self._hidden_goal.weights),
            alpha=self._hidden_goal.alpha,
            primary_kpi=self._hidden_goal.primary_kpi,
            risk_posture=self._hidden_goal.risk_posture,
            planning_horizon=self._hidden_goal.planning_horizon,
            segment_focus=self._hidden_goal.segment_focus,
            governance_strictness=self._hidden_goal.governance_strictness,
        )

    def _oracle_belief_report(self, step_index: int) -> BeliefReport:
        """Return a one-hot oracle posterior for belief-scored tasks."""
        assert self._hidden_goal is not None
        target = belief_target(self._hidden_goal, step_index)
        shift_detected = self._hidden_goal.shift_step is not None and step_index >= self._hidden_goal.shift_step
        return BeliefReport(
            archetype_probs={target["archetype"]: 1.0},
            risk_posture_probs={target["risk_posture"]: 1.0},
            planning_horizon_probs={target["planning_horizon"]: 1.0},
            segment_focus_probs={target["segment_focus"]: 1.0},
            governance_strictness_probs={target["governance_strictness"]: 1.0},
            shift_detected_confidence=1.0 if shift_detected else 0.0,
            notes="Oracle latent-state posterior.",
        )

    def _step_task3(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        current_step_index = self._episode["step_index"]
        belief_metrics = self._record_belief_metrics(action, current_step_index)
        current_objective = snapshot_hidden_goal(self._hidden_goal, current_step_index)
        current_weights = dict(current_objective.weights)
        oracle_action, oracle_decision_value = solve_task3_oracle_action(
            self._episode,
            current_weights,
            objective=current_objective,
        )
        del oracle_action
        agent_decision_value = evaluate_task3_action_value(
            self._episode,
            action,
            current_weights,
            objective=current_objective,
        )
        stale_quality: float | None = None
        if (
            self._hidden_goal.shift_step is not None
            and self._hidden_goal.shift_weights is not None
            and current_step_index >= self._hidden_goal.shift_step
        ):
            stale_objective = self._base_objective()
            stale_oracle_action, stale_oracle_value = solve_task3_oracle_action(
                self._episode,
                self._hidden_goal.weights,
                objective=stale_objective,
            )
            del stale_oracle_action
            stale_agent_value = evaluate_task3_action_value(
                self._episode,
                action,
                self._hidden_goal.weights,
                objective=stale_objective,
            )
            stale_quality = (
                1.0
                if stale_oracle_value <= 1e-9
                else max(0.0, min(1.0, stale_agent_value / stale_oracle_value))
            )
        item_kinds_by_id = {
            item.item_id: item.kind
            for item in self._episode["backlog"]
            if item.item_id not in self._episode["completed_ids"]
        }
        current_strategy_signature = strategy_embedding(action, item_kinds_by_id=item_kinds_by_id)
        decision_quality = (
            1.0
            if oracle_decision_value <= 1e-9
            else max(0.0, min(1.0, agent_decision_value / oracle_decision_value))
        )
        previous_metrics = self._episode["dashboard"].to_metric_vector()
        step_result = apply_task3_action(self._rng, self._hidden_goal, self._episode, action)
        self._state.step_count += 1

        current_step_index = self._episode["step_index"]
        shift_active = (
            self._hidden_goal.shift_step is not None and current_step_index >= self._hidden_goal.shift_step
        )
        self._state.budget_remaining = float(self._episode["budget_remaining"])
        self._state.capacity_remaining = float(self._episode["capacity_remaining"])

        new_metrics = self._episode["dashboard"].to_metric_vector()
        reward = compute_proxy_reward(
            previous_metrics=previous_metrics,
            new_metrics=new_metrics,
            previous_action=self._previous_action,
            current_action=action,
            invalid=step_result["invalid"],
            unused_budget_ratio=self._episode["budget_remaining"] / max(self._episode["initial_budget"], 1.0),
            previous_strategy=self._previous_strategy_signature,
            current_strategy=current_strategy_signature,
        )
        reward -= 0.05 * len(step_result["governance_flags"])
        if self._config.reward_mode == "sparse":
            reward = 0.0
        self._state.cumulative_reward += reward
        self._episode["strategy_signature_history"].append(current_strategy_signature)
        self._previous_strategy_signature = current_strategy_signature
        self._episode["decision_quality_history"].append(
            {
                "step_index": current_step_index - 1,
                "quality": round(float(decision_quality), 4),
                "stale_quality": None if stale_quality is None else round(float(stale_quality), 4),
            }
        )

        done = current_step_index >= self._episode["horizon"]
        active_goal = active_goal_name(self._hidden_goal, current_step_index)
        metadata = {
            "invalid": step_result["invalid"],
            "events": [event["name"] for event in step_result["step_events"]],
            "scheduled_effect_ids": [effect.effect_id for effect in step_result["scheduled_effects"]],
            "realized_effect_ids": [effect.effect_id for effect in step_result["realized_effects"]],
            "decision_id": step_result["decision_id"],
            "governance_flags": step_result["governance_flags"],
            "belief_quality": round(float(belief_metrics["belief_quality"]), 4),
        }
        if done:
            final_utility = compute_utility(
                self._episode["dashboard"].to_metric_vector(),
                self._hidden_goal,
                step_index=current_step_index,
            )
            if self._hidden_goal.shift_step is None:
                adaptation_score = None
                adaptation_details = {"adaptation_scored": False}
            else:
                post_shift_rows = [
                    row
                    for row in self._episode["decision_quality_history"]
                    if row["step_index"] >= self._hidden_goal.shift_step
                ]
                if not post_shift_rows:
                    adaptation_score = 0.0
                    adaptation_details = {
                        "adaptation_scored": True,
                        "behavioral_shift_detection_delay_steps": self._episode["horizon"],
                        "post_shift_recovery": 0.0,
                        "post_shift_regret": 1.0,
                    }
                else:
                    detection_delay = len(post_shift_rows)
                    for index, row in enumerate(post_shift_rows):
                        stale = row.get("stale_quality")
                        if stale is None:
                            if row["quality"] >= 0.72:
                                detection_delay = index
                                break
                        elif row["quality"] >= max(0.72, float(stale) + 0.05):
                            detection_delay = index
                            break
                    remaining_window = max(len(post_shift_rows), 1)
                    post_detection = post_shift_rows[detection_delay:] if detection_delay < len(post_shift_rows) else []
                    recovery = (
                        sum(float(row["quality"]) for row in post_detection) / len(post_detection)
                        if post_detection
                        else 0.0
                    )
                    regret = sum(1.0 - float(row["quality"]) for row in post_shift_rows) / len(post_shift_rows)
                    speed = max(0.0, 1.0 - detection_delay / remaining_window)
                    adaptation_score = 0.55 * speed + 0.45 * recovery
                    adaptation_details = {
                        "adaptation_scored": True,
                        "behavioral_shift_detection_delay_steps": int(detection_delay),
                        "post_shift_recovery": round(float(recovery), 4),
                        "post_shift_regret": round(float(regret), 4),
                    }
            belief_summary = self._belief_summary()
            coherence_score = trajectory_coherence(
                self._episode["strategy_signature_history"],
                split_index=self._hidden_goal.shift_step,
            )
            constraint_score = max(
                0.0,
                1.0 - (
                    self._episode["invalid_actions"] + 0.5 * self._episode["policy_violations"]
                ) / max(self._episode["horizon"], 1),
            )
            grade = grade_task3(
                latent_utility=final_utility,
                adaptation_score=adaptation_score,
                coherence_score=coherence_score,
                constraint_score=constraint_score,
                belief_score=belief_summary["belief_tracking"],
            )
            grade = grade.model_copy(
                update={
                    "details": {
                        **grade.details,
                        **adaptation_details,
                        "belief_ece": round(float(belief_summary["belief_ece"]), 4),
                        "belief_nll": round(float(belief_summary["belief_nll"]), 4),
                        "belief_shift_detection_delay_steps": round(float(belief_summary["shift_detection_delay_steps"]), 4),
                    }
                }
            )
            self._last_grade = grade
            self._state.last_score = grade.score
            self._state.completed = True
            if self._config.reward_mode == "sparse":
                reward = grade.score
                self._state.cumulative_reward += grade.score
            metadata["grader"] = grade.model_dump(mode="json")

        return self._build_observation(reward=reward, done=done, metadata=metadata)

    def _step_task6(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        current_step_index = self._episode["step_index"]
        belief_metrics = self._record_belief_metrics(action, current_step_index)
        current_objective = snapshot_hidden_goal(self._hidden_goal, current_step_index)
        current_weights = dict(current_objective.weights)
        _, oracle_decision_value = solve_task3_oracle_action(
            self._episode,
            current_weights,
            objective=current_objective,
        )
        agent_decision_value = evaluate_task3_action_value(
            self._episode,
            action,
            current_weights,
            objective=current_objective,
        )
        stale_quality: float | None = None
        if (
            self._hidden_goal.shift_step is not None
            and self._hidden_goal.shift_weights is not None
            and current_step_index >= self._hidden_goal.shift_step
        ):
            stale_objective = self._base_objective()
            _, stale_oracle_value = solve_task3_oracle_action(
                self._episode,
                self._hidden_goal.weights,
                objective=stale_objective,
            )
            stale_agent_value = evaluate_task3_action_value(
                self._episode,
                action,
                self._hidden_goal.weights,
                objective=stale_objective,
            )
            stale_quality = 1.0 if stale_oracle_value <= 1e-9 else max(0.0, min(1.0, stale_agent_value / stale_oracle_value))
        item_kinds_by_id = {
            item.item_id: item.kind
            for item in self._episode["backlog"]
            if item.item_id not in self._episode["completed_ids"]
        }
        current_strategy_signature = strategy_embedding(action, item_kinds_by_id=item_kinds_by_id)
        decision_quality = 1.0 if oracle_decision_value <= 1e-9 else max(0.0, min(1.0, agent_decision_value / oracle_decision_value))
        previous_metrics = self._episode["dashboard"].to_metric_vector()
        step_result = apply_task3_action(self._rng, self._hidden_goal, self._episode, action)
        self._state.step_count += 1
        current_step_index = self._episode["step_index"]
        self._state.budget_remaining = float(self._episode["budget_remaining"])
        self._state.capacity_remaining = float(self._episode["capacity_remaining"])
        new_metrics = self._episode["dashboard"].to_metric_vector()
        reward = compute_proxy_reward(
            previous_metrics=previous_metrics,
            new_metrics=new_metrics,
            previous_action=self._previous_action,
            current_action=action,
            invalid=step_result["invalid"],
            unused_budget_ratio=self._episode["budget_remaining"] / max(self._episode["initial_budget"], 1.0),
            previous_strategy=self._previous_strategy_signature,
            current_strategy=current_strategy_signature,
        )
        reward -= 0.05 * len(step_result["governance_flags"])
        if self._config.reward_mode == "sparse":
            reward = 0.0
        self._state.cumulative_reward += reward
        self._episode["strategy_signature_history"].append(current_strategy_signature)
        self._previous_strategy_signature = current_strategy_signature
        self._episode["decision_quality_history"].append(
            {
                "step_index": current_step_index - 1,
                "quality": round(float(decision_quality), 4),
                "stale_quality": None if stale_quality is None else round(float(stale_quality), 4),
            }
        )
        done = current_step_index >= self._episode["horizon"]
        metadata = {
            "invalid": step_result["invalid"],
            "events": [event["name"] for event in step_result["step_events"]],
            "scheduled_effect_ids": [effect.effect_id for effect in step_result["scheduled_effects"]],
            "realized_effect_ids": [effect.effect_id for effect in step_result["realized_effects"]],
            "decision_id": step_result["decision_id"],
            "governance_flags": step_result["governance_flags"],
            "belief_quality": round(float(belief_metrics["belief_quality"]), 4),
        }
        if done:
            final_utility = compute_utility(
                self._episode["dashboard"].to_metric_vector(),
                self._hidden_goal,
                step_index=current_step_index,
            )
            if self._hidden_goal.shift_step is None:
                adaptation_score = None
                adaptation_details = {"adaptation_scored": False}
            else:
                post_shift_rows = [
                    row
                    for row in self._episode["decision_quality_history"]
                    if row["step_index"] >= self._hidden_goal.shift_step
                ]
                if not post_shift_rows:
                    adaptation_score = 0.0
                    adaptation_details = {
                        "adaptation_scored": True,
                        "behavioral_shift_detection_delay_steps": self._episode["horizon"],
                        "post_shift_recovery": 0.0,
                        "post_shift_regret": 1.0,
                    }
                else:
                    detection_delay = len(post_shift_rows)
                    for index, row in enumerate(post_shift_rows):
                        stale = row.get("stale_quality")
                        if stale is None:
                            if row["quality"] >= 0.72:
                                detection_delay = index
                                break
                        elif row["quality"] >= max(0.72, float(stale) + 0.05):
                            detection_delay = index
                            break
                    remaining_window = max(len(post_shift_rows), 1)
                    post_detection = post_shift_rows[detection_delay:] if detection_delay < len(post_shift_rows) else []
                    recovery = (
                        sum(float(row["quality"]) for row in post_detection) / len(post_detection)
                        if post_detection
                        else 0.0
                    )
                    regret = sum(1.0 - float(row["quality"]) for row in post_shift_rows) / len(post_shift_rows)
                    speed = max(0.0, 1.0 - detection_delay / remaining_window)
                    adaptation_score = 0.55 * speed + 0.45 * recovery
                    adaptation_details = {
                        "adaptation_scored": True,
                        "behavioral_shift_detection_delay_steps": int(detection_delay),
                        "post_shift_recovery": round(float(recovery), 4),
                        "post_shift_regret": round(float(regret), 4),
                    }
            belief_summary = self._belief_summary()
            coherence_score = trajectory_coherence(
                self._episode["strategy_signature_history"],
                split_index=self._hidden_goal.shift_step,
            )
            constraint_score = max(
                0.0,
                1.0 - (self._episode["invalid_actions"] + 0.5 * self._episode["policy_violations"]) / max(self._episode["horizon"], 1),
            )
            grade = grade_task6(
                latent_utility=final_utility,
                adaptation_score=adaptation_score,
                coherence_score=coherence_score,
                constraint_score=constraint_score,
                belief_score=belief_summary["belief_tracking"],
            )
            grade = grade.model_copy(
                update={
                    "details": {
                        **grade.details,
                        **adaptation_details,
                        "belief_ece": round(float(belief_summary["belief_ece"]), 4),
                        "belief_nll": round(float(belief_summary["belief_nll"]), 4),
                        "belief_shift_detection_delay_steps": round(float(belief_summary["shift_detection_delay_steps"]), 4),
                    }
                }
            )
            self._last_grade = grade
            self._state.last_score = grade.score
            self._state.completed = True
            if self._config.reward_mode == "sparse":
                reward = grade.score
                self._state.cumulative_reward += grade.score
            metadata["grader"] = grade.model_dump(mode="json")
        return self._build_observation(reward=reward, done=done, metadata=metadata)

    def _step_task7(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        current_step_index = self._episode["step_index"]
        belief_metrics = self._record_belief_metrics(action, current_step_index)
        current_objective = snapshot_hidden_goal(self._hidden_goal, current_step_index)
        current_weights = dict(current_objective.weights)
        _, oracle_decision_value = solve_task7_oracle_action(
            self._episode,
            current_weights,
            objective=current_objective,
            step_index=current_step_index,
        )
        agent_decision_value = evaluate_task7_action_value(
            self._episode,
            action,
            current_weights,
            objective=current_objective,
        )
        stale_quality: float | None = None
        if (
            self._hidden_goal.shift_step is not None
            and self._hidden_goal.shift_weights is not None
            and current_step_index >= self._hidden_goal.shift_step
        ):
            stale_objective = self._base_objective()
            _, stale_oracle_value = solve_task7_oracle_action(
                self._episode,
                self._hidden_goal.weights,
                objective=stale_objective,
            )
            stale_agent_value = evaluate_task7_action_value(
                self._episode,
                action,
                self._hidden_goal.weights,
                objective=stale_objective,
            )
            stale_quality = 1.0 if stale_oracle_value <= 1e-9 else max(0.0, min(1.0, stale_agent_value / stale_oracle_value))
        item_kinds_by_id = {item.item_id: item.kind for item in self._episode["backlog"]}
        current_strategy_signature = strategy_embedding(action, item_kinds_by_id=item_kinds_by_id)
        decision_quality = 1.0 if oracle_decision_value <= 1e-9 else max(0.0, min(1.0, agent_decision_value / oracle_decision_value))
        previous_metrics = self._episode["dashboard"].to_metric_vector()
        step_result = apply_task7_action(self._rng, self._hidden_goal, self._episode, action)
        self._state.step_count += 1
        current_step_index = self._episode["step_index"]
        self._state.budget_remaining = float(self._episode["budget_remaining"])
        self._state.capacity_remaining = 0.0
        new_metrics = self._episode["dashboard"].to_metric_vector()
        reward = compute_proxy_reward(
            previous_metrics=previous_metrics,
            new_metrics=new_metrics,
            previous_action=self._previous_action,
            current_action=action,
            invalid=step_result["invalid"],
            unused_budget_ratio=self._episode["budget_remaining"] / max(self._episode["quarterly_budget"], 1.0),
            previous_strategy=self._previous_strategy_signature,
            current_strategy=current_strategy_signature,
        )
        if self._config.reward_mode == "sparse":
            reward = 0.0
        self._state.cumulative_reward += reward
        self._episode["strategy_signature_history"].append(current_strategy_signature)
        self._previous_strategy_signature = current_strategy_signature
        self._episode["decision_quality_history"].append(
            {
                "step_index": current_step_index - 1,
                "quality": round(float(decision_quality), 4),
                "stale_quality": None if stale_quality is None else round(float(stale_quality), 4),
            }
        )
        done = current_step_index >= self._episode["horizon"] or self._episode["total_budget_remaining"] <= 0
        metadata = {
            "invalid": step_result["invalid"],
            "scheduled_effect_ids": [effect.effect_id for effect in step_result["scheduled_effects"]],
            "realized_effect_ids": [effect.effect_id for effect in step_result["realized_effects"]],
            "decision_id": step_result["decision_id"],
            "belief_quality": round(float(belief_metrics["belief_quality"]), 4),
        }
        if done:
            final_utility = compute_utility(
                self._episode["dashboard"].to_metric_vector(),
                self._hidden_goal,
                step_index=current_step_index,
            )
            post_shift_rows = []
            if self._hidden_goal.shift_step is not None:
                post_shift_rows = [
                    row for row in self._episode["decision_quality_history"] if row["step_index"] >= self._hidden_goal.shift_step
                ]
            if not post_shift_rows:
                adaptation_score = None if self._hidden_goal.shift_step is None else 0.0
                adaptation_details = {
                    "adaptation_scored": self._hidden_goal.shift_step is not None,
                    "behavioral_shift_detection_delay_steps": 0 if self._hidden_goal.shift_step is None else self._episode["horizon"],
                    "post_shift_recovery": 1.0 if self._hidden_goal.shift_step is None else 0.0,
                    "post_shift_regret": 0.0 if self._hidden_goal.shift_step is None else 1.0,
                }
            else:
                detection_delay = len(post_shift_rows)
                for index, row in enumerate(post_shift_rows):
                    stale = row.get("stale_quality")
                    if stale is None:
                        if row["quality"] >= 0.60:
                            detection_delay = index
                            break
                    elif row["quality"] >= max(0.60, float(stale) + 0.02):
                        detection_delay = index
                        break
                remaining_window = max(len(post_shift_rows), 1)
                post_detection = post_shift_rows[detection_delay:] if detection_delay < len(post_shift_rows) else []
                recovery = (
                    sum(float(row["quality"]) for row in post_detection) / len(post_detection)
                    if post_detection
                    else 0.0
                )
                regret = sum(1.0 - float(row["quality"]) for row in post_shift_rows) / len(post_shift_rows)
                speed = max(0.0, 1.0 - detection_delay / remaining_window)
                post_shift_actions = self._episode["strategy_signature_history"][self._hidden_goal.shift_step :]
                post_shift_coherence = trajectory_coherence(post_shift_actions)
                adaptation_score = (0.40 * speed + 0.60 * recovery) * post_shift_coherence
                adaptation_details = {
                    "adaptation_scored": True,
                    "behavioral_shift_detection_delay_steps": int(detection_delay),
                    "post_shift_recovery": round(float(recovery), 4),
                    "post_shift_regret": round(float(regret), 4),
                    "post_shift_coherence": round(float(post_shift_coherence), 4),
                }
            belief_summary = self._belief_summary()
            coherence_score = trajectory_coherence(
                self._episode["strategy_signature_history"],
                split_index=self._hidden_goal.shift_step,
            )
            constraint_score = max(
                0.0,
                1.0 - self._episode.get("invalid_actions", 0) / max(self._episode["horizon"], 1),
            )
            grade = grade_task7(
                latent_utility=final_utility,
                adaptation_score=adaptation_score,
                coherence_score=coherence_score,
                constraint_score=constraint_score,
                belief_score=belief_summary["belief_tracking"],
            )
            grade = grade.model_copy(
                update={
                    "details": {
                        **grade.details,
                        **adaptation_details,
                        "belief_ece": round(float(belief_summary["belief_ece"]), 4),
                        "belief_nll": round(float(belief_summary["belief_nll"]), 4),
                        "belief_shift_detection_delay_steps": round(float(belief_summary["shift_detection_delay_steps"]), 4),
                    }
                }
            )
            self._last_grade = grade
            self._state.last_score = grade.score
            self._state.completed = True
            if self._config.reward_mode == "sparse":
                reward = grade.score
                self._state.cumulative_reward += grade.score
            metadata["grader"] = grade.model_dump(mode="json")
        return self._build_observation(reward=reward, done=done, metadata=metadata)

    def _step_task4(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        backlog = self._episode["backlog"]
        valid_ids = {item.item_id for item in backlog}
        raw_allocations = {
            item_id: float(amount)
            for item_id, amount in action.budget_allocations.items()
            if item_id in valid_ids and float(amount) > 0.0
        }
        invalid = any(float(amount) < 0.0 for amount in action.budget_allocations.values())
        allocations = {item_id: float(int(round(amount))) for item_id, amount in raw_allocations.items()}
        spent_budget = sum(allocations.values())
        budget = float(self._episode["sprint_budget"])
        if spent_budget > budget + 1e-9:
            invalid = True
            allocations = {}
            spent_budget = 0.0
        agent_value = allocation_value(
            allocations,
            backlog,
            self._hidden_goal.weights,
            self._hidden_goal,
            self._episode.get("company_profile"),
            step_index=0,
        )
        budget_use_ratio = spent_budget / max(budget, 1.0)
        grade = grade_task4(
            agent_value=agent_value,
            random_baseline=float(self._episode["random_value"]),
            oracle_value=float(self._episode["oracle_value"]),
            budget_use_ratio=budget_use_ratio,
        )
        reward = max(0.0, grade.score - (0.15 if invalid else 0.0))
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = reward
        self._state.budget_remaining = max(0.0, budget - spent_budget)
        return self._build_observation(
            reward=reward,
            done=True,
            metadata={
                "grader": grade.model_dump(mode="json"),
                "spent_budget": spent_budget,
                "invalid_budget": invalid,
                "resolved_allocations": allocations,
            },
        )

    def _step_task5(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        backlog = self._episode["backlog"]
        valid_ids = {item.item_id for item in backlog}
        chosen_ids = [item_id for item_id in action.chosen_initiatives if item_id in valid_ids]
        spent_budget = sum(item.cost for item in backlog if item.item_id in chosen_ids)
        capacity_used = len(chosen_ids)
        invalid = spent_budget > float(self._episode["budget_remaining"]) + 1e-9 or capacity_used > int(
            self._episode["capacity_remaining"]
        )
        if invalid:
            spent_budget = 0.0
            capacity_used = 0
            raw_value = 0.0
            constraint_score = 0.0
            flags = ["invalid_selection"]
        else:
            scored_action = action.model_copy(update={"chosen_initiatives": chosen_ids})
            raw_value, constraint_score, flags = evaluate_task5_action_value(
                self._episode,
                scored_action,
                self._hidden_goal.weights,
            )
        composite_value = raw_value + 0.12 * constraint_score
        grade = grade_task5(
            agent_value=composite_value,
            random_baseline=float(self._episode["random_value"]),
            oracle_value=float(self._episode["oracle_value"]),
            constraint_score=constraint_score,
        )
        reward = max(0.0, grade.score - (0.15 if invalid else 0.0))
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = reward
        self._state.budget_remaining = max(0.0, float(self._episode["budget_remaining"]) - spent_budget)
        self._state.capacity_remaining = max(0.0, float(self._episode["capacity_remaining"]) - capacity_used)
        return self._build_observation(
            reward=reward,
            done=True,
            metadata={
                "grader": grade.model_dump(mode="json"),
                "spent_budget": spent_budget,
                "capacity_used": capacity_used,
                "invalid_selection": invalid,
                "governance_flags": flags,
            },
        )

    def _available_actions(self) -> list[str]:
        if self._task_id == TaskId.TASK1:
            return [
                "Assign labels for every feedback item.",
                "Set priorities from 1-5.",
                "Escalate up to three item IDs.",
            ]
        if self._task_id == TaskId.TASK2:
            return [
                "Select a subset of roadmap item IDs within budget.",
                "Optionally provide a short rationale_summary.",
            ]
        if self._task_id == TaskId.TASK4:
            return [
                "Allocate discrete budget points across visible program IDs using budget_allocations.",
                "Respect each program's allocation_max and the overall sprint_budget.",
                "Account for diminishing returns, dependencies, conflicts, and stakeholder pressure.",
            ]
        if self._task_id == TaskId.TASK5:
            return [
                "Choose a small crisis-response package from the visible initiative IDs.",
                "Optionally adjust pricing, messaging, and support policy in the same decision.",
                "Respect visible governance constraints around pricing, SLA handling, and margin stress.",
            ]
        if self._task_id == TaskId.TASK7:
            return [
                "Allocate quarterly hiring slots across visible headcount programs using budget_allocations.",
                "Use memory to track which teams and accounts remain bottlenecks over time.",
                "Account for delayed staffing effects, execution risk, and budget carryover into future quarters.",
            ]
        return [
            "Choose initiatives for the current day.",
            "Optionally change pricing, messaging, and support policy.",
            "Optionally submit memory_focus, memory_writes, and a belief_report with the action.",
            "Reason over delayed effects, calendar events, account renewals, and prior decisions.",
            "Respect visible governance constraints around pricing, SLA handling, and margin pressure.",
            "Stay coherent across steps and adapt if the objective silently shifts.",
        ]

    def _current_public_observation_payload(self) -> dict[str, Any]:
        observation = self._build_observation(
            reward=0.0,
            done=bool(self._state.completed),
            metadata={},
        )
        return observation.model_dump(mode="json")

    @staticmethod
    def _goal_hint_weights(goal_hint: str) -> dict[str, float]:
        return {
            "growth": {"growth": 0.55, "retention": 0.20, "revenue": 0.15, "efficiency": 0.10},
            "retention": {"growth": 0.15, "retention": 0.55, "revenue": 0.15, "efficiency": 0.15},
            "revenue": {"growth": 0.10, "retention": 0.20, "revenue": 0.55, "efficiency": 0.15},
            "efficiency": {"growth": 0.10, "retention": 0.15, "revenue": 0.15, "efficiency": 0.60},
        }[goal_hint]

    @staticmethod
    def _visible_channel_proxy(item: dict[str, Any]) -> dict[str, float]:
        return visible_item_proxy(item)

    @classmethod
    def _visible_item_alignment(
        cls,
        item: dict[str, Any],
        goal_weights: dict[str, float],
        accounts_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> float:
        proxy = visible_item_proxy(item, accounts_by_id=accounts_by_id)
        alignment = sum(proxy[channel] * goal_weights[channel] for channel in CHANNELS)
        alignment += 0.004 * min(len(item.get("beneficiary_segments", [])), 3)
        alignment += 0.004 * min(len(item.get("beneficiary_account_ids", [])), 3)
        alignment -= float(item.get("implementation_risk", 0.0) or 0.0) * 0.05
        return alignment

    @classmethod
    def _visible_bundle_score(
        cls,
        selected_item_ids: list[str],
        observation: dict[str, Any],
        goal_weights: dict[str, float],
    ) -> float:
        backlog = {str(item.get("item_id")): item for item in observation.get("backlog", [])}
        accounts_by_id = {
            str(account.get("account_id")): account
            for account in observation.get("accounts", [])
        }
        chosen = {
            item_id: backlog[item_id]
            for item_id in selected_item_ids
            if item_id in backlog
        }
        if not chosen:
            return 0.0

        budget_limit = float(
            observation.get("budget_remaining")
            or observation.get("sprint_budget")
            or 0.0
        )
        spent_budget = sum(float(item.get("cost", 0.0) or 0.0) for item in chosen.values())
        if budget_limit > 0.0 and spent_budget > budget_limit + 1e-9:
            return -1.0

        score = 0.0
        utilities: dict[str, float] = {}
        synergy_bonus = 0.0
        dependency_coverage = 0
        focus_support = {channel: 0.0 for channel in CHANNELS}
        for item_id, item in chosen.items():
            proxy = visible_item_proxy(item, accounts_by_id=accounts_by_id)
            utility = cls._visible_item_alignment(item, goal_weights, accounts_by_id)
            if item.get("requires_item_ids") and not any(
                required in chosen for required in item.get("requires_item_ids", [])
            ):
                utility *= 0.60
            elif item.get("requires_item_ids"):
                dependency_coverage += 1
            if item.get("synergy_item_ids") and any(
                synergy in chosen for synergy in item.get("synergy_item_ids", [])
            ):
                utility *= 1.10
            utilities[item_id] = utility
            score += utility
            for channel, value in proxy.items():
                focus_support[channel] += max(0.0, float(value))

        for item_id, item in chosen.items():
            for synergy_id in item.get("synergy_item_ids", []):
                synergy_id = str(synergy_id)
                if synergy_id in chosen and item_id < synergy_id:
                    synergy_bonus += 0.06 * min(utilities[item_id], utilities[synergy_id])
            for conflict_id in item.get("conflicts_with_ids", []):
                conflict_id = str(conflict_id)
                if conflict_id in chosen and item_id < conflict_id:
                    score -= 0.14 * min(utilities[item_id], utilities[conflict_id])

        score -= 0.02 * max(0, len(chosen) - 4)
        spend_ratio = spent_budget / max(budget_limit, 1.0) if budget_limit > 0.0 else 1.0
        if budget_limit > 0.0:
            score += 0.03 * min(1.0, spent_budget / budget_limit)
        total_focus = sum(focus_support.values())
        if total_focus > 0.0:
            dominant_share = max(focus_support.values()) / total_focus
            if dominant_share >= 0.55:
                score += 0.02
            elif len(chosen) >= 4 and dominant_share < 0.38:
                score -= 0.02
        if len(chosen) >= 4 and spend_ratio >= 0.72:
            score += 0.05
        if dependency_coverage > 0:
            score += 0.02 * min(dependency_coverage, 2)
        if len(chosen) <= 3 and spend_ratio < 0.65:
            score -= 0.10
        if len(chosen) <= 2:
            score -= 0.04
        if spend_ratio < 0.50:
            score -= 0.04
        score += synergy_bonus
        return score

    @classmethod
    def _visible_select_task2_bundle(cls, observation: dict[str, Any], goal_weights: dict[str, float]) -> list[str]:
        backlog = list(observation.get("backlog", []))
        budget = float(observation.get("sprint_budget") or 0.0)
        best_ids: list[str] = []
        best_score = 0.0
        for subset_size in range(len(backlog) + 1):
            for subset in itertools.combinations(backlog, subset_size):
                cost = sum(float(item.get("cost", 0.0) or 0.0) for item in subset)
                if budget > 0.0 and cost > budget + 1e-9:
                    continue
                selected_ids = [str(item.get("item_id")) for item in subset]
                score = cls._visible_bundle_score(selected_ids, observation, goal_weights)
                if score > best_score:
                    best_score = score
                    best_ids = selected_ids
        return best_ids

    @classmethod
    def _visible_allocation_score(
        cls,
        allocations: dict[str, float],
        observation: dict[str, Any],
        goal_weights: dict[str, float],
    ) -> float:
        backlog = {str(item.get("item_id")): item for item in observation.get("backlog", [])}
        accounts_by_id = {
            str(account.get("account_id")): account
            for account in observation.get("accounts", [])
        }
        budget_limit = int(round(float(
            observation.get("budget_remaining")
            or observation.get("sprint_budget")
            or 0.0
        )))
        spent = int(round(sum(max(0.0, float(amount)) for amount in allocations.values())))
        if budget_limit > 0 and spent > budget_limit:
            return -1.0

        chosen: dict[str, tuple[dict[str, Any], float]] = {}
        for item_id, raw_amount in allocations.items():
            if item_id not in backlog:
                continue
            item = backlog[item_id]
            capped = max(
                0.0,
                min(
                    float(raw_amount),
                    float(item.get("allocation_max", 0.0) or 0.0),
                ),
            )
            if capped > 0.0:
                chosen[item_id] = (item, capped)
        if not chosen:
            return 0.0

        score = 0.0
        risk_penalty = 0.0
        allocated_focus: dict[str, float] = {channel: 0.0 for channel in CHANNELS}
        utilities: dict[str, float] = {}
        for item_id, (item, amount) in chosen.items():
            saturation = float(item.get("saturation_point", amount) or amount)
            effective_units = amount if amount <= saturation else saturation + 0.55 * (amount - saturation)
            proxy = visible_item_proxy(item, accounts_by_id=accounts_by_id)
            utility = cls._visible_item_alignment(item, goal_weights, accounts_by_id)
            if item.get("requires_item_ids") and not any(
                required in chosen for required in item.get("requires_item_ids", [])
            ):
                utility *= 0.60
            if item.get("synergy_item_ids") and any(
                synergy in chosen for synergy in item.get("synergy_item_ids", [])
            ):
                utility *= 1.08
            utilities[item_id] = utility * effective_units
            score += utilities[item_id]
            risk_penalty += float(item.get("implementation_risk", 0.0) or 0.0) * 0.016 * amount
            for channel, value in proxy.items():
                allocated_focus[channel] += max(0.0, float(value)) * amount

        for item_id, (item, _) in chosen.items():
            for conflict_id in item.get("conflicts_with_ids", []):
                conflict_id = str(conflict_id)
                if conflict_id in chosen and item_id < conflict_id:
                    score -= 0.10 * min(utilities[item_id], utilities[conflict_id])

        score -= risk_penalty
        score -= 0.020 * max(0, len(chosen) - 4)
        total_allocated = sum(amount for _, amount in chosen.values())
        active_focuses = sum(1 for value in allocated_focus.values() if value > 0.05)
        if active_focuses >= 3:
            score -= 0.045 * (active_focuses - 2)
        if total_allocated > 0.0 and max(allocated_focus.values(), default=0.0) > 0.0:
            dominant_share = max(allocated_focus.values()) / sum(allocated_focus.values())
            if dominant_share >= 0.55:
                score += 0.05
            elif active_focuses >= 3 and dominant_share < 0.45:
                score -= 0.06
        return score

    @classmethod
    def _visible_select_allocations(
        cls,
        observation: dict[str, Any],
        goal_weights: dict[str, float],
    ) -> dict[str, float]:
        backlog = list(observation.get("backlog", []))
        budget = int(round(float(
            observation.get("budget_remaining")
            or observation.get("sprint_budget")
            or 0.0
        )))
        best_allocations: dict[str, float] = {}
        best_score = 0.0

        def search(index: int, remaining: int, current: dict[str, float]) -> None:
            nonlocal best_allocations, best_score
            if index >= len(backlog):
                score = cls._visible_allocation_score(current, observation, goal_weights)
                if score > best_score:
                    best_score = score
                    best_allocations = {
                        item_id: float(amount)
                        for item_id, amount in current.items()
                        if amount > 0.0
                    }
                return

            item = backlog[index]
            limit = min(
                remaining,
                int(round(float(item.get("allocation_max", 0.0) or 0.0))),
            )
            for amount in range(limit, -1, -1):
                item_id = str(item.get("item_id"))
                if amount > 0:
                    current[item_id] = float(amount)
                else:
                    current.pop(item_id, None)
                search(index + 1, remaining - amount, current)
            current.pop(str(item.get("item_id")), None)

        search(0, budget, {})
        return best_allocations

    @staticmethod
    def _visible_messaging_channels(action: MessagingAction | None) -> dict[str, float]:
        return {
            MessagingAction.GROWTH_PUSH: {"growth": 0.03, "retention": -0.004},
            MessagingAction.RETENTION_CAMPAIGN: {"retention": 0.03, "growth": -0.002},
            MessagingAction.REVENUE_UPSELL: {"revenue": 0.035, "retention": -0.006},
            MessagingAction.COST_COMMS: {"efficiency": 0.03, "growth": -0.004},
            MessagingAction.NONE: {},
            None: {},
        }[action]

    @staticmethod
    def _visible_support_channels(policy: SupportPolicy | None) -> dict[str, float]:
        return {
            SupportPolicy.PREMIUM_SLA: {"retention": 0.035, "efficiency": -0.012},
            SupportPolicy.BALANCED_TRIAGE: {"retention": 0.015, "efficiency": 0.008},
            SupportPolicy.AUTOMATION_FIRST: {"efficiency": 0.035, "retention": -0.015},
            SupportPolicy.INCIDENT_SWARM: {"retention": 0.025, "efficiency": 0.008},
            None: {},
        }[policy]

    @staticmethod
    def _visible_pricing_channels(pricing_change_pct: float | None) -> dict[str, float]:
        price = float(pricing_change_pct or 0.0)
        if price > 0.0:
            scale = min(price / 0.06, 1.0)
            return {
                "growth": -0.010 * scale,
                "retention": -0.015 * scale,
                "revenue": 0.040 * scale,
                "efficiency": 0.004 * scale,
            }
        if price < 0.0:
            scale = min(abs(price) / 0.06, 1.0)
            return {
                "growth": 0.020 * scale,
                "retention": 0.008 * scale,
                "revenue": -0.026 * scale,
                "efficiency": -0.004 * scale,
            }
        return {}

    @classmethod
    def _visible_mixed_action_score(
        cls,
        observation: dict[str, Any],
        selected_item_ids: list[str],
        messaging_action: MessagingAction | None,
        support_policy: SupportPolicy | None,
        pricing_change_pct: float,
        goal_weights: dict[str, float],
        task_id: TaskId,
    ) -> float:
        backlog = {str(item.get("item_id")): item for item in observation.get("backlog", [])}
        chosen_items = [
            backlog[item_id]
            for item_id in selected_item_ids
            if item_id in backlog
        ]
        accounts_by_id = {
            str(account.get("account_id")): account
            for account in observation.get("accounts", [])
        }
        spent_budget = sum(float(item.get("cost", 0.0) or 0.0) for item in chosen_items)
        budget_limit = float(observation.get("budget_remaining") or 0.0)
        if budget_limit > 0.0 and spent_budget > budget_limit + 1e-9:
            return -1.0

        if task_id == TaskId.TASK5:
            capacity_limit = int(round(float(observation.get("capacity_remaining") or 0.0)))
            if len(chosen_items) > capacity_limit:
                return -1.0
        else:
            capacity_used = sum(max(1.0, float(item.get("cost", 0.0) or 0.0) / 2.0) for item in chosen_items)
            capacity_limit = float(observation.get("capacity_remaining") or 0.0)
            if capacity_limit > 0.0 and capacity_used > capacity_limit + 1e-9:
                return -1.0

        score = cls._visible_bundle_score(selected_item_ids, observation, goal_weights)
        for control_proxy in (
            cls._visible_messaging_channels(messaging_action),
            cls._visible_support_channels(support_policy),
            cls._visible_pricing_channels(pricing_change_pct),
        ):
            score += sum(float(control_proxy.get(channel, 0.0)) * goal_weights[channel] for channel in CHANNELS)

        high_touch_accounts = [
            account
            for account in observation.get("accounts", [])
            if str(account.get("support_tier", "")) == "premium"
            or (
                str(account.get("segment", "")) in {"enterprise", "strategic"}
                and int(account.get("renewal_window_days", 999) or 999) <= 30
            )
        ]
        visible_constraint_ids = {
            str(constraint.get("constraint_id"))
            for constraint in observation.get("governance_constraints", [])
        }
        if high_touch_accounts and support_policy == SupportPolicy.AUTOMATION_FIRST and "sla_guardrail" in visible_constraint_ids:
            score -= 0.08
        if high_touch_accounts and pricing_change_pct > 0.02 and "pricing_guardrail" in visible_constraint_ids:
            score -= 0.09
        item_focus_support = {channel: 0.0 for channel in CHANNELS}
        for item in chosen_items:
            proxy = visible_item_proxy(item, accounts_by_id=accounts_by_id)
            for channel, value in proxy.items():
                item_focus_support[channel] += max(0.0, float(value))

        if (
            float(observation.get("dashboard", {}).get("ops_margin", 0.0) or 0.0) <= 0.32
            and support_policy == SupportPolicy.PREMIUM_SLA
            and item_focus_support["growth"] >= item_focus_support["efficiency"]
            and "margin_guardrail" in visible_constraint_ids
        ):
            score -= 0.05

        if high_touch_accounts and (
            support_policy in {SupportPolicy.PREMIUM_SLA, SupportPolicy.INCIDENT_SWARM}
            or item_focus_support["retention"] >= 0.05
        ):
            score += 0.02
        if float(observation.get("market_context", {}).get("board_pressure_level", 0.0) or 0.0) >= 0.68 and (
            messaging_action == MessagingAction.REVENUE_UPSELL
            or item_focus_support["revenue"] >= 0.05
        ):
            score += 0.015

        channel_support: dict[str, float] = {}
        for item in chosen_items:
            proxy = visible_item_proxy(item, accounts_by_id=accounts_by_id)
            focus = dominant_visible_focus(proxy)
            if focus is not None:
                channel_support[focus] = channel_support.get(focus, 0.0) + 1.0
        message_channel = {
            MessagingAction.GROWTH_PUSH: "growth",
            MessagingAction.RETENTION_CAMPAIGN: "retention",
            MessagingAction.REVENUE_UPSELL: "revenue",
            MessagingAction.COST_COMMS: "efficiency",
        }.get(messaging_action)
        if message_channel is not None:
            channel_support[message_channel] = channel_support.get(message_channel, 0.0) + 0.5
        if support_policy == SupportPolicy.PREMIUM_SLA:
            channel_support["retention"] = channel_support.get("retention", 0.0) + 0.4
        elif support_policy == SupportPolicy.AUTOMATION_FIRST:
            channel_support["efficiency"] = channel_support.get("efficiency", 0.0) + 0.4
        if channel_support:
            dominant_share = max(channel_support.values()) / sum(channel_support.values())
            score += 0.02 if dominant_share >= 0.5 else -0.02

        if task_id != TaskId.TASK5 and not chosen_items and messaging_action == MessagingAction.NONE:
            score -= 0.01
        return score

    @classmethod
    def _visible_response_action(cls, observation: dict[str, Any], task_id: TaskId) -> LatentGoalOpsAction:
        evidence = " ".join(
            [str(observation.get("narrative", ""))]
            + [str(message.get("text", "")) for message in observation.get("inbox", [])]
            + [str(note) for note in observation.get("stakeholder_notes", [])]
            + [str(alert) for alert in observation.get("alerts", [])]
        )
        goal_hint = cls._infer_goal_hint_from_evidence(evidence)
        goal_weights = cls._goal_hint_weights(goal_hint)
        backlog = list(observation.get("backlog", []))

        if task_id == TaskId.TASK5:
            max_subset_size = min(
                len(backlog),
                int(round(float(observation.get("capacity_remaining") or 0.0))),
            )
            pricing_grid = [-0.06, -0.03, 0.0, 0.03, 0.06]
        else:
            max_subset_size = min(3, len(backlog))
            pricing_grid = [-0.05, 0.0, 0.05]

        best_action = LatentGoalOpsAction(
            task_id=task_id,
            chosen_initiatives=[],
            messaging_action=MessagingAction.NONE,
            pricing_change_pct=0.0,
            support_policy=SupportPolicy.BALANCED_TRIAGE,
            rationale=(
                "Visible-only heuristic crisis-response package."
                if task_id == TaskId.TASK5
                else "Visible-only heuristic policy based on public signals."
            ),
        )
        best_score = cls._visible_mixed_action_score(
            observation,
            [],
            MessagingAction.NONE,
            SupportPolicy.BALANCED_TRIAGE,
            0.0,
            goal_weights,
            task_id,
        )

        for subset_size in range(max_subset_size + 1):
            for subset in itertools.combinations(backlog, subset_size):
                selected_ids = [str(item.get("item_id")) for item in subset]
                for messaging_action in MessagingAction:
                    for support_policy in SupportPolicy:
                        for pricing_change_pct in pricing_grid:
                            score = cls._visible_mixed_action_score(
                                observation,
                                selected_ids,
                                messaging_action,
                                support_policy,
                                float(pricing_change_pct),
                                goal_weights,
                                task_id,
                            )
                            if score > best_score:
                                best_score = score
                                best_action = LatentGoalOpsAction(
                                    task_id=task_id,
                                    chosen_initiatives=selected_ids,
                                    messaging_action=messaging_action,
                                    pricing_change_pct=float(pricing_change_pct),
                                    support_policy=support_policy,
                                    rationale=(
                                        "Visible-only heuristic crisis-response package."
                                        if task_id == TaskId.TASK5
                                        else "Visible-only heuristic policy based on public signals."
                                    ),
                                )
        return best_action

    def _build_observation(
        self,
        reward: float,
        done: bool,
        metadata: dict | None = None,
    ) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._task_id is not None
        if self._task_id == TaskId.TASK1:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                dashboard=self._episode["dashboard"],
                inbox=self._episode["inbox"],
                backlog=[],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                company_profile=self._episode.get("company_profile"),
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=[],
                budget_remaining=0.0,
                capacity_remaining=0.0,
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )
        if self._task_id == TaskId.TASK2:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                dashboard=self._episode["dashboard"],
                inbox=[],
                backlog=[self._public_initiative_view(item) for item in self._episode["backlog"]],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                company_profile=self._episode.get("company_profile"),
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=[],
                budget_remaining=self._state.budget_remaining,
                capacity_remaining=0.0,
                sprint_budget=self._episode["sprint_budget"],
                stakeholder_notes=self._episode["stakeholder_notes"],
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )
        if self._task_id == TaskId.TASK4:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                dashboard=self._episode["dashboard"],
                inbox=[],
                backlog=[self._public_initiative_view(item) for item in self._episode["backlog"]],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                company_profile=self._episode.get("company_profile"),
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=[],
                budget_remaining=self._state.budget_remaining,
                capacity_remaining=0.0,
                sprint_budget=self._episode["sprint_budget"],
                stakeholder_notes=self._episode["stakeholder_notes"],
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )
        if self._task_id == TaskId.TASK5:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                narrative=self._episode["narrative"],
                dashboard=self._episode["dashboard"],
                inbox=self._episode["inbox"],
                backlog=[self._public_initiative_view(item) for item in self._episode["backlog"]],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                company_profile=self._episode.get("company_profile"),
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=self._episode["alerts"],
                budget_remaining=self._state.budget_remaining,
                capacity_remaining=self._state.capacity_remaining,
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )

        view_rng = random.Random(self._seed + self._episode["step_index"] * 9973)
        if self._task_id == TaskId.TASK6:
            rollout_view = build_task6_view(view_rng, self._hidden_goal, self._episode)
        elif self._task_id == TaskId.TASK7:
            rollout_view = build_task7_view(self._episode)
        else:
            rollout_view = build_task3_view(view_rng, self._hidden_goal, self._episode)
        if not self._config.expose_decision_ledger:
            rollout_view["decision_ledger"] = []
            rollout_view["pending_effects"] = []
            rollout_view["realized_effects"] = []
            rollout_view["memory_summary"] = None
            rollout_view["memory_workspace"] = None
        self._state.sim_date = rollout_view["sim_date"]
        self._state.sim_day_label = rollout_view["sim_day_label"]
        self._state.decision_count = len(rollout_view["decision_ledger"])
        self._state.pending_effect_count = len(rollout_view["pending_effects"])
        return LatentGoalOpsObservation(
            task_id=self._task_id,
            step_index=rollout_view["step_index"],
            horizon=rollout_view["horizon"],
            sim_date=rollout_view["sim_date"],
            sim_day_label=rollout_view["sim_day_label"],
            task_summary=rollout_view["task_summary"],
            narrative=rollout_view.get("narrative"),
            dashboard=rollout_view["dashboard"],
            inbox=rollout_view.get("inbox", []),
            backlog=[self._public_initiative_view(item) for item in rollout_view["backlog"]],
            accounts=rollout_view["accounts"],
            stakeholders=rollout_view["stakeholders"],
            teams=rollout_view["teams"],
            company_profile=rollout_view.get("company_profile"),
            market_context=rollout_view["market_context"],
            governance_constraints=rollout_view["governance_constraints"],
            alerts=rollout_view["alerts"],
            calendar_events=rollout_view.get("calendar_events", []),
            decision_ledger=[self._public_decision_view(entry) for entry in rollout_view["decision_ledger"]],
            pending_effects=[self._public_effect_view(effect) for effect in rollout_view["pending_effects"]],
            realized_effects=[self._public_effect_view(effect) for effect in rollout_view["realized_effects"]],
            budget_remaining=rollout_view["budget_remaining"],
            capacity_remaining=rollout_view.get("capacity_remaining", 0.0),
            sprint_budget=rollout_view.get("sprint_budget"),
            stakeholder_notes=rollout_view.get("stakeholder_notes", []),
            memory_summary=rollout_view.get("memory_summary"),
            memory_workspace=rollout_view.get("memory_workspace"),
            memory_budget_remaining=rollout_view.get("memory_budget_remaining", 0),
            memory_write_budget_remaining=rollout_view.get("memory_write_budget_remaining", 0),
            available_actions=self._available_actions(),
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    def sample_random_action(self) -> LatentGoalOpsAction:
        """Sample a deterministic random action for the active state."""
        if self._episode is None or self._task_id is None:
            raise RuntimeError("Reset before sampling actions.")
        rng = random.Random(self._seed + 17 * (self._state.step_count + 1))
        if self._task_id == TaskId.TASK1:
            labels = []
            priorities = []
            for item in self._episode["inbox"]:
                labels.append(ItemLabelAssignment(item_id=item.item_id, label=rng.choice(list(FeedbackLabel))))
                priorities.append(ItemPriorityAssignment(item_id=item.item_id, priority=rng.randint(1, 5)))
            escalate_count = min(3, len(self._episode["inbox"]))
            escalate_ids = [item.item_id for item in rng.sample(self._episode["inbox"], k=escalate_count)]
            return LatentGoalOpsAction(task_id=self._task_id, labels=labels, priorities=priorities, escalate_ids=escalate_ids)
        if self._task_id == TaskId.TASK2:
            shuffled = self._episode["backlog"][:]
            rng.shuffle(shuffled)
            remaining = self._episode["sprint_budget"]
            selected_ids = []
            for item in shuffled:
                if item.cost <= remaining and rng.random() > 0.5:
                    selected_ids.append(item.item_id)
                    remaining -= item.cost
            return LatentGoalOpsAction(task_id=self._task_id, selected_item_ids=selected_ids)
        if self._task_id == TaskId.TASK4:
            remaining = int(round(float(self._episode["sprint_budget"])))
            allocations: dict[str, float] = {}
            shuffled = self._episode["backlog"][:]
            rng.shuffle(shuffled)
            for item in shuffled:
                if remaining <= 0:
                    break
                max_amount = min(remaining, int(round(float(item.allocation_max or 0.0))))
                amount = rng.randint(0, max_amount)
                if amount > 0:
                    allocations[item.item_id] = float(amount)
                    remaining -= amount
            return LatentGoalOpsAction(task_id=self._task_id, budget_allocations=allocations)
        if self._task_id == TaskId.TASK7:
            remaining = int(round(float(self._episode["budget_remaining"])))
            allocations: dict[str, float] = {}
            shuffled = self._episode["backlog"][:]
            rng.shuffle(shuffled)
            for item in shuffled:
                if remaining <= 0:
                    break
                max_amount = min(remaining, int(round(float(item.allocation_max or 0.0))))
                amount = rng.randint(0, max_amount)
                if amount > 0:
                    allocations[item.item_id] = float(amount)
                    remaining -= amount
            return LatentGoalOpsAction(task_id=self._task_id, budget_allocations=allocations)
        if self._task_id == TaskId.TASK5:
            visible = self._episode["backlog"][:]
            rng.shuffle(visible)
            chosen: list[str] = []
            remaining_budget = float(self._episode["budget_remaining"])
            for item in visible:
                if len(chosen) >= int(self._episode["capacity_remaining"]):
                    break
                if item.cost > remaining_budget:
                    continue
                if rng.random() > 0.5:
                    chosen.append(item.item_id)
                    remaining_budget -= item.cost
            return LatentGoalOpsAction(
                task_id=self._task_id,
                chosen_initiatives=chosen,
                messaging_action=rng.choice(list(MessagingAction)),
                pricing_change_pct=rng.choice([-0.06, -0.03, 0.0, 0.03, 0.06]),
                support_policy=rng.choice(list(SupportPolicy)),
            )
        visible = [item for item in self._episode["backlog"] if item.item_id not in self._episode["completed_ids"]]
        rng.shuffle(visible)
        chosen = [item.item_id for item in visible[: rng.randint(0, min(2, len(visible)))]]
        return LatentGoalOpsAction(
            task_id=self._task_id,
            chosen_initiatives=chosen,
            messaging_action=rng.choice(list(MessagingAction)),
            pricing_change_pct=round(rng.uniform(-0.08, 0.08), 3),
            support_policy=rng.choice(list(SupportPolicy)),
        )

    def sample_heuristic_action(self) -> LatentGoalOpsAction:
        """Generate a simple heuristic baseline action from visible context only."""
        if self._episode is None or self._task_id is None:
            raise RuntimeError("Reset before sampling actions.")
        if self._task_id == TaskId.TASK1:
            keyword_map = {
                "charged": FeedbackLabel.BILLING_ISSUE,
                "invoice": FeedbackLabel.BILLING_ISSUE,
                "cancel": FeedbackLabel.CHURN_RISK,
                "renew": FeedbackLabel.CHURN_RISK,
                "slow": FeedbackLabel.LATENCY_COMPLAINT,
                "latency": FeedbackLabel.LATENCY_COMPLAINT,
                "crash": FeedbackLabel.BUG,
                "blank screen": FeedbackLabel.BUG,
                "please add": FeedbackLabel.FEATURE_REQUEST,
                "want": FeedbackLabel.FEATURE_REQUEST,
            }
            labels = []
            priorities = []
            scored = []
            for item in self._episode["inbox"]:
                lowered = item.text.lower()
                predicted = FeedbackLabel.PRAISE
                for keyword, label in keyword_map.items():
                    if keyword in lowered:
                        predicted = label
                        break
                severity = int(item.metadata.get("severity", 3))
                tier = str(item.metadata.get("user_tier", "pro"))
                priority = min(5, max(1, severity + (1 if tier == "enterprise" else 0)))
                labels.append(ItemLabelAssignment(item_id=item.item_id, label=predicted))
                priorities.append(ItemPriorityAssignment(item_id=item.item_id, priority=priority))
                scored.append((item.item_id, priority))
            scored.sort(key=lambda row: row[1], reverse=True)
            return LatentGoalOpsAction(
                task_id=self._task_id,
                labels=labels,
                priorities=priorities,
                escalate_ids=[item_id for item_id, _ in scored[:3]],
            )
        if self._task_id == TaskId.TASK2:
            observation = self._current_public_observation_payload()
            evidence = " ".join([str(note) for note in observation.get("stakeholder_notes", [])])
            goal_hint = self._infer_goal_hint_from_evidence(evidence)
            selected_ids = self._visible_select_task2_bundle(observation, self._goal_hint_weights(goal_hint))
            return LatentGoalOpsAction(
                task_id=self._task_id,
                selected_item_ids=selected_ids,
                rationale_summary="Visible-only heuristic roadmap plan.",
            )
        if self._task_id == TaskId.TASK4:
            observation = self._current_public_observation_payload()
            evidence = " ".join([str(note) for note in observation.get("stakeholder_notes", [])])
            goal_hint = self._infer_goal_hint_from_evidence(evidence)
            allocations = self._visible_select_allocations(observation, self._goal_hint_weights(goal_hint))
            return LatentGoalOpsAction(
                task_id=self._task_id,
                budget_allocations=allocations,
                rationale_summary="Visible-only heuristic capital allocation.",
            )
        if self._task_id == TaskId.TASK5:
            observation = self._current_public_observation_payload()
            return self._visible_response_action(observation, self._task_id)
        if self._task_id == TaskId.TASK7:
            observation = self._current_public_observation_payload()
            evidence = " ".join(
                [str(observation.get("narrative", ""))]
                + [str(note) for note in observation.get("stakeholder_notes", [])]
                + [str(alert) for alert in observation.get("alerts", [])]
            )
            goal_hint = self._infer_goal_hint_from_evidence(evidence)
            allocations = self._visible_select_allocations(observation, self._goal_hint_weights(goal_hint))
            return LatentGoalOpsAction(
                task_id=self._task_id,
                budget_allocations=allocations,
                rationale_summary="Visible-only heuristic quarterly hiring plan.",
            )

        observation = self._current_public_observation_payload()
        return self._visible_response_action(observation, self._task_id)

    def sample_oracle_action(self) -> LatentGoalOpsAction:
        """Construct an oracle action using hidden episode state."""
        if self._episode is None or self._task_id is None or self._hidden_goal is None:
            raise RuntimeError("Reset before sampling actions.")
        if self._task_id == TaskId.TASK1:
            return LatentGoalOpsAction(
                task_id=self._task_id,
                labels=[
                    ItemLabelAssignment(item_id=item_id, label=label)
                    for item_id, label in self._episode["true_labels"].items()
                ],
                priorities=[
                    ItemPriorityAssignment(item_id=item_id, priority=priority)
                    for item_id, priority in self._episode["true_priorities"].items()
                ],
                escalate_ids=self._episode["oracle_escalations"],
            )
        if self._task_id == TaskId.TASK2:
            return LatentGoalOpsAction(
                task_id=self._task_id,
                selected_item_ids=self._episode["oracle_selection"],
                rationale_summary="Oracle roadmap plan.",
            )
        if self._task_id == TaskId.TASK4:
            return LatentGoalOpsAction(
                task_id=self._task_id,
                budget_allocations=self._episode["oracle_allocations"],
                rationale_summary="Oracle capital allocation.",
            )
        if self._task_id == TaskId.TASK7:
            step_index = self._episode["step_index"]
            current_objective = snapshot_hidden_goal(self._hidden_goal, step_index)
            oracle_action, _ = solve_task7_oracle_action(
                self._episode,
                current_objective.weights,
                objective=current_objective,
                step_index=step_index,
            )
            return oracle_action.model_copy(
                update={
                    "rationale_summary": "Oracle quarterly headcount plan.",
                    "belief_report": self._oracle_belief_report(step_index),
                }
            )
        if self._task_id == TaskId.TASK5:
            return self._episode["oracle_action"].model_copy(
                update={"rationale": "Oracle crisis-response package."}
            )
        if self._task_id == TaskId.TASK6:
            step_index = self._episode["step_index"]
            active_goal = active_goal_name(self._hidden_goal, step_index)
            current_objective = snapshot_hidden_goal(self._hidden_goal, step_index)
            oracle_action, _ = solve_task3_oracle_action(
                self._episode,
                current_objective.weights,
                objective=current_objective,
            )
            return oracle_action.model_copy(
                update={
                    "task_id": self._task_id,
                    "rationale": f"Oracle incident-response action using latent goal knowledge for {active_goal}.",
                    "belief_report": self._oracle_belief_report(step_index),
                }
            )

        step_index = self._episode["step_index"]
        active_goal = active_goal_name(self._hidden_goal, step_index)
        current_objective = snapshot_hidden_goal(self._hidden_goal, step_index)
        weights = dict(current_objective.weights)
        planning_weights = dict(weights)
        if (
            self._hidden_goal.shift_step is not None
            and self._hidden_goal.shift_weights is not None
            and step_index < self._hidden_goal.shift_step
        ):
            current_window = max(self._hidden_goal.shift_step - step_index, 1)
            future_window = max(self._episode["horizon"] - self._hidden_goal.shift_step, 1)
            total_window = current_window + future_window
            planning_weights = {
                channel: (
                    float(weights[channel]) * current_window
                    + float(self._hidden_goal.shift_weights[channel]) * future_window
                )
                / total_window
                for channel in weights
            }
        planning_objective = snapshot_hidden_goal(self._hidden_goal, step_index)
        planning_objective.weights = planning_weights
        oracle_action, _ = solve_task3_oracle_action(
            self._episode,
            planning_weights,
            objective=planning_objective,
        )
        return oracle_action.model_copy(
            update={
                "rationale": f"Oracle action using latent goal knowledge for {active_goal}.",
                "belief_report": self._oracle_belief_report(step_index),
            }
        )

    @property
    def state(self) -> LatentGoalOpsState:
        """Return current public environment state."""
        return self._state

    @property
    def last_grade(self) -> GraderResult | None:
        """Return last terminal grade."""
        return self._last_grade
