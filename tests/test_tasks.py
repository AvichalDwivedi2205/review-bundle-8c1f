"""Core task smoke tests."""

import re

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.models import LatentGoalOpsAction, MessagingAction, SupportPolicy, TaskId
from latentgoalops.server.environment import LatentGoalOpsEnvironment


def test_task1_oracle_scores_perfectly():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=7, task_id="task1_feedback_triage")
    assert observation.accounts
    assert observation.stakeholders
    assert observation.market_context is not None
    assert observation.governance_constraints
    assert "annual_contract_value" in observation.inbox[0].metadata
    observation = env.step(env.sample_oracle_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert env.last_grade.score == 1.0


def test_task2_oracle_scores_perfectly():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=7, task_id="task2_roadmap_priority")
    assert observation.accounts
    assert observation.stakeholders
    assert observation.market_context is not None
    assert observation.governance_constraints
    assert observation.backlog[0].beneficiary_segments
    assert observation.backlog[0].policy_tags is not None
    observation = env.step(env.sample_oracle_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert env.last_grade.score == 1.0


def test_task2_notes_do_not_name_hidden_goal():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=9, task_id="task2_roadmap_priority")
    hidden_goal = env._hidden_goal
    assert hidden_goal is not None
    combined_notes = " ".join(observation.stakeholder_notes).lower()
    assert hidden_goal.archetype.value not in combined_notes


def test_task2_backlog_exposes_bundle_structure():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=10, task_id="task2_roadmap_priority")
    assert any(
        item.requires_item_ids or item.conflicts_with_ids or item.synergy_item_ids or item.risk_notes
        for item in observation.backlog
    )
    family_to_items: dict[str, list[str]] = {}
    for item in observation.backlog:
        family = item.item_id.rsplit("_", 1)[0]
        family_to_items.setdefault(family, []).append(item.item_id)
    assert any(
        all(
            other in next(backlog_item for backlog_item in observation.backlog if backlog_item.item_id == item_id).conflicts_with_ids
            for other in item_ids
            if other != item_id
        )
        for item_id, item_ids in (
            (item_id, item_ids)
            for item_ids in family_to_items.values()
            if len(item_ids) > 1
            for item_id in item_ids
        )
    )


def test_task2_public_backlog_redacts_exact_kpi_deltas():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=10, task_id="task2_roadmap_priority")
    assert observation.backlog
    assert all("kpi_deltas" not in item.model_dump(mode="json") for item in observation.backlog)
    assert all("kind" not in item.model_dump(mode="json") for item in observation.backlog)
    assert all(item.impact_summary for item in observation.backlog)


def test_task2_summaries_surface_visible_portfolio_paths():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=10, task_id="task2_roadmap_priority")
    summaries = [item.impact_summary for item in observation.backlog]
    assert any(
        summary and (
            "Works best after" in summary
            or "Pairs naturally with" in summary
            or "Creates visible overlap or sequencing tension" in summary
        )
        for summary in summaries
    )


def test_task2_visible_accounts_cover_backlog_and_notes():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=10, task_id="task2_roadmap_priority")
    visible_account_ids = {account.account_id for account in observation.accounts}
    visible_account_names = {account.company_name for account in observation.accounts}
    linked_account_ids = {
        account_id
        for item in observation.backlog
        for account_id in item.beneficiary_account_ids
    }
    mentioned_account_names = {
        name
        for note in observation.stakeholder_notes
        for name in re.findall(r"[A-Z][A-Za-z]+ Analytics Customer \\d+", note)
    }
    assert linked_account_ids.issubset(visible_account_ids)
    assert mentioned_account_names.issubset(visible_account_names)


def test_task2_heldout_split_uses_alt_initiative_family():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(scenario_split="heldout"))
    observation = env.reset(seed=14, task_id="task2_roadmap_priority")
    visible_ids = {item.item_id.rsplit("_", 1)[0] for item in observation.backlog}
    assert visible_ids
    assert any(
        item_id in {
            "partner_marketplace_seed",
            "developer_conversion_grants",
            "security_assurance_pool",
            "enterprise_procurement_desk",
        }
        for item_id in visible_ids
    )


def test_task3_runs_to_completion():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=7, task_id="task3_startup_week")
    assert observation.sim_date is not None
    assert observation.sim_day_label == "Day 1"
    assert isinstance(observation.calendar_events, list)
    assert isinstance(observation.pending_effects, list)
    assert isinstance(observation.decision_ledger, list)
    while not observation.done:
        observation = env.step(env.sample_heuristic_action())
    assert env.last_grade is not None
    assert 0.0 <= env.last_grade.score <= 1.0
    assert env.state.step_count == 10


def test_task3_temporal_ledger_populates_after_step():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=11, task_id="task3_startup_week")
    observation = env.step(env.sample_heuristic_action())
    assert observation.sim_date is not None
    assert observation.decision_ledger
    assert observation.decision_ledger[-1].sim_date is not None


def test_task3_public_metadata_does_not_expose_latent_goal_state():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=11, task_id="task3_startup_week")
    observation = env.step(env.sample_random_action())
    assert "active_goal" not in observation.metadata
    assert "decision_quality" not in observation.metadata
    assert "shift_active" not in observation.metadata
    assert "shift_active" not in env.state.model_dump(mode="json")


def test_task3_visible_text_does_not_name_hidden_goal():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=12, task_id="task3_startup_week")
    hidden_goal = env._hidden_goal
    assert hidden_goal is not None
    visible_text = " ".join([observation.narrative or ""] + [item.text for item in observation.inbox]).lower()
    assert hidden_goal.archetype.value not in visible_text


def test_task3_accounts_and_constraints_are_visible():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=15, task_id="task3_startup_week")
    assert observation.company_profile is not None
    assert observation.accounts
    assert observation.stakeholders
    assert observation.teams
    assert observation.market_context is not None
    assert observation.governance_constraints
    assert observation.memory_workspace is not None
    assert observation.memory_workspace.records


def test_task3_backlog_exposes_bundle_structure():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=16, task_id="task3_startup_week")
    assert any(
        item.requires_item_ids or item.conflicts_with_ids or item.synergy_item_ids or item.risk_notes
        for item in observation.backlog
    )


def test_task3_account_renewal_windows_progress_over_time():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=17, task_id="task3_startup_week")
    before = {account.account_id: account.renewal_window_days for account in observation.accounts}
    observation = env.step(env.sample_heuristic_action())
    after = {account.account_id: account.renewal_window_days for account in observation.accounts}
    shared_ids = set(before) & set(after)
    assert shared_ids
    assert any(after[account_id] <= before[account_id] - 1 for account_id in shared_ids)


def test_task3_ablation_can_hide_ledger():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(expose_decision_ledger=False))
    observation = env.reset(seed=12, task_id="task3_startup_week")
    observation = env.step(env.sample_heuristic_action())
    assert observation.decision_ledger == []
    assert observation.pending_effects == []
    assert observation.realized_effects == []
    assert observation.memory_summary is None
    assert "delayed effect" not in (observation.narrative or "").lower()
    assert all(item.sender != "analytics" for item in observation.inbox)


def test_task3_public_effects_and_ledger_redact_internal_deltas():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=11, task_id="task3_startup_week")
    observation = env.step(env.sample_random_action())
    assert all("channel_deltas" not in effect.model_dump(mode="json") for effect in observation.pending_effects)
    assert all("dashboard_deltas" not in effect.model_dump(mode="json") for effect in observation.pending_effects)
    assert all("channel_deltas" not in effect.model_dump(mode="json") for effect in observation.realized_effects)
    assert all("dashboard_deltas" not in effect.model_dump(mode="json") for effect in observation.realized_effects)
    assert all("expected_channels" not in entry.model_dump(mode="json") for entry in observation.decision_ledger)


def test_task3_sparse_reward_only_pays_on_terminal_step():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(reward_mode="sparse"))
    observation = env.reset(seed=13, task_id="task3_startup_week")
    while not observation.done:
        observation = env.step(env.sample_heuristic_action())
        if not observation.done:
            assert observation.reward == 0.0
    assert observation.reward is not None
    assert observation.reward >= 0.0


def test_task4_oracle_scores_perfectly():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=19, task_id="task4_capital_allocation")
    assert observation.backlog
    assert observation.sprint_budget is not None
    assert observation.stakeholder_notes
    observation = env.step(env.sample_oracle_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert env.last_grade.score == 1.0


def test_task4_backlog_exposes_allocation_fields():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=20, task_id="task4_capital_allocation")
    assert all(item.allocation_max is not None and item.saturation_point is not None for item in observation.backlog)


def test_task5_runs_to_completion():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=21, task_id="task5_crisis_response")
    assert observation.narrative is not None
    assert observation.inbox
    observation = env.step(env.sample_heuristic_action())
    assert observation.done is True
    assert env.last_grade is not None
    assert 0.0 <= env.last_grade.score <= 1.0


def test_task5_visible_text_does_not_name_hidden_goal():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=22, task_id="task5_crisis_response")
    hidden_goal = env._hidden_goal
    assert hidden_goal is not None
    visible_text = " ".join([observation.narrative or ""] + [item.text for item in observation.inbox]).lower()
    assert hidden_goal.archetype.value not in visible_text


def test_task5_invalid_package_scores_zero():
    env = LatentGoalOpsEnvironment()
    env.reset(seed=21, task_id="task5_crisis_response")
    all_ids = [item.item_id for item in env._episode["backlog"]]  # noqa: SLF001
    observation = env.step(
        LatentGoalOpsAction(
            task_id=TaskId.TASK5,
            chosen_initiatives=all_ids,
            messaging_action=MessagingAction.RETENTION_CAMPAIGN,
            pricing_change_pct=0.0,
            support_policy=SupportPolicy.PREMIUM_SLA,
        )
    )
    assert observation.metadata["invalid_selection"] is True
    assert env.last_grade is not None
    assert env.last_grade.score == 0.0
    assert observation.reward == 0.0


def test_task2_visible_only_heuristic_operates_on_redacted_payload():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=10, task_id="task2_roadmap_priority")
    payload = observation.model_dump(mode="json")
    assert payload["backlog"]
    assert all("kpi_deltas" not in item and "kind" not in item for item in payload["backlog"])
    goal_hint = env._infer_goal_hint_from_evidence(" ".join(observation.stakeholder_notes))  # noqa: SLF001
    selected_ids = env._visible_select_task2_bundle(payload, env._goal_hint_weights(goal_hint))  # noqa: SLF001
    assert isinstance(selected_ids, list)
    assert all(isinstance(item_id, str) for item_id in selected_ids)


def test_task6_runs_to_completion_and_exposes_memory():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=24, task_id="task6_incident_response_week")
    assert observation.company_profile is not None
    assert observation.memory_workspace is not None
    assert observation.memory_workspace.records
    while not observation.done:
        observation = env.step(env.sample_heuristic_action())
    assert env.last_grade is not None
    assert 0.0 <= env.last_grade.score <= 1.0


def test_task6_oracle_action_keeps_task_id():
    env = LatentGoalOpsEnvironment()
    env.reset(seed=24, task_id="task6_incident_response_week")
    action = env.sample_oracle_action()
    assert action.task_id == TaskId.TASK6
    observation = env.step(action)
    assert observation.task_id == TaskId.TASK6


def test_task7_runs_to_completion_and_tracks_quarters():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=25, task_id="task7_quarterly_headcount_plan")
    assert observation.company_profile is not None
    assert observation.sim_day_label == "Quarter 1"
    assert observation.memory_workspace is not None
    while not observation.done:
        observation = env.step(env.sample_heuristic_action())
    assert env.last_grade is not None
    assert 0.0 <= env.last_grade.score <= 1.0
    assert env.state.step_count == 4


def test_task7_backlog_exposes_bundle_structure():
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=26, task_id="task7_quarterly_headcount_plan")
    assert any(
        item.requires_item_ids or item.conflicts_with_ids or item.synergy_item_ids or item.risk_notes
        for item in observation.backlog
    )


def test_task7_oracle_action_includes_perfect_belief_report():
    env = LatentGoalOpsEnvironment()
    env.reset(seed=26, task_id="task7_quarterly_headcount_plan")
    action = env.sample_oracle_action()
    hidden_goal = env._hidden_goal  # noqa: SLF001
    assert hidden_goal is not None
    assert action.belief_report is not None
    assert action.belief_report.archetype_probs[hidden_goal.archetype.value] == 1.0


def test_task7_shift_occurs_with_reaction_window_when_present():
    env = LatentGoalOpsEnvironment()
    observed_shift_steps: list[int] = []
    observed_shift_types: list[str] = []
    observed_shift_durations: list[int] = []
    for seed in range(40):
        env.reset(seed=seed, task_id="task7_quarterly_headcount_plan")
        hidden_goal = env._hidden_goal  # noqa: SLF001
        assert hidden_goal is not None
        if hidden_goal.shift_step is not None:
            observed_shift_steps.append(hidden_goal.shift_step)
            observed_shift_types.append(hidden_goal.shift_type)
            observed_shift_durations.append(hidden_goal.shift_duration_steps)
    assert observed_shift_steps
    assert all(step <= 1 for step in observed_shift_steps)
    assert all(shift_type == "abrupt" for shift_type in observed_shift_types)
    assert all(duration == 1 for duration in observed_shift_durations)


def test_task7_goal_history_tracks_active_goal_after_shift():
    env = LatentGoalOpsEnvironment()
    shifted_seed = None
    for seed in range(100):
        env.reset(seed=seed, task_id="task7_quarterly_headcount_plan")
        hidden_goal = env._hidden_goal  # noqa: SLF001
        if hidden_goal is not None and hidden_goal.shift_step is not None:
            shifted_seed = seed
            break
    assert shifted_seed is not None

    observation = env.reset(seed=shifted_seed, task_id="task7_quarterly_headcount_plan")
    while not observation.done:
        observation = env.step(env.sample_oracle_action())

    hidden_goal = env._hidden_goal  # noqa: SLF001
    assert hidden_goal is not None
    goal_history = env._episode["goal_history"]  # noqa: SLF001
    assert len(goal_history) == env.state.step_count + 1
    assert goal_history[0] == hidden_goal.archetype.value
    assert hidden_goal.shift_goal is not None
    assert hidden_goal.shift_goal.value in goal_history[hidden_goal.shift_step + 1 :]


def test_task3_horizon_override_caps_delayed_effect_schedule():
    env = LatentGoalOpsEnvironment(experiment_config=ExperimentConfig(task3_horizon_override=3))
    env.reset(seed=31, task_id="task3_startup_week")
    env.step(env.sample_heuristic_action())
    assert all(effect.scheduled_for_step <= 3 for effect in env._episode["pending_effects"])  # noqa: SLF001
