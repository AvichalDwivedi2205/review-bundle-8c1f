"""Determinism checks."""

from latentgoalops.server.environment import LatentGoalOpsEnvironment


def test_task1_reset_is_deterministic():
    env_a = LatentGoalOpsEnvironment()
    env_b = LatentGoalOpsEnvironment()
    obs_a = env_a.reset(seed=123, task_id="task1_feedback_triage")
    obs_b = env_b.reset(seed=123, task_id="task1_feedback_triage")
    assert obs_a.model_dump(mode="json") == obs_b.model_dump(mode="json")


def test_task3_first_observation_is_deterministic():
    env_a = LatentGoalOpsEnvironment()
    env_b = LatentGoalOpsEnvironment()
    obs_a = env_a.reset(seed=123, task_id="task3_startup_week")
    obs_b = env_b.reset(seed=123, task_id="task3_startup_week")
    assert obs_a.model_dump(mode="json") == obs_b.model_dump(mode="json")

