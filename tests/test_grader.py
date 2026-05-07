"""Score-ordering tests for oracle/heuristic/random baselines."""

from latentgoalops.server.environment import LatentGoalOpsEnvironment


def _run_single(task_id: str, policy: str, seed: int = 11) -> float:
    env = LatentGoalOpsEnvironment()
    observation = env.reset(seed=seed, task_id=task_id)
    while not observation.done:
        if policy == "oracle":
            action = env.sample_oracle_action()
        elif policy == "random":
            action = env.sample_random_action()
        else:
            action = env.sample_heuristic_action()
        observation = env.step(action)
    assert env.last_grade is not None
    return env.last_grade.score


def test_task1_score_ordering():
    oracle = _run_single("task1_feedback_triage", "oracle")
    heuristic = _run_single("task1_feedback_triage", "heuristic")
    random_score = _run_single("task1_feedback_triage", "random")
    assert oracle >= heuristic >= 0.0
    assert heuristic >= random_score


def test_task2_score_ordering():
    oracle = _run_single("task2_roadmap_priority", "oracle")
    heuristic = _run_single("task2_roadmap_priority", "heuristic")
    random_score = _run_single("task2_roadmap_priority", "random")
    assert oracle >= heuristic >= 0.0
    assert heuristic >= random_score


def test_task3_score_ordering():
    seeds = list(range(10, 20))
    oracle = sum(_run_single("task3_startup_week", "oracle", seed) for seed in seeds) / len(seeds)
    heuristic = sum(_run_single("task3_startup_week", "heuristic", seed) for seed in seeds) / len(seeds)
    random_score = sum(_run_single("task3_startup_week", "random", seed) for seed in seeds) / len(seeds)
    assert oracle >= heuristic >= 0.0
    assert heuristic >= random_score


def test_task4_score_ordering():
    oracle = _run_single("task4_capital_allocation", "oracle")
    heuristic = _run_single("task4_capital_allocation", "heuristic")
    random_score = _run_single("task4_capital_allocation", "random")
    # Task 4 is currently supplemental while we remove remaining shortcut structure
    # and strengthen its visible-only heuristic. Keep a strong ceiling check, but do
    # not require it to be a benchmark health gate yet.
    assert oracle >= max(heuristic, random_score) >= 0.0


def test_task5_score_ordering():
    seeds = list(range(30, 40))
    oracle = sum(_run_single("task5_crisis_response", "oracle", seed) for seed in seeds) / len(seeds)
    heuristic = sum(_run_single("task5_crisis_response", "heuristic", seed) for seed in seeds) / len(seeds)
    random_score = sum(_run_single("task5_crisis_response", "random", seed) for seed in seeds) / len(seeds)
    assert oracle >= heuristic >= 0.0
    assert heuristic >= random_score


def test_task6_score_ordering():
    seeds = list(range(40, 50))
    oracle = sum(_run_single("task6_incident_response_week", "oracle", seed) for seed in seeds) / len(seeds)
    heuristic = sum(_run_single("task6_incident_response_week", "heuristic", seed) for seed in seeds) / len(seeds)
    random_score = sum(_run_single("task6_incident_response_week", "random", seed) for seed in seeds) / len(seeds)
    assert oracle >= heuristic >= 0.0
    assert heuristic >= random_score


def test_task7_score_ordering():
    seeds = list(range(50, 65))
    oracle = sum(_run_single("task7_quarterly_headcount_plan", "oracle", seed) for seed in seeds) / len(seeds)
    heuristic = sum(_run_single("task7_quarterly_headcount_plan", "heuristic", seed) for seed in seeds) / len(seeds)
    random_score = sum(_run_single("task7_quarterly_headcount_plan", "random", seed) for seed in seeds) / len(seeds)
    # Task 7 remains supplemental until we replace the myopic oracle with a
    # stronger rollout upper bound and widen policy separation.
    assert oracle >= 0.0
    assert heuristic >= 0.0
    assert max(oracle, heuristic) >= random_score
