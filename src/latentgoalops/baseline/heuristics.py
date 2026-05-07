"""Non-LLM baseline policies."""

from __future__ import annotations

from latentgoalops.server.environment import LatentGoalOpsEnvironment


def choose_action(env: LatentGoalOpsEnvironment, policy: str):
    """Dispatch to one of the built-in policies."""
    if policy == "random":
        return env.sample_random_action()
    if policy == "oracle":
        return env.sample_oracle_action()
    return env.sample_heuristic_action()

