"""LatentGoalOps package."""

from latentgoalops.client import LatentGoalOpsEnv
from latentgoalops.models import LatentGoalOpsAction, LatentGoalOpsObservation

__all__ = [
    "LatentGoalOpsAction",
    "LatentGoalOpsEnv",
    "LatentGoalOpsObservation",
]


def main() -> None:
    print("LatentGoalOps is installed. Run `uv run server` or `uv run baseline`.")
