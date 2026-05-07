"""Persistent client for the LatentGoalOps environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from latentgoalops.models import (
    LatentGoalOpsAction,
    LatentGoalOpsObservation,
    LatentGoalOpsState,
)


class LatentGoalOpsEnv(
    EnvClient[LatentGoalOpsAction, LatentGoalOpsObservation, LatentGoalOpsState]
):
    """Typed WebSocket client for a running LatentGoalOps server."""

    def _step_payload(self, action: LatentGoalOpsAction) -> dict[str, Any]:
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[LatentGoalOpsObservation]:
        observation = LatentGoalOpsObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> LatentGoalOpsState:
        return LatentGoalOpsState.model_validate(payload)

