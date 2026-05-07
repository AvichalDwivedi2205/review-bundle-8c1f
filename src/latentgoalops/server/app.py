"""FastAPI app for the LatentGoalOps environment."""

from __future__ import annotations

from fastapi import APIRouter
from openenv.core.env_server.http_server import create_app

from latentgoalops.models import LatentGoalOpsAction, LatentGoalOpsObservation
from latentgoalops.server.environment import LatentGoalOpsEnvironment


_shared_env = LatentGoalOpsEnvironment()


def _env_factory() -> LatentGoalOpsEnvironment:
    """Return the shared environment instance used by the HTTP server."""
    return _shared_env


app = create_app(
    _env_factory,
    LatentGoalOpsAction,
    LatentGoalOpsObservation,
    env_name="latentgoalops",
    max_concurrent_envs=1,
)

router = APIRouter()


@router.get("/tasks")
def tasks() -> list[dict]:
    """Return the human-readable task catalog."""
    return [task.model_dump(mode="json") for task in LatentGoalOpsEnvironment.describe_tasks()]


@router.get("/recovery-policy")
def recovery_policy() -> dict:
    """Return the token-exhaustion recovery strategy documented for the project."""
    return {
        "failure_modes": [
            "invalid_or_revoked_token",
            "temporary_rate_or_capacity_limit",
            "account_or_budget_exhaustion",
        ],
        "behavior": {
            "401_403": "fail fast, mark checkpoint resumable, rotate credential source if configured",
            "429_5xx": "retry with exponential backoff and jitter",
            "budget_exhaustion": "persist checkpoint and stop cleanly rather than looping",
        },
    }


app.include_router(router)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the environment server directly."""
    import argparse
    import sys
    import uvicorn

    # Allow `uv run server --port ...` while still exposing a plain `main()` entrypoint
    # for OpenEnv validation and programmatic imports.
    if host == "0.0.0.0" and port == 8000 and len(sys.argv) > 1:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--host", default=host)
        parser.add_argument("--port", type=int, default=port)
        parsed_args, _ = parser.parse_known_args()
        host = parsed_args.host
        port = parsed_args.port

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
