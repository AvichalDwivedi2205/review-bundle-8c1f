"""Root OpenEnv app wrapper for validation and deployment."""

from latentgoalops.server.app import app as app
from latentgoalops.server.app import main as _package_main

__all__ = ["app", "main"]


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the packaged application entrypoint."""
    _package_main(host=host, port=port)


if __name__ == "__main__":
    main()
