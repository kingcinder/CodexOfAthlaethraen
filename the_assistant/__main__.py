"""Module entry point for running the assistant CLI as a package."""

from __future__ import annotations

from .assistant_cli import main


def cli_entrypoint() -> None:
    """Console script entry point."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - module execution hook
    cli_entrypoint()
