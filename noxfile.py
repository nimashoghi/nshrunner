"""Nox configuration for testing against multiple dependency versions."""

from __future__ import annotations

import nox

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests against different Python and Pydantic versions."""
    # Install dependencies
    deps: list[str] = []
    # This package
    deps.append(".[extra]")
    # Base deps
    deps.extend(["pytest", "pytest-cov", "ruff", "basedpyright"])

    session.install(*deps)

    # Run linting and type checking
    session.run("ruff", "check", "src")
    session.run("basedpyright", "src")

    # Run tests
    session.run("pytest")


if __name__ == "__main__":
    nox.main()
