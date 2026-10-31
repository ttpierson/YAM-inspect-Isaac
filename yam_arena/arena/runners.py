"""Arena's experiment and policy runners, with the YAM tabletop environment registered.

Arena registers environments by importing its own `isaaclab_arena_environments` package once
the Isaac Sim app is up; it has no plugin hook. These wrappers add ``yam_tabletop`` to that
step, so Arena's unmodified runners can select it by name, then hand over to them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _register_yam_environment() -> None:
    import isaaclab_arena_environments.cli as environments_cli

    arena_registration = environments_cli.ensure_environments_registered

    def ensure_environments_registered() -> None:
        arena_registration()
        import yam_arena.arena.environment  # noqa: F401  (registers yam_tabletop)

    environments_cli.ensure_environments_registered = ensure_environments_registered


def _prepare_process() -> None:
    # Both must be set before Isaac Sim starts: Kit asks about the EULA on stdin at launch.
    if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X1").exists():
        os.environ["DISPLAY"] = ":1"  # the launchable's noVNC desktop
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    _register_yam_environment()


def run_experiment(experiment_config: str, args: list[str]) -> None:
    """Run an Arena Experiment YAML with Arena's experiment runner."""
    _prepare_process()
    from isaaclab_arena.evaluation import experiment_runner

    sys.argv = ["experiment_runner", "--experiment_config", experiment_config, *args]
    experiment_runner.main()


def run_policy(args: list[str]) -> None:
    """Run Arena's policy runner with `args` (its own flags, then ``yam_tabletop`` and the environment's)."""
    _prepare_process()
    from isaaclab_arena.evaluation import policy_runner

    sys.argv = ["policy_runner", *args]
    policy_runner.main()
