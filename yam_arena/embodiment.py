"""Inspect Robots embodiment ``yam_arena``: the bimanual YAM tabletop, run in Isaac Lab Arena.

It speaks the same contract as the real rig's ``yam_arms`` embodiment (from
`inspect-robots-yam`), so any policy built for the real arms (``molmoact2``,
``gr00t``, the ``agent`` LLM policy) runs against the simulator unchanged:

- observation images ``top_cam`` (the overhead D435) and ``left_cam`` / ``right_cam``
  (the D405 on each wrist), RGB uint8 resized to ``cam_height`` x ``cam_width``;
- state ``joint_pos``: 14-D ``[left j0..j5, left gripper, right j0..j5, right gripper]``,
  joints in rad, grippers normalized (0 closed, 1 open);
- action: the same 14-D layout as absolute joint targets.

Each step holds the target for one control period (``control_hz``, 10 Hz like the real rig) of joint PD on
gravity-compensated arms, the way i2rt drives the real arms.

The simulation is Arena's ``yam_tabletop`` environment, and its task is the episode's judge:
a step that puts the object on its destination ends the episode with termination reason
``"success"`` (score it with ``--scorer success_at_end``); knocking it off the table ends it
with ``"object_dropped"``. ``-E scene=SCENE.usda`` runs in a scene saved by ``yam-arena edit``,
and ``-E pick_up_object=... -E destination=...`` names that scene's objects for the task.

Importing this module is cheap: Isaac Sim only starts on the first ``reset()``.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

import numpy as np
from inspect_robots import Scene
from inspect_robots.embodiment import EmbodimentBase
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_yam.packing import pack

from yam_arena.rig import CONTROL_HZ, READY, action_limits, embodiment_info, observation_from_arena

# Settle time after a reset, so the first observation shows the arms and objects at rest.
SETTLE_S = 0.5
# Hold the ready pose with the grippers open while settling.
READY_ACTION = pack(np.append(READY, 1.0), np.append(READY, 1.0))


class YamArenaEmbodiment(EmbodimentBase):
    """Bimanual YAM in Isaac Lab Arena, observed through the rig's overhead and wrist cameras."""

    def __init__(
        self,
        *,
        headless: bool = True,
        device: str = "cuda:0",
        cam_height: int = 224,
        cam_width: int = 224,
        scene: str | None = None,
        pick_up_object: str = "cube",
        destination: str = "plate",
        episode_length_s: float = 600.0,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        self.headless = headless
        self.device = device
        self.cam_height, self.cam_width = cam_height, cam_width
        self.env_kwargs = {
            "scene": scene,
            "pick_up_object": pick_up_object,
            "destination": destination,
            "episode_length_s": episode_length_s,
            "control_hz": control_hz,
        }
        self._low, self._high = action_limits()
        self.info = embodiment_info(cam_height, cam_width, control_hz)
        self._app: Any = None
        self._env: Any = None
        self._instruction: str | None = None
        self._obs: dict | None = None

    # -- Isaac lifecycle ---------------------------------------------------------------

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher

        launch = {"headless": self.headless, "device": self.device, "enable_cameras": True}
        if not self.headless:
            launch["visualizer"] = ["kit"]
        self._app = AppLauncher(launch).app

        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg

        from yam_arena.arena.environment import YamTabletopEnvironment, YamTabletopEnvironmentCfg

        arena_env = YamTabletopEnvironment().build(YamTabletopEnvironmentCfg(enable_cameras=True, **self.env_kwargs))
        builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, env_spacing=3.0, device=self.device))
        self._env = builder.make_registered()

    def close(self) -> None:
        """Close the environment; the app closes at interpreter exit.

        ``SimulationApp.close()`` ends the process, which would swallow whatever the caller
        prints afterwards (the inspect-robots CLI prints its results after ``close()``).
        """
        if self._env is not None:
            self._env.close()
        if self._app is not None:
            atexit.register(self._app.close)
        self._app = self._env = None

    # -- Embodiment contract -----------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Return both arms to the ready pose, put every object back (shifted, seeded by `seed`), and settle."""
        self._ensure_env()
        self._instruction = scene.instruction
        self._obs, _ = self._env.reset(seed=seed)
        base = self._env.unwrapped
        for _ in range(max(1, round(SETTLE_S / base.step_dt))):
            self._obs, *_ = self._env.step(self._tensor(READY_ACTION))
        return self._observation()

    def step(self, action: Action) -> StepResult:
        """Hold the 14-D absolute joint target for one control period."""
        target = np.clip(np.asarray(action.data, dtype=np.float64).reshape(-1), self._low, self._high)
        observation = self._observation()  # fallback if the episode ends: Isaac Lab resets the env in step
        self._obs, _, terminated, truncated, _ = self._env.step(self._tensor(target))
        terminated, truncated = bool(terminated[0]), bool(truncated[0])
        reason = None
        if terminated or truncated:
            reason = self._termination_reason()
        else:
            observation = self._observation()
        return StepResult(observation=observation, terminated=terminated, truncated=truncated, termination_reason=reason)

    # -- internals ---------------------------------------------------------------------

    def _tensor(self, action: np.ndarray):
        import torch

        return torch.tensor(action, dtype=torch.float32, device=self._env.unwrapped.device).unsqueeze(0)

    def _observation(self) -> Observation:
        base = self._env.unwrapped
        return observation_from_arena(
            self._obs,
            cam_height=self.cam_height,
            cam_width=self.cam_width,
            instruction=self._instruction,
            now=float(base.episode_length_buf[0]) * base.step_dt,
        )

    def _termination_reason(self) -> str | None:
        """The termination term that fired on the last step: ``success``, a task failure, or ``time_out``."""
        manager = self._env.unwrapped.termination_manager
        for name in manager.active_terms:
            if bool(manager.get_term(name)[0]):
                return name
        return None


def yam_arena_embodiment(**kwargs: Any) -> YamArenaEmbodiment:
    """Embodiment ``yam_arena``; ``-E headless=false`` renders on DISPLAY (e.g. the noVNC desktop)."""
    return YamArenaEmbodiment(**kwargs)
