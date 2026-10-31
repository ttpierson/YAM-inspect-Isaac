"""Arena policies for the YAM tabletop.

- ``inspect_robots``: runs any Inspect Robots policy (``agent``, ``molmoact2``, ``gr00t``, ...)
  in an Arena evaluation. The policy sees the real rig's ``yam_arms`` observations and its
  14-D actions drive the arms, so Arena's metrics score a policy written for the real arms.
- ``yam_demo``: a scripted bimanual motion that exercises contact. The left arm picks the cube
  and sets it down elsewhere, the right arm presses its closed fingers onto the plate. A
  damped-least-squares IK layer turns tool-centre-point waypoints into joint targets.

Pass either to an Arena runner by its class path, e.g.
``--policy_type yam_arena.arena.policies.InspectRobotsPolicy``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from yam_arena.rig import (
    ARM_JOINTS,
    CUBE_POS,
    CUBE_SIZE,
    FINGER_CLOSED,
    FINGER_OPEN,
    PLATE_HEIGHT,
    PLATE_POS,
    TABLE_HEIGHT,
    TCP_OFFSET,
    action_limits,
    embodiment_info,
    observation_from_arena,
)


@dataclass
class InspectRobotsPolicyCfg(PolicyCfg):
    """Configure an Inspect Robots policy running in Arena."""

    inspect_policy: str = "agent"
    """Registered Inspect Robots policy name (see `inspect-robots list policies`)."""

    policy_args: str = "{}"
    """Keyword arguments for the policy factory, as a JSON object (like `inspect-robots run -P`)."""

    cam_height: int = 224
    """Height the policy's camera images are resized to."""

    cam_width: int = 224
    """Width the policy's camera images are resized to."""


@register_policy
class InspectRobotsPolicy(PolicyBase[InspectRobotsPolicyCfg]):
    """An Inspect Robots policy, speaking the real rig's ``yam_arms`` contract.

    Action chunks run open-loop: the policy is asked again once ``replan_interval`` actions of
    its chunk (default: all of them) have been executed. One environment at a time.
    """

    name = "inspect_robots"

    def __init__(self, config: InspectRobotsPolicyCfg):
        from inspect_robots.registry import resolve

        super().__init__(config)
        self.policy = resolve("policy", config.inspect_policy, **json.loads(config.policy_args))
        self.info = None  # bound on the first step, when the environment's control rate is known
        self._low, self._high = action_limits()
        self._queue: deque[np.ndarray] = deque()
        self._needs_reset = True
        self.task_description = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._queue.clear()
        self._needs_reset = True

    def get_action(self, env: gym.Env, observation: dict) -> torch.Tensor:
        from inspect_robots import Scene

        base = env.unwrapped
        assert base.num_envs == 1, "InspectRobotsPolicy drives one environment; use --num_envs 1"
        if self.info is None:
            self.info = embodiment_info(self.config.cam_height, self.config.cam_width, 1.0 / base.step_dt)
            bind = getattr(self.policy, "bind", None)
            if bind is not None:
                bind(self.info)
        if self._needs_reset:
            self.policy.reset(Scene(id="yam_tabletop", instruction=self.task_description or ""))
            self._needs_reset = False
        if not self._queue:
            obs = observation_from_arena(
                observation,
                cam_height=self.config.cam_height,
                cam_width=self.config.cam_width,
                instruction=self.task_description,
                now=float(base.episode_length_buf[0]) * base.step_dt,
            )
            chunk = self.policy.act(obs)
            keep = getattr(self.policy.config, "replan_interval", None) or len(chunk)
            self._queue.extend(np.asarray(a.data, dtype=np.float64).reshape(-1) for a in chunk.actions[:keep])
        action = np.clip(self._queue.popleft(), self._low, self._high)
        return torch.tensor(action, dtype=torch.float32, device=base.device).unsqueeze(0)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if close is not None:
            close()


# -- scripted demo ----------------------------------------------------------------------

# Gripper pointing straight down, fingers closing along world Y (gripper X -> world Y,
# gripper Z -> world -Z).
DOWN = (0.70711, 0.70711, 0.0, 0.0)
# The TCP sits 13 mm above the pad tips, so this is the TCP height at which the tips touch the table.
TIPS_ON_TABLE = TABLE_HEIGHT + 0.013
HOVER = TABLE_HEIGHT + 0.2


@dataclass
class Waypoint:
    duration: float
    pos: tuple[float, float, float]
    finger: float


def _pick_and_place() -> list[Waypoint]:
    x, y, _ = CUBE_POS
    grasp_z = TIPS_ON_TABLE + CUBE_SIZE / 2
    drop = (x - 0.1, y + 0.08)
    return [
        Waypoint(1.0, (x, y, HOVER), FINGER_OPEN),
        Waypoint(2.5, (x, y, HOVER), FINGER_OPEN),
        Waypoint(2.0, (x, y, grasp_z), FINGER_OPEN),
        Waypoint(1.0, (x, y, grasp_z), FINGER_CLOSED),
        Waypoint(2.0, (x, y, HOVER), FINGER_CLOSED),
        Waypoint(2.0, (*drop, HOVER), FINGER_CLOSED),
        Waypoint(2.0, (*drop, grasp_z + 0.005), FINGER_CLOSED),
        Waypoint(1.0, (*drop, grasp_z + 0.005), FINGER_OPEN),
        Waypoint(2.0, (*drop, HOVER), FINGER_OPEN),
    ]


def _press_plate() -> list[Waypoint]:
    x, y, _ = PLATE_POS
    on_plate = TIPS_ON_TABLE + PLATE_HEIGHT
    return [
        Waypoint(1.0, (x, y, HOVER), FINGER_CLOSED),
        Waypoint(2.5, (x, y, HOVER), FINGER_CLOSED),
        # Command the tips 1 cm *into* the plate: with working collisions they stop on its surface.
        Waypoint(2.5, (x, y, on_plate - 0.01), FINGER_CLOSED),
        Waypoint(2.0, (x, y, on_plate - 0.01), FINGER_CLOSED),
        Waypoint(2.0, (x, y, HOVER), FINGER_CLOSED),
    ]


SCRIPTS = {"left_arm": _pick_and_place(), "right_arm": _press_plate()}
DEMO_DURATION = max(sum(wp.duration for wp in script) for script in SCRIPTS.values())


@dataclass
class YamDemoPolicyCfg(PolicyCfg):
    """Configure the scripted YAM demo motion."""


class _ArmScript:
    """Turns one arm's TCP waypoints into joint targets with damped-least-squares IK."""

    def __init__(self, env, arm_name: str, script: list[Waypoint]):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.env = env
        self.arm = env.scene[arm_name]
        self.script = script
        self.device = self.arm.device
        self.arm_ids = self.arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        self.ee_idx = self.arm.find_bodies("gripper")[0][0]
        self.offset = torch.tensor([TCP_OFFSET], device=self.device).expand(env.num_envs, -1)
        self.down = torch.tensor(DOWN, device=self.device).expand(env.num_envs, -1)
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.05}),
            num_envs=env.num_envs,
            device=self.device,
        )
        self.start_pos, self.start_quat = self.tcp_pose()

    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils.math import quat_apply

        quat = self.arm.data.body_quat_w.torch[:, self.ee_idx]
        pos = self.arm.data.body_pos_w.torch[:, self.ee_idx] + quat_apply(quat, self.offset)
        return pos, quat

    def restart(self, env_ids: torch.Tensor) -> None:
        pos, quat = self.tcp_pose()
        self.start_pos[env_ids], self.start_quat[env_ids] = pos[env_ids], quat[env_ids]

    def _target(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Smoothly interpolate the script; the first segment also turns the gripper to point down."""
        from isaaclab.utils.math import quat_slerp

        origins = self.env.scene.env_origins
        pos, quat = self.start_pos.clone(), self.start_quat.clone()
        finger = torch.full_like(t, FINGER_OPEN)
        prev_pos, prev_quat, prev_finger = self.start_pos, self.start_quat, torch.full_like(t, FINGER_OPEN)
        seg_start = torch.zeros_like(t)
        done = torch.zeros_like(t, dtype=torch.bool)
        for wp in self.script:
            goal = torch.tensor(wp.pos, device=self.device) + origins
            active = ~done & (t < seg_start + wp.duration)
            s = ((t - seg_start) / wp.duration).clamp(0.0, 1.0)
            s = s * s * (3 - 2 * s)
            for i in torch.nonzero(active).flatten().tolist():
                quat[i] = quat_slerp(prev_quat[i], self.down[i], float(s[i]))
            pos = torch.where(active[:, None], prev_pos + s[:, None] * (goal - prev_pos), pos)
            finger = torch.where(active, prev_finger + s * (wp.finger - prev_finger), finger)
            done |= active
            seg_start = seg_start + wp.duration
            prev_pos, prev_quat, prev_finger = goal, self.down, torch.full_like(t, wp.finger)
        # Past the end of the script: hold the last waypoint.
        pos = torch.where(done[:, None], pos, prev_pos)
        quat = torch.where(done[:, None], quat, prev_quat)
        finger = torch.where(done, finger, prev_finger)
        return pos, quat, finger

    def action(self, t: torch.Tensor) -> torch.Tensor:
        """This arm's 7-D action: IK joint targets toward the waypoint at time `t`, and the gripper opening."""
        from isaaclab.utils.math import skew_symmetric_matrix

        pos, quat, finger = self._target(t)
        tcp_pos, tcp_quat = self.tcp_pose()
        jac = self.arm.data.body_link_jacobian_w.torch[:, self.ee_idx, :, self.arm_ids].clone()
        # Shift the Jacobian from the gripper link origin to the TCP: v_tcp = v_link + w x r.
        r = tcp_pos - self.arm.data.body_pos_w.torch[:, self.ee_idx]
        jac[:, 0:3] -= torch.bmm(skew_symmetric_matrix(r), jac[:, 3:6])
        self.ik.set_command(torch.cat([pos, quat], dim=-1))
        q = self.ik.compute(tcp_pos, tcp_quat, jac, self.arm.data.joint_pos.torch[:, self.arm_ids])
        return torch.cat([q, (finger / FINGER_OPEN).unsqueeze(-1)], dim=-1)


@register_policy
class YamDemoPolicy(PolicyBase[YamDemoPolicyCfg]):
    """The scripted bimanual contact demo, restarted at every episode."""

    name = "yam_demo"

    def __init__(self, config: YamDemoPolicyCfg):
        super().__init__(config)
        self._arms: list[_ArmScript] | None = None
        self._restart: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._arms is None:
            return
        env_ids = torch.arange(self._restart.numel()) if env_ids is None else env_ids
        self._restart[env_ids] = True

    def get_action(self, env: gym.Env, observation: dict) -> torch.Tensor:
        base = env.unwrapped
        if self._arms is None:
            self._arms = [_ArmScript(base, name, script) for name, script in SCRIPTS.items()]
            self._restart = torch.zeros(base.num_envs, dtype=torch.bool)
        restart = torch.nonzero(self._restart).flatten()
        if restart.numel():
            for arm in self._arms:
                arm.restart(restart.to(base.device))
            self._restart[:] = False
        # Aim for where the script will be at the end of this control step.
        t = (base.episode_length_buf.float() + 1.0) * base.step_dt
        return torch.cat([arm.action(t) for arm in self._arms], dim=-1)
