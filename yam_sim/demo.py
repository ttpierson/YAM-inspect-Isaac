"""Scripted bimanual demo that exercises contact: the left arm picks the cube and sets it
down elsewhere, the right arm presses its closed fingers onto the plate.

Each arm is driven the way i2rt drives the real YAM: joint-space PD plus gravity
compensation, with a damped-least-squares IK layer turning Cartesian waypoints for the
tool centre point into joint targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_slerp, skew_symmetric_matrix

from yam_sim.assets import ARM_JOINTS, FINGER_CLOSED, FINGER_JOINTS, FINGER_OPEN, TCP_OFFSET
from yam_sim.scene import CUBE_POS, CUBE_SIZE, PLATE_POS, PLATE_HEIGHT, TABLE_HEIGHT

# Gripper pointing straight down, fingers closing along world Y (gripper X -> world Y,
# gripper Z -> world -Z).
DOWN = (0.0, 0.70711, 0.70711, 0.0)
# The TCP sits 13 mm above the pad tips, so this is the TCP height at which the tips touch the table.
TIPS_ON_TABLE = TABLE_HEIGHT + 0.013
HOVER = TABLE_HEIGHT + 0.2


@dataclass
class Waypoint:
    duration: float
    pos: tuple[float, float, float]
    grip: float


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


def reset_scene(scene: InteractiveScene) -> None:
    """Put the arms and objects back at their spawn state."""
    for arm in scene.articulations.values():
        arm.write_joint_state_to_sim(arm.data.default_joint_pos, arm.data.default_joint_vel)
        arm.set_joint_position_target(arm.data.default_joint_pos)
    for obj in scene.rigid_objects.values():
        state = obj.data.default_root_state.clone()
        state[:, :3] += scene.env_origins
        obj.write_root_state_to_sim(state)
    scene.reset()


def make_drivers(scene: InteractiveScene) -> list["YamDriver"]:
    return [YamDriver(scene[name], script, scene.env_origins[0]) for name, script in SCRIPTS.items()]


class YamDriver:
    """Turns TCP pose + gripper commands into joint targets and gravity-compensation torques."""

    def __init__(self, arm: Articulation, script: list[Waypoint], env_origin: torch.Tensor):
        self.arm = arm
        self.script = script
        self.device = arm.device
        self.arm_ids = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        self.finger_ids = arm.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        self.ee_idx = arm.find_bodies("gripper")[0][0]
        # With a fixed base the root body has no Jacobian row, so body indices shift by one.
        self.jac_idx = self.ee_idx - 1
        self.offset = torch.tensor([TCP_OFFSET], device=self.device)
        self.origin = env_origin
        self.down = torch.tensor(DOWN, device=self.device)
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.05}),
            num_envs=1,
            device=self.device,
        )
        self.start_pos, self.start_quat = None, None

    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self.arm.data.body_quat_w[:, self.ee_idx]
        pos = self.arm.data.body_pos_w[:, self.ee_idx] + quat_apply(quat, self.offset)
        return pos, quat

    def _target(self, t: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Smoothly interpolate the script; the first segment also turns the gripper to point down."""
        prev_pos, prev_quat, prev_grip = self.start_pos[0], self.start_quat[0], FINGER_OPEN
        for wp in self.script:
            goal = torch.tensor(wp.pos, device=self.device) + self.origin
            if t < wp.duration:
                s = t / wp.duration
                s = s * s * (3 - 2 * s)
                pos = prev_pos + s * (goal - prev_pos)
                quat = quat_slerp(prev_quat, self.down, s)
                return pos, quat, prev_grip + s * (wp.grip - prev_grip)
            t -= wp.duration
            prev_pos, prev_quat, prev_grip = goal, self.down, wp.grip
        return prev_pos, prev_quat, prev_grip

    @property
    def duration(self) -> float:
        return sum(wp.duration for wp in self.script)

    def step(self, t: float) -> None:
        if self.start_pos is None:
            self.start_pos, self.start_quat = self.tcp_pose()
        pos, quat, grip = self._target(t)
        tcp_pos, tcp_quat = self.tcp_pose()
        jac = self.arm.root_physx_view.get_jacobians()[:, self.jac_idx, :, self.arm_ids]
        # Shift the Jacobian from the gripper link origin to the TCP: v_tcp = v_link + w x r.
        r = tcp_pos - self.arm.data.body_pos_w[:, self.ee_idx]
        jac[:, 0:3] -= torch.bmm(skew_symmetric_matrix(r), jac[:, 3:6])
        self.ik.set_command(torch.cat([pos, quat]).unsqueeze(0))
        q = self.ik.compute(tcp_pos, tcp_quat, jac, self.arm.data.joint_pos[:, self.arm_ids])
        self.arm.set_joint_position_target(q, joint_ids=self.arm_ids)
        self.arm.set_joint_position_target(torch.full((1, 2), grip, device=self.device), joint_ids=self.finger_ids)
        gravity = self.arm.root_physx_view.get_gravity_compensation_forces()[:, self.arm_ids]
        self.arm.set_joint_effort_target(gravity, joint_ids=self.arm_ids)
