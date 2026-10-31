"""Action and observation terms that give the Arena environment the real rig's contract.

Each arm takes 7 numbers: absolute targets for joint1..joint6 in rad and a gripper opening
from 0 (closed) to 1 (open). The joint-state observation packs both arms the same way.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from yam_arena.rig import ARM_JOINTS, ARMS, FINGER_JOINTS, FINGER_OPEN, WRIST_CAM_POS, WRIST_CAM_ROT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class YamArmAction(ActionTerm):
    """Absolute joint targets plus a 0..1 gripper opening for one YAM arm.

    It also carries the arm's wrist camera on the `gripper` link. A camera prim parented under
    an articulation link stays where the link was at import instead of following the simulated
    link, so the term moves the camera onto its mount before every physics step.
    """

    cfg: YamArmActionCfg
    _asset: Articulation

    def __init__(self, cfg: YamArmActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._arm_ids = self._asset.find_joints(ARM_JOINTS, preserve_order=True)[0]
        self._finger_ids = self._asset.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        self._gripper_body = self._asset.find_bodies("gripper")[0][0]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(ARM_JOINTS) + 2, device=self.device)
        limits = self._asset.data.joint_pos_limits.torch[:, self._arm_ids]
        self._joint_low, self._joint_high = limits[..., 0], limits[..., 1]
        self._mount_pos = torch.tensor([WRIST_CAM_POS], device=self.device).expand(self.num_envs, -1)
        self._mount_quat = torch.tensor([WRIST_CAM_ROT], device=self.device).expand(self.num_envs, -1)
        # The managers are built after the first sim reset, so the bodies sit at their initial
        # (reset) pose now. Reset puts the camera back here: on reset the link data still holds
        # the previous episode's pose until physics steps again.
        self._reset_cam_pos, self._reset_cam_quat = self._mount_pose_w()
        self._reset_cam_pos -= env.scene.env_origins

    @property
    def action_dim(self) -> int:
        return len(ARM_JOINTS) + 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        joints = torch.clamp(actions[:, : len(ARM_JOINTS)], self._joint_low, self._joint_high)
        finger = torch.clamp(actions[:, len(ARM_JOINTS) :], 0.0, 1.0) * FINGER_OPEN
        self._processed_actions[:] = torch.cat([joints, finger.expand(-1, 2)], dim=-1)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._processed_actions[:, : len(ARM_JOINTS)], joint_ids=self._arm_ids
        )
        self._asset.set_joint_position_target_index(
            target=self._processed_actions[:, len(ARM_JOINTS) :], joint_ids=self._finger_ids
        )
        camera = self._camera()
        if camera is not None:
            camera.set_world_poses(*self._mount_pose_w(), convention="ros")

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._raw_actions[env_ids] = 0.0
        camera = self._camera()
        if camera is not None:
            pos = self._reset_cam_pos[env_ids] + self._env.scene.env_origins[env_ids]
            camera.set_world_poses(pos, self._reset_cam_quat[env_ids], env_ids=env_ids, convention="ros")

    def _camera(self):
        name = self.cfg.wrist_camera
        return self._env.scene.sensors.get(name) if name else None

    def _mount_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        link_pos = self._asset.data.body_pos_w.torch[:, self._gripper_body]
        link_quat = self._asset.data.body_quat_w.torch[:, self._gripper_body]
        return link_pos + quat_apply(link_quat, self._mount_pos), quat_mul(link_quat, self._mount_quat)


@configclass
class YamArmActionCfg(ActionTermCfg):
    """One YAM arm's 7-D action: joint1..joint6 targets (rad) and gripper opening (0..1)."""

    class_type: type[ActionTerm] = YamArmAction

    wrist_camera: str | None = MISSING
    """Scene name of the camera mounted on this arm's wrist, or None for no wrist camera."""


def yam_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Both arms' state as the real rig reports it: ``[left j1..j6, left grip, right j1..j6, right grip]``.

    Joints are in rad; each gripper is the mean opening of its two rack-coupled fingers, 0 closed to 1 open.
    """
    per_arm = []
    for name in ARMS:
        arm: Articulation = env.scene[name]
        q = arm.data.joint_pos.torch
        arm_ids = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = arm.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        grip = torch.clamp(q[:, finger_ids].mean(dim=-1, keepdim=True) / FINGER_OPEN, 0.0, 1.0)
        per_arm += [q[:, arm_ids], grip]
    return torch.cat(per_arm, dim=-1)



def reset_arm_to_ready(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> None:
    """Put an arm back in its spawn (ready) pose, at rest, with its drives targeting that pose."""
    arm: Articulation = env.scene[asset_cfg.name]
    joint_pos = arm.data.default_joint_pos.torch[env_ids].clone()
    joint_vel = arm.data.default_joint_vel.torch[env_ids].clone()
    arm.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    arm.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    arm.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
