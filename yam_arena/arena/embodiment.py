"""Arena embodiment ``yam_bimanual``: two I2RT YAM arms and the robocurve rig's cameras.

The arms stand 0.6 m apart on the near edge of the table, both reaching along +X. The action
and the state observation are the real rig's 14-D ``yam_arms`` layout (see `yam_arena.mdp`);
the cameras are ``top_cam`` (overhead D435) and ``left_cam`` / ``right_cam`` (wrist D405s).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg

from yam_arena.arena.mdp import YamArmActionCfg, reset_arm_to_ready, yam_joint_pos
from yam_arena.assets import YAM_CFG
from yam_arena.rig import (
    ARM_SPACING,
    CAM_HEIGHT,
    CAM_WIDTH,
    D405_FOCAL,
    D435_FOCAL,
    TABLE_HEIGHT,
    TOP_CAM_POS,
    TOP_CAM_ROT,
)


@configclass
class YamBimanualSceneCfg:
    """The two arms, as entries of the composed scene."""

    left_arm: ArticulationCfg = YAM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        init_state=YAM_CFG.init_state.replace(pos=(0.0, ARM_SPACING / 2, TABLE_HEIGHT)),
    )
    right_arm: ArticulationCfg = YAM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightArm",
        init_state=YAM_CFG.init_state.replace(pos=(0.0, -ARM_SPACING / 2, TABLE_HEIGHT)),
    )


_WRIST_CAM = CameraCfg(
    prim_path="{ENV_REGEX_NS}/LeftWristCam",
    update_period=0.0,
    height=CAM_HEIGHT,
    width=CAM_WIDTH,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=D405_FOCAL, clipping_range=(0.01, 20.0)),
)


@configclass
class YamCameraCfg(ArenaCameraCfg):
    """The rig's three policy cameras."""

    top_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCam",
        update_period=0.0,
        height=CAM_HEIGHT,
        width=CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=D435_FOCAL, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=TOP_CAM_POS, rot=TOP_CAM_ROT, convention="world"),
    )
    # The wrist cameras live at the env root; YamArmAction keeps each on its gripper mount.
    left_cam: CameraCfg = _WRIST_CAM
    right_cam: CameraCfg = _WRIST_CAM.replace(prim_path="{ENV_REGEX_NS}/RightWristCam")


@configclass
class YamActionsCfg:
    """14-D action: the left arm's 7 numbers, then the right arm's."""

    left_arm = YamArmActionCfg(asset_name="left_arm", wrist_camera="left_cam")
    right_arm = YamArmActionCfg(asset_name="right_arm", wrist_camera="right_cam")


@configclass
class YamObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=yam_joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class YamEventsCfg:
    reset_left_arm = EventTermCfg(func=reset_arm_to_ready, mode="reset", params={"asset_cfg": SceneEntityCfg("left_arm")})
    reset_right_arm = EventTermCfg(
        func=reset_arm_to_ready, mode="reset", params={"asset_cfg": SceneEntityCfg("right_arm")}
    )


@register_asset
class YamBimanualEmbodiment(EmbodimentBase):
    """Two YAM arms on a table edge, observed through the rig's overhead and wrist cameras."""

    name = "yam_bimanual"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(self, enable_cameras: bool = False, **kwargs):
        super().__init__(enable_cameras=enable_cameras, **kwargs)
        self.scene_config = YamBimanualSceneCfg()
        self.camera_config = YamCameraCfg()
        self.action_config = YamActionsCfg()
        self.observation_config = YamObservationsCfg()
        self.event_config = YamEventsCfg()

    def get_scene_key(self) -> str:
        return "left_arm"

    def get_ee_frame_transformer_names(self) -> list[str]:
        return []
