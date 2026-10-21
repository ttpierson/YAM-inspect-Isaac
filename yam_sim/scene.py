"""Bimanual YAM tabletop scene: two arms mounted on the near edge of a table."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from yam_sim.assets import YAM_CFG

TABLE_HEIGHT = 0.75
TABLE_SIZE = (0.8, 1.2, 0.04)
TABLE_CENTER_X = 0.35
# Arms sit 0.6 m apart, both reaching along +X across the table.
ARM_SPACING = 0.6


def _on_table(x: float, y: float, height: float) -> tuple[float, float, float]:
    return (x, y, TABLE_HEIGHT + height / 2)


@configclass
class BimanualYamSceneCfg(InteractiveSceneCfg):
    """Two YAM arms, a table, a few graspable objects and an overview camera."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_HEIGHT - TABLE_SIZE[2] / 2)),
    )

    left_arm = YAM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        init_state=YAM_CFG.init_state.replace(pos=(0.0, ARM_SPACING / 2, TABLE_HEIGHT)),
    )
    right_arm = YAM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightArm",
        init_state=YAM_CFG.init_state.replace(pos=(0.0, -ARM_SPACING / 2, TABLE_HEIGHT)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_on_table(0.35, 0.12, 0.04)),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.09,
            height=0.015,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.90)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_on_table(0.38, -0.12, 0.015)),
    )

    overview_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        # Behind-and-above the arms, looking along +X and pitched 30 degrees down.
        offset=CameraCfg.OffsetCfg(pos=(-1.0, 0.0, 1.65), rot=(0.9659, 0.0, 0.2588, 0.0), convention="world"),
    )
