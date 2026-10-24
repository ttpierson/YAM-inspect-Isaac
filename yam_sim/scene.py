"""Bimanual YAM tabletop scene: two arms mounted on the near edge of a table.

The arms and the policy cameras always come from this config. Everything else (table,
objects, lights, ground) is the *environment*: the built-in one below, or a scene file
saved by `yam-sim edit` (see `scene_cfg`).
"""

import re
from pathlib import Path

import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from yam_sim.assets import ARM_JOINTS, YAM_CFG

# 240 Hz physics keeps finger/object contacts stable; render every 4th step (60 Hz).
PHYSICS_DT = 1 / 240
RENDER_INTERVAL = 4

TABLE_HEIGHT = 0.75
TABLE_SIZE = (0.8, 1.2, 0.04)
TABLE_CENTER_X = 0.35
# Arms sit 0.6 m apart, both reaching along +X across the table.
ARM_SPACING = 0.6
CUBE_SIZE = 0.04
CUBE_POS = (0.35, 0.12, TABLE_HEIGHT + CUBE_SIZE / 2)
PLATE_HEIGHT = 0.015
PLATE_POS = (0.38, -0.12, TABLE_HEIGHT + PLATE_HEIGHT / 2)
# Cameras follow the robocurve YAM rig: a RealSense D435 "top" camera behind and above the
# arms, and a D405 on each wrist. Focal lengths reproduce each sensor's colour HFOV on the
# default 20.955 mm aperture (D435 69 deg, D405 87 deg). 640x360 matches robocurve's frames.
CAM_HEIGHT, CAM_WIDTH = 360, 640
D435_FOCAL = 15.24
D405_FOCAL = 11.04
# Top camera: between the arms, 0.19 m behind and 0.43 m above their bases, pitched 45 deg
# down. Solved from robocurve's roboharm setup photos: the arm bases fall just past the
# bottom corners of the frame and the far table edge sits at its top.
TOP_CAM_POS = (-0.19, 0.0, TABLE_HEIGHT + 0.43)
TOP_CAM_ROT = (0.9239, 0.0, 0.3827, 0.0)
# Wrist camera in the `gripper` link frame (ROS optical convention: +Z forward, +Y down).
# It sits 8 cm off the -Y face (the side toward the arm base when the gripper points down)
# and looks along the approach axis, tilted 8 deg toward the fingers, with image right along
# the finger travel. The open fingertips land in the bottom corners, as in robocurve's real
# wrist frames.
WRIST_CAM_POS = (0.0, -0.08, 0.0)
WRIST_CAM_ROT = (0.9976, -0.0698, 0.0, 0.0)
# Grippy rubber-like contact for the graspable objects (PhysX averages it with the fingers' default 0.5).
GRIPPY = sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0)


@configclass
class BimanualYamSceneCfg(InteractiveSceneCfg):
    """Two YAM arms, a table, a few graspable objects, an overview camera
    and the rig's three policy cameras (`top_cam`, `left_cam`, `right_cam`)."""

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
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=GRIPPY,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS),
    )
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.09,
            height=PLATE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=GRIPPY,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.90)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PLATE_POS),
    )

    # Net contact force on every arm link; `debug_vis` marks links in contact in the GUI.
    left_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/LeftArm/.*", debug_vis=True)
    right_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/RightArm/.*", debug_vis=True)

    overview_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        # Behind-and-above the arms, looking along +X and pitched 30 degrees down.
        offset=CameraCfg.OffsetCfg(pos=(-1.0, 0.0, 1.65), rot=(0.9659, 0.0, 0.2588, 0.0), convention="world"),
    )

    top_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCam",
        height=CAM_HEIGHT,
        width=CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=D435_FOCAL, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=TOP_CAM_POS, rot=TOP_CAM_ROT, convention="world"),
    )
    # The wrist cameras live at the scene root and `update_wrist_cams` moves them onto the
    # gripper mounts: a camera prim parented under an articulation link stays where the link
    # was at import instead of following the simulated link.
    left_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/LeftWristCam",
        height=CAM_HEIGHT,
        width=CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=D405_FOCAL, clipping_range=(0.01, 20.0)),
    )
    right_cam = left_cam.replace(prim_path="{ENV_REGEX_NS}/RightWristCam")


def update_wrist_cams(scene: InteractiveScene) -> None:
    """Put each wrist camera at its mount on that arm's `gripper` link; call before rendering."""
    for side in ("left", "right"):
        if f"{side}_cam" not in scene.sensors:
            continue
        arm, cam = scene[f"{side}_arm"], scene[f"{side}_cam"]
        body = arm.find_bodies("gripper")[0][0]
        link_pos, link_quat = arm.data.body_pos_w[:, body], arm.data.body_quat_w[:, body]
        mount_pos = torch.tensor([WRIST_CAM_POS], device=link_pos.device)
        mount_quat = torch.tensor([WRIST_CAM_ROT], device=link_pos.device)
        cam.set_world_poses(
            link_pos + quat_apply(link_quat, mount_pos), quat_mul(link_quat, mount_quat), convention="ros"
        )


def hold_arms(scene: InteractiveScene) -> None:
    """Gravity-compensate both arms so they hold their joint targets; call every physics step."""
    for name in ("left_arm", "right_arm"):
        arm = scene[name]
        arm_ids = arm.find_joints(ARM_JOINTS, preserve_order=True)[0]
        gravity = arm.root_physx_view.get_gravity_compensation_forces()[:, arm_ids]
        arm.set_joint_effort_target(gravity, joint_ids=arm_ids)


# -- scene files ----------------------------------------------------------------------

ENV_NS = "/World/envs/env_0"
# Prims under ENV_NS that come from this config, never from a scene file.
CONFIG_PRIMS = {"LeftArm", "RightArm", "TopCam", "LeftWristCam", "RightWristCam", "OverviewCam"}
# Prims under /World that Isaac Lab generates for every scene: the env tree and the
# per-env collision filter groups (stale copies of those would filter the wrong colliders).
GENERATED_PRIMS = {"envs", "collisions"}
# Keeps the anonymous environment layer alive while a stage uses it.
_loaded_layers: list[Sdf.Layer] = []


def _environment_roots(layer: Sdf.Layer) -> list[Sdf.Path]:
    """Top-level environment prims in `layer`: /World/* and ENV_NS/*, minus the config's and Isaac Lab's own."""
    world = layer.GetPrimAtPath("/World")
    if world is None:
        return []
    roots = [child.path for child in world.nameChildren if child.name not in GENERATED_PRIMS]
    env = layer.GetPrimAtPath(ENV_NS)
    if env is not None:
        roots += [child.path for child in env.nameChildren if child.name not in CONFIG_PRIMS]
    return roots


def _copy_environment(src: Sdf.Layer, dst: Sdf.Layer) -> int:
    """Copy the environment prims of a flattened `src` into `dst` at the same paths."""
    roots = _environment_roots(src)
    for path in roots:
        for parent in path.GetParentPath().GetPrefixes():
            if dst.GetPrimAtPath(parent) is None:
                spec = Sdf.CreatePrimInLayer(dst, parent)
                spec.specifier, spec.typeName = Sdf.SpecifierDef, "Xform"
        Sdf.CopySpec(src, path, dst, path)
    dst.defaultPrim = "World"
    dst.pseudoRoot.SetInfo("upAxis", "Z")
    dst.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    return len(roots)


def save_environment(path: str | Path) -> int:
    """Write the current stage's environment (not the arms or cameras) to `path`; return the prim count.

    The stage is flattened first, so the file is self-contained: assets dragged in from a
    library are copied in, not referenced.
    """
    flat = omni.usd.get_context().get_stage().Flatten()
    out = Sdf.Layer.CreateAnonymous(".usda")
    count = _copy_environment(flat, out)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.Export(str(path))
    return count


def _rigid_object_cfgs(layer: Sdf.Layer) -> dict[str, RigidObjectCfg]:
    """One RigidObjectCfg per free rigid body in `layer`, starting at its authored pose."""
    stage = Usd.Stage.Open(layer)
    cfgs: dict[str, RigidObjectCfg] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parent = prim.GetParent()
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        if parent:  # part of a larger rigid body
            continue
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = tuple(xf.ExtractTranslation())
        q = xf.RemoveScaleShear().ExtractRotationQuat()
        name = re.sub(r"\W", "_", prim.GetName()).lower()
        while name in cfgs or hasattr(BimanualYamSceneCfg, name):
            name += "_"
        cfgs[name] = RigidObjectCfg(
            prim_path=str(prim.GetPath()),
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(q.GetReal(), *q.GetImaginary())),
        )
    return cfgs


def scene_cfg(scene_file: str | Path | None = None, *, cameras: bool = True) -> BimanualYamSceneCfg:
    """The scene config for one env, with the environment from `scene_file` if given.

    Call after the SimulationContext exists and before building the InteractiveScene: a
    scene file is composed into the open stage as a sublayer. Every free rigid body in it
    becomes a rigid object (named after its prim, lower-cased) that resets to where the file
    put it. Any saved stage works, including one from Isaac Sim's own File > Save As: the
    arms and cameras in it are ignored, because they always come from this config.
    """
    cfg = BimanualYamSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.overview_cam = None
    if not cameras:
        cfg.top_cam = cfg.left_cam = cfg.right_cam = None
    if scene_file is None:
        return cfg
    scene_file = Path(scene_file).resolve()
    if not scene_file.is_file():
        raise FileNotFoundError(f"scene file {scene_file} does not exist")
    env = Sdf.Layer.CreateAnonymous("yam_scene.usda")
    _copy_environment(Usd.Stage.Open(str(scene_file)).Flatten(), env)
    _loaded_layers.append(env)
    omni.usd.get_context().get_stage().GetRootLayer().subLayerPaths.insert(0, env.identifier)
    cfg.ground = cfg.light = cfg.table = cfg.cube = cfg.plate = None
    for name, obj in _rigid_object_cfgs(env).items():
        setattr(cfg, name, obj)
    return cfg
