"""Arena environment ``yam_tabletop``: the bimanual YAM rig putting an object on another.

Scene: a table with a red cube in front of the left arm and a white plate in front of the
right one, or the environment of a scene file saved by `yam-arena edit` (``scene``). Task:
Arena's PickAndPlaceTask, so an episode succeeds once ``pick_up_object`` has been lifted and
rests on ``destination``, and fails if it falls off the table. Each reset puts the arms in
their ready pose and shifts the built-in scene's objects by up to ``object_jitter`` in x and y.

Run it with any Arena policy through `yam-arena eval` (experiment YAML) or `yam-arena rollout`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from yam_arena.rig import (
    CUBE_POS,
    CONTROL_HZ,
    CUBE_SIZE,
    DROP_MARGIN,
    OBJECT_JITTER,
    PHYSICS_DT,
    PLATE_HEIGHT,
    PLATE_POS,
    PLATE_RADIUS,
    TABLE_CENTER_X,
    TABLE_HEIGHT,
    TABLE_SIZE,
    decimation,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class YamTabletopEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the YAM tabletop environment."""

    scene: str | None = None
    """Scene file saved by `yam-arena edit`; None for the built-in table, cube and plate."""

    pick_up_object: str = "cube"
    """Object to pick up: a built-in object, or a free rigid body of the scene file by prim name."""

    destination: str = "plate"
    """Object to put it on, named the same way."""

    task_description: str = "put the red cube on the plate"
    """Instruction handed to the policy."""

    episode_length_s: float = 60.0
    """Episode time limit in simulated seconds."""

    object_jitter: float = OBJECT_JITTER
    """Largest x and y shift of each built-in object at reset, in metres."""

    control_hz: float = CONTROL_HZ
    """Policy rate in Hz; physics runs at 240 Hz and each action is held for the steps in between."""


@register_environment
class YamTabletopEnvironment(ArenaEnvironmentFactory[YamTabletopEnvironmentCfg]):
    """Two YAM arms at a table, putting one object on another."""

    name = "yam_tabletop"
    _legacy_argparse_cfg_type = YamTabletopEnvironmentCfg

    def build(self, cfg: YamTabletopEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        from yam_arena.arena.embodiment import YamBimanualEmbodiment

        if cfg.scene is None:
            background, objects = _builtin_scene(cfg.object_jitter)
            assets = [background, *objects.values(), self.asset_registry.get_asset_by_name("ground_plane")(), _light()]
        else:
            background, objects = _scene_file(cfg.scene)
            assets = [background, *objects.values()]
        missing = {cfg.pick_up_object, cfg.destination} - objects.keys()
        if missing:
            print(f"[yam-arena] no object(s) {sorted(missing)} in the scene ({sorted(objects)}): running without a task")
            task = NoTask()
        else:
            task = PickAndPlaceTask(
                pick_up_object=objects[cfg.pick_up_object],
                destination_location=objects[cfg.destination],
                background_scene=background,
                episode_length_s=cfg.episode_length_s,
                task_description=cfg.task_description,
            )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=YamBimanualEmbodiment(enable_cameras=cfg.enable_cameras),
            scene=Scene(assets=assets),
            task=task,
            env_cfg_callback=lambda env_cfg: _rig_timing(env_cfg, cfg.control_hz),
        )


def _rig_timing(env_cfg, control_hz: float):
    """240 Hz physics, control at `control_hz` (the real rig's 10 Hz by default), one render per control step."""
    env_cfg.sim.dt = PHYSICS_DT
    env_cfg.decimation = decimation(control_hz)
    env_cfg.sim.render_interval = env_cfg.decimation
    return env_cfg


def _light():
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object_library import DomeLight

    return DomeLight(spawner_cfg=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))


def _builtin_scene(jitter: float):
    """The table (also the task's background) and the cube and plate on it, by name."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose, PoseRange

    from yam_arena.arena.objects import PrimitiveObject

    def on_table(pos: tuple[float, float, float]) -> PoseRange:
        x, y, z = pos
        return PoseRange(position_xyz_min=(x - jitter, y - jitter, z), position_xyz_max=(x + jitter, y + jitter, z))

    # Grippy rubber-like contact for the graspable objects (PhysX averages it with the fingers' default 0.5).
    grippy = sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0)
    table = PrimitiveObject(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30), roughness=0.7),
        ),
        initial_pose=Pose(position_xyz=(TABLE_CENTER_X, 0.0, TABLE_HEIGHT - TABLE_SIZE[2] / 2)),
        object_min_z=TABLE_HEIGHT - DROP_MARGIN,
    )
    cube = PrimitiveObject(
        name="cube",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=grippy,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
            activate_contact_sensors=True,
        ),
        initial_pose=on_table(CUBE_POS),
    )
    plate = PrimitiveObject(
        name="plate",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CylinderCfg(
            radius=PLATE_RADIUS,
            height=PLATE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=grippy,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.90)),
            activate_contact_sensors=True,
        ),
        initial_pose=on_table(PLATE_POS),
    )
    return table, {"cube": cube, "plate": plate}


def _scene_file(path: str):
    """A scene file as the background, and its free rigid bodies as objects, by lower-cased prim name."""
    from pathlib import Path

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType

    from yam_arena.arena.scene_file import BACKGROUND_NAME, object_names

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"scene file {path} does not exist")
    background = Background(
        name=BACKGROUND_NAME,
        usd_path=str(path),
        object_min_z=TABLE_HEIGHT - DROP_MARGIN,
        prim_path="{ENV_REGEX_NS}/" + BACKGROUND_NAME,
    )
    objects = {
        name: ObjectReference(
            name=name,
            prim_path="{ENV_REGEX_NS}/" + f"{BACKGROUND_NAME}/{relative}",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        for name, relative in object_names(path).items()
    }
    return background, objects
