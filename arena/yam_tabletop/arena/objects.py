"""Arena objects spawned from Isaac Lab shape primitives instead of USD files."""

from __future__ import annotations

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose, PoseRange


class PrimitiveObject(Object):
    """An Arena object made from a shape spawner (cuboid, cylinder, ...).

    The shape's root prim is its rigid body, so contact sensors and contact filters attach
    there; Arena's own lookup searches a USD file for it, and there is none.
    """

    def __init__(
        self,
        name: str,
        object_type: ObjectType,
        spawner_cfg: SpawnerCfg,
        initial_pose: Pose | PoseRange | None = None,
        object_min_z: float | None = None,
    ):
        super().__init__(name=name, object_type=object_type, spawner_cfg=spawner_cfg)
        if initial_pose is not None:
            self.set_initial_pose(initial_pose)
        self.object_min_z = object_min_z
        """Height below which a task counts an object as dropped, when this object is the background."""

    def get_contact_sensor_prim_path(self, usd_path: str | None = None) -> str:
        return self.prim_path
