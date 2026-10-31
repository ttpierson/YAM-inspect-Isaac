"""Isaac Lab articulation config for one I2RT YAM arm (6 DoF + linear gripper).

Drive gains and joint friction come from i2rt's own controller config
(`i2rt/robots/config/yam_v1.yml` and `linear_4310.yml` at the vendored commit),
so the simulated arm responds to a joint target the way the real one does.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from yam_arena import YAM_URDF
from yam_arena.rig import FINGER_JOINTS, FINGER_OPEN, READY

# Gripper transmission (linear_4310.yml): 6.57 rad of motor travel moves the fingers 0.096 m
# apart in total, so each finger travels 0.0073 m per motor radian.
_FINGER_M_PER_RAD = 0.096 / 2 / 6.57

YAM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(YAM_URDF),
        # Isaac Lab 3's importer anchors the base with a joint to the world but keeps the
        # articulation root on the base link, so PhysX reports a floating base (is_fixed_base is
        # False, and Jacobians carry six base columns). fix_root_link can't relocate that root:
        # its properties are authored inside the imported USD.
        fix_base=True,
        merge_fixed_joints=False,
        # Each URDF <collision> is already one convex CoACD hull (make_collision_meshes.py), so
        # PhysX keeps it as-is instead of re-decomposing the whole link mesh.
        collision_type="Convex Hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        # Start generating contacts 5 mm out and rest at zero separation, so thin finger pads
        # register contact before they penetrate a small object.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        # i2rt drives the real arm with joint PD plus gravity compensation of the arm's own links.
        # Gravity-free links are the same thing, exactly: a held object still weighs on the fingers.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The finger carriages slide inside the gripper housing and past each other, so their
            # convex hulls overlap by design; self-collision would push the fingers apart.
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**{f"joint{i + 1}": q for i, q in enumerate(READY)}, "joint[7-8]": FINGER_OPEN},
    ),
    actuators={
        # Joints 1-3 are Damiao DM4340 (27 Nm peak, ~5.5 rad/s no-load at 24 V);
        # kp/kd and Coulomb friction from yam_v1.yml.
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            joint_effort_limit=27.0,
            joint_velocity_limit=5.5,
            stiffness=80.0,
            damping=5.0,
            friction=0.3,
        ),
        # Joints 4-6 are DM4310 (7 Nm peak, ~20 rad/s no-load at 24 V).
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            joint_effort_limit=7.0,
            joint_velocity_limit=20.0,
            stiffness=10.0,
            damping=1.5,
            friction=0.06,
        ),
        # One DM4310 drives both fingers through a rack; i2rt's driver caps grip force at 50 N.
        # Finger speed is the motor's no-load speed through the rack (~0.15 m/s).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=FINGER_JOINTS,
            joint_effort_limit=50.0,
            joint_velocity_limit=20.0 * _FINGER_M_PER_RAD,
            stiffness=5000.0,
            damping=100.0,
        ),
    },
)
"""One YAM arm. At zero joint angles the gripper's approach axis points along +X."""
