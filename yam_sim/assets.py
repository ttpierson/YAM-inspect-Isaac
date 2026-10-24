"""Isaac Lab articulation config for one I2RT YAM arm (6 DoF + linear gripper).

Drive gains and joint friction come from i2rt's own controller config
(`i2rt/robots/config/yam_v1.yml` and `linear_4310.yml` at the vendored commit),
so the simulated arm responds to a joint target the way the real one does.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from yam_sim import YAM_URDF

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
# Finger joints run from 0 (closed, pads touching) to -0.04695 m (open, pads 94 mm apart).
FINGER_OPEN = -0.04695
FINGER_CLOSED = 0.0

# A raised "ready" pose: shoulder up, elbow bent, wrist pitched down toward the table.
# The arm spawns here so the first physics step does not snap it from the folded zero pose.
READY = [0.0, 1.0, 1.0, -0.6, 0.0, 0.0]

# Tool centre point in the `gripper` link frame: between the finger pads, 13 mm short of
# the pad tips. The gripper's approach axis is +Z of that frame.
TCP_OFFSET = (0.0, 0.0, 0.134)

# Gripper transmission (linear_4310.yml): 6.57 rad of motor travel moves the fingers 0.096 m
# apart in total, so each finger travels 0.0073 m per motor radian.
_FINGER_M_PER_RAD = 0.096 / 2 / 6.57

YAM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(YAM_URDF),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        # Each URDF <collision> is already one convex CoACD hull (make_collision_meshes.py), so
        # PhysX keeps it as-is instead of re-decomposing the whole link mesh.
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        # Start generating contacts 5 mm out and rest at zero separation, so thin finger pads
        # register contact before they penetrate a small object.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
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
            effort_limit_sim=27.0,
            velocity_limit_sim=5.5,
            stiffness=80.0,
            damping=5.0,
            friction=0.3,
        ),
        # Joints 4-6 are DM4310 (7 Nm peak, ~20 rad/s no-load at 24 V).
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.0,
            stiffness=10.0,
            damping=1.5,
            friction=0.06,
        ),
        # One DM4310 drives both fingers through a rack; i2rt's driver caps grip force at 50 N.
        # Finger speed is the motor's no-load speed through the rack (~0.15 m/s).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=FINGER_JOINTS,
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0 * _FINGER_M_PER_RAD,
            stiffness=5000.0,
            damping=100.0,
        ),
    },
)
"""One YAM arm. At zero joint angles the gripper's approach axis points along +X."""
