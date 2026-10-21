"""Isaac Lab articulation config for one I2RT YAM arm (6 DoF + linear gripper)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from yam_sim import YAM_URDF

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
# Finger joints run from 0 (closed) to -0.04695 m (open) in the URDF.
FINGER_OPEN = -0.04695

YAM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(YAM_URDF),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        collider_type="convex_decomposition",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"joint[1-6]": 0.0, "joint[7-8]": FINGER_OPEN},
    ),
    actuators={
        # joints 1-3 are DM4340 motors (~27 Nm peak), 4-6 are DM4310 (~7 Nm).
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"], effort_limit_sim=27.0, velocity_limit_sim=10.0, stiffness=80.0, damping=5.0
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"], effort_limit_sim=7.0, velocity_limit_sim=10.0, stiffness=20.0, damping=1.0
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=FINGER_JOINTS, effort_limit_sim=50.0, velocity_limit_sim=0.5, stiffness=2000.0, damping=50.0
        ),
    },
)
"""One YAM arm. At zero joint angles the gripper's approach axis points along +X."""
