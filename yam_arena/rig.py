"""The simulated rig and its policy contract, without importing Isaac Sim.

Layout, timing and camera constants for the bimanual YAM tabletop, plus the conversions
between Arena observations and the real rig's ``yam_arms`` contract (from
`inspect-robots-yam`). Both the Inspect Robots embodiment and the Arena policy adapter use
them, and both must be usable before the Isaac Sim app starts, so nothing here imports
Isaac Lab.

Quaternions are (x, y, z, w), Isaac Lab 3's order.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from yam_arena import YAM_URDF

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
# Finger joints run from 0 (closed, pads touching) to -0.04695 m (open, pads 94 mm apart).
FINGER_OPEN = -0.04695
FINGER_CLOSED = 0.0
ARMS = ("left_arm", "right_arm")
# Scene entries of the wrist cameras, in ARMS order.
WRIST_CAMS = ("left_cam", "right_cam")

# A raised "ready" pose: shoulder up, elbow bent, wrist pitched down toward the table.
# The arms spawn and reset here so the first physics step does not snap them from the folded zero pose.
READY = [0.0, 1.0, 1.0, -0.6, 0.0, 0.0]

# Tool centre point in the `gripper` link frame: between the finger pads, 13 mm short of
# the pad tips. The gripper's approach axis is +Z of that frame.
TCP_OFFSET = (0.0, 0.0, 0.134)

# 240 Hz physics keeps finger/object contacts stable; policies act at the real rig's 10 Hz.
PHYSICS_DT = 1 / 240
CONTROL_HZ = 10.0


def decimation(control_hz: float) -> int:
    """Physics steps per control step at `control_hz`."""
    return max(1, round(1 / (control_hz * PHYSICS_DT)))


TABLE_HEIGHT = 0.75
TABLE_SIZE = (0.8, 1.2, 0.04)
TABLE_CENTER_X = 0.35
# Arms sit 0.6 m apart on the near edge of the table, both reaching along +X across it.
ARM_SPACING = 0.6
CUBE_SIZE = 0.04
CUBE_POS = (0.35, 0.12, TABLE_HEIGHT + CUBE_SIZE / 2)
PLATE_RADIUS = 0.09
PLATE_HEIGHT = 0.015
PLATE_POS = (0.38, -0.12, TABLE_HEIGHT + PLATE_HEIGHT / 2)
# Episodes start with every object shifted by up to this much in x and y.
OBJECT_JITTER = 0.03
# An object that falls this far below the table top has left the table: the episode fails.
DROP_MARGIN = 0.1

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
TOP_CAM_ROT = (0.0, 0.3827, 0.0, 0.9239)
# Wrist camera in the `gripper` link frame (ROS optical convention: +Z forward, +Y down).
# It sits 8 cm off the -Y face (the side toward the arm base when the gripper points down)
# and looks along the approach axis, tilted 8 deg toward the fingers, with image right along
# the finger travel. The open fingertips land in the bottom corners, as in robocurve's real
# wrist frames.
WRIST_CAM_POS = (0.0, -0.08, 0.0)
WRIST_CAM_ROT = (-0.0698, 0.0, 0.0, 0.9976)

# Observation keys in the Arena environment: the packed joint state, and one image per camera.
STATE_OBS = ("policy", "joint_pos")
CAMERA_OBS_GROUP = "camera_obs"

DOCS_RIG = """

Simulated rig (Isaac Lab Arena): both arms are mounted on the near edge of a table 0.6 m apart,
left arm at +y, both facing +x across it; each arm's base frame is its own mount. On the
table: a red 4 cm cube in front of the left arm and a white plate in front of the right.
Cameras: top_cam looks down over the arms from behind; left_cam and right_cam are on the
wrists, looking out along the fingers."""


def urdf_arm_limits() -> tuple[np.ndarray, np.ndarray]:
    """Joint limits of joint1..joint6 straight from the URDF."""
    limits = {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in ET.parse(YAM_URDF).getroot().iter("joint")
        if j.find("limit") is not None
    }
    lower, upper = zip(*(limits[name] for name in ARM_JOINTS))
    return np.array(lower), np.array(upper)


def action_limits() -> tuple[np.ndarray, np.ndarray]:
    """Bounds of the 14-D action: both arms' joint limits, grippers 0 (closed) to 1 (open)."""
    from inspect_robots_yam.packing import pack

    lower, upper = urdf_arm_limits()
    per_arm_low, per_arm_high = np.append(lower, 0.0), np.append(upper, 1.0)
    return pack(per_arm_low, per_arm_low), pack(per_arm_high, per_arm_high)


def embodiment_info(cam_height: int, cam_width: int, control_hz: float = CONTROL_HZ) -> Any:
    """The ``yam_arena`` EmbodimentInfo: the real rig's observation and action spaces."""
    from inspect_robots.embodiment import RESETTABLE, SEEDABLE, EmbodimentInfo
    from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
    from inspect_robots_yam.embodiment import _DOCS_JOINTS  # the real rig's joint guide; holds for the sim too

    low, high = action_limits()
    return EmbodimentInfo(
        name="yam_arena",
        action_space=action_box(low, high),
        observation_space=observation_space(cam_height, cam_width, DEFAULT_CAMERAS),
        control_hz=control_hz,
        is_simulated=True,
        capabilities=frozenset({SEEDABLE, RESETTABLE}),
        docs=_DOCS_JOINTS + DOCS_RIG,
        environment_id="isaaclab-arena-yam-tabletop",
    )


def observation_from_arena(
    obs: dict[str, Any],
    *,
    env_id: int = 0,
    cam_height: int,
    cam_width: int,
    instruction: str | None,
    now: float,
) -> Any:
    """One env's Arena observation as an Inspect Robots Observation in the ``yam_arms`` contract.

    Images are resized from the simulated sensors' 640x360 to ``cam_height`` x ``cam_width``.
    """
    import cv2
    from inspect_robots.types import Observation
    from inspect_robots_yam.config import DEFAULT_CAMERAS
    from inspect_robots_yam.packing import STATE_KEY

    images = {}
    for name in DEFAULT_CAMERAS:
        rgb = obs[CAMERA_OBS_GROUP][f"{name}_rgb"][env_id, ..., :3].cpu().numpy()
        images[name] = np.ascontiguousarray(cv2.resize(rgb, (cam_width, cam_height)), dtype=np.uint8)
    group, key = STATE_OBS
    state = obs[group][key][env_id].cpu().numpy().astype(np.float64)
    return Observation(
        images=images,
        state={STATE_KEY: state},
        instruction=instruction,
        image_times=dict.fromkeys(images, now),
        state_time=now,
    )
