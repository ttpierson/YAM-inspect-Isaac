"""Inspect Robots embodiment ``yam_sim``: the bimanual YAM tabletop scene in Isaac Lab.

It speaks the same contract as the real rig's ``yam_arms`` embodiment (from
`inspect-robots-yam`), so any policy built for the real arms (``molmoact2``,
``gr00t``, the ``agent`` LLM policy) runs against the simulator unchanged:

- observation images ``top_cam`` (the overhead D435) and ``left_cam`` / ``right_cam``
  (the D405 on each wrist), RGB uint8 resized to ``cam_height`` x ``cam_width``;
- state ``joint_pos``: 14-D ``[left j0..j5, left gripper, right j0..j5, right gripper]``,
  joints in rad, grippers normalized (0 closed, 1 open);
- action: the same 14-D layout as absolute joint targets.

Each step holds the target for one control period (``control_hz``, 10 Hz like the real
rig) of joint PD plus gravity compensation, the way i2rt drives the real arms.

``-E scene=SCENE.usda`` runs in a scene saved by ``yam-sim edit`` instead of the built-in
one; every free rigid body in it is reset (and jittered, when seeded) like the cube is.

Importing this module is cheap: Isaac Sim only starts on the first ``reset()``.
"""

from __future__ import annotations

import atexit
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from inspect_robots import Scene
from inspect_robots.embodiment import RESETTABLE, SEEDABLE, EmbodimentBase, EmbodimentInfo
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
from inspect_robots_yam.embodiment import _DOCS_JOINTS  # the real rig's joint guide; holds for the sim too
from inspect_robots_yam.packing import ARM_DOF, STATE_KEY, pack, split

from yam_sim import YAM_URDF

# Mirrors yam_sim.assets / yam_sim.scene, which import Isaac Lab and so cannot be imported
# before the app starts.
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
FINGER_OPEN = -0.04695
ARMS = ("left_arm", "right_arm")
# Settle time after a reset, so the first observation shows the arms at rest.
SETTLE_S = 0.5
# Frames rendered after a reset before the first observation: the RTX renderer needs a few
# to converge once the wrist cameras jump to their mounts, or the first frame is grainy.
RESET_RENDERS = 4
# Seeded resets shift every object (the cube and plate by default) by up to this much in x and y.
OBJECT_JITTER = 0.03

DOCS = (
    _DOCS_JOINTS
    + """

Simulated rig (Isaac Lab): both arms are mounted on the near edge of a table 0.6 m apart,
left arm at +y, both facing +x across it; each arm's base frame is its own mount. On the
table: a red 4 cm cube in front of the left arm and a white plate in front of the right.
Cameras: top_cam looks down over the arms from behind; left_cam and right_cam are on the
wrists, looking out along the fingers."""
)


def _urdf_arm_limits() -> tuple[np.ndarray, np.ndarray]:
    """Joint limits of joint1..joint6 straight from the URDF, so `info` needs no Isaac."""
    limits = {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in ET.parse(YAM_URDF).getroot().iter("joint")
        if j.find("limit") is not None
    }
    lower, upper = zip(*(limits[name] for name in ARM_JOINTS))
    return np.array(lower), np.array(upper)


class YamSimEmbodiment(EmbodimentBase):
    """Bimanual YAM in Isaac Lab, observed through the rig's overhead and wrist cameras."""

    def __init__(
        self,
        *,
        headless: bool = True,
        device: str = "cuda:0",
        cam_height: int = 224,
        cam_width: int = 224,
        control_hz: float = 10.0,
        scene: str | None = None,
    ) -> None:
        self.headless = headless
        self.scene_file = scene
        self.device = device
        self.cam_height, self.cam_width = cam_height, cam_width
        lower, upper = _urdf_arm_limits()
        per_arm_low, per_arm_high = np.append(lower, 0.0), np.append(upper, 1.0)
        self._low, self._high = pack(per_arm_low, per_arm_low), pack(per_arm_high, per_arm_high)
        self.info = EmbodimentInfo(
            name="yam_sim",
            action_space=action_box(self._low, self._high),
            observation_space=observation_space(cam_height, cam_width, DEFAULT_CAMERAS),
            control_hz=control_hz,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE, RESETTABLE}),
            docs=DOCS,
            environment_id="isaaclab-yam-bimanual-tabletop",
        )
        self._app: Any = None
        self._sim: Any = None
        self._scene: Any = None
        self._instruction: str | None = None

    # -- Isaac lifecycle ---------------------------------------------------------------

    def _ensure_sim(self) -> None:
        if self._scene is not None:
            return
        from isaaclab.app import AppLauncher

        self._app = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        from yam_sim.scene import PHYSICS_DT, scene_cfg

        self._dt = PHYSICS_DT
        self._substeps = max(1, round(1.0 / (self.info.control_hz * PHYSICS_DT)))
        self._sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=self.device))
        self._sim.set_camera_view(eye=(-0.9, 0.9, 1.5), target=(0.3, 0.0, 0.8))
        self._scene = InteractiveScene(scene_cfg(self.scene_file))
        self._sim.reset()
        self._ids = {
            name: (
                self._scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                self._scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0],
            )
            for name in ARMS
        }

    def close(self) -> None:
        """Drop the scene; the app closes at interpreter exit.

        ``SimulationApp.close()`` ends the process, which would swallow whatever the caller
        prints afterwards (the inspect-robots CLI prints its results after ``close()``).
        """
        if self._app is not None:
            atexit.register(self._app.close)
        self._app = self._sim = self._scene = None

    # -- Embodiment contract -----------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Return both arms to the ready pose, put every object back (jittered if seeded), and settle."""
        self._ensure_sim()
        import torch

        from yam_sim.demo import reset_scene

        self._instruction = scene.instruction
        reset_scene(self._scene)
        if seed is not None:
            rng = np.random.default_rng(seed)
            for obj in self._scene.rigid_objects.values():
                state = obj.data.default_root_state.clone()
                state[:, :3] += self._scene.env_origins
                jitter = rng.uniform(-OBJECT_JITTER, OBJECT_JITTER, size=2)
                state[:, :2] += torch.tensor(jitter, dtype=state.dtype, device=state.device)
                obj.write_root_state_to_sim(state)
        for _ in range(round(SETTLE_S / self._dt)):
            self._physics_step()
        return self._render_observation(renders=RESET_RENDERS)

    def step(self, action: Action) -> StepResult:
        """Hold the 14-D absolute joint target for one control period."""
        import torch

        target = np.clip(np.asarray(action.data, dtype=np.float64).reshape(-1), self._low, self._high)
        for name, arm_target in zip(ARMS, split(target)):
            arm = self._scene[name]
            arm_ids, finger_ids = self._ids[name]
            joints = torch.tensor(arm_target[:ARM_DOF], dtype=torch.float32, device=arm.device).unsqueeze(0)
            finger = float(arm_target[ARM_DOF]) * FINGER_OPEN
            arm.set_joint_position_target(joints, joint_ids=arm_ids)
            arm.set_joint_position_target(torch.full((1, 2), finger, device=arm.device), joint_ids=finger_ids)
        for _ in range(self._substeps):
            self._physics_step()
        return StepResult(observation=self._render_observation())

    # -- internals ---------------------------------------------------------------------

    def _physics_step(self) -> None:
        from yam_sim.scene import hold_arms

        hold_arms(self._scene)
        self._scene.write_data_to_sim()
        self._sim.step(render=False)
        self._scene.update(self._dt)

    def _render_observation(self, renders: int = 1) -> Observation:
        import cv2

        from yam_sim.scene import update_wrist_cams

        update_wrist_cams(self._scene)
        for _ in range(renders):
            self._sim.render()
        self._scene.update(0.0)

        now = float(self._sim.current_time)
        images = {}
        for name in DEFAULT_CAMERAS:
            rgb = self._scene[name].data.output["rgb"][0, ..., :3].cpu().numpy()
            images[name] = np.ascontiguousarray(cv2.resize(rgb, (self.cam_width, self.cam_height)), dtype=np.uint8)
        per_arm = []
        for name in ARMS:
            arm = self._scene[name]
            arm_ids, finger_ids = self._ids[name]
            q = arm.data.joint_pos[0]
            joints = q[arm_ids].cpu().numpy().astype(np.float64)
            # Both fingers mirror one rack; report their mean opening as the 0..1 gripper.
            grip = float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0))
            per_arm.append(np.append(joints, grip))
        return Observation(
            images=images,
            state={STATE_KEY: pack(*per_arm)},
            instruction=self._instruction,
            image_times=dict.fromkeys(images, now),
            state_time=now,
        )


def yam_sim_embodiment(**kwargs: Any) -> YamSimEmbodiment:
    """Embodiment ``yam_sim``; ``-E headless=false`` renders on DISPLAY (e.g. the noVNC desktop)."""
    return YamSimEmbodiment(**kwargs)
