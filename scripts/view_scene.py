"""Run the `yam_tabletop` Arena environment and loop the contact demo: the left arm picks and
places the cube, the right arm presses the plate. Link contact forces are printed as they happen.

    python scripts/view_scene.py                                  # GUI (needs a display, e.g. the noVNC desktop)
    python scripts/view_scene.py --viz none --screenshots out/ --steps 120   # headless
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--screenshots", type=str, default=None, help="Directory to save overview, top and wrist camera PNGs into.")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many control steps (0 = run until closed).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# The policy cameras only render (and cost GPU time) when frames are saved. AppLauncher
# consumes args.enable_cameras, so the environment reads this copy.
CAMERAS_ON = args.enable_cameras = bool(args.screenshots)
if args.visualizer is None:  # open the Kit viewer unless --viz says otherwise (--viz none: headless)
    args.visualizer = ["kit"]
if "kit" in args.visualizer and not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X1").exists():
    os.environ["DISPLAY"] = ":1"  # the launchable's noVNC desktop
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
app = AppLauncher(args).app

import sys  # noqa: E402

import cv2  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg  # noqa: E402

from yam_arena.arena.environment import YamTabletopEnvironment, YamTabletopEnvironmentCfg  # noqa: E402
from yam_arena.arena.policies import DEMO_DURATION, YamDemoPolicy, YamDemoPolicyCfg  # noqa: E402

POLICY_CAMERAS = ("top_cam", "left_cam", "right_cam")
EYE, TARGET = (-0.9, 0.9, 1.5), (0.3, 0.0, 0.8)


def add_debug_sensors(env_cfg, overview: bool):
    """Net contact force on every arm link (`debug_vis` marks links in contact), and an overview camera."""
    env_cfg.scene.left_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/LeftArm/.*", debug_vis=True)
    env_cfg.scene.right_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/RightArm/.*", debug_vis=True)
    if overview:
        env_cfg.scene.overview_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/OverviewCam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
            # Behind-and-above the arms, looking along +X and pitched 30 degrees down.
            offset=CameraCfg.OffsetCfg(pos=(-1.0, 0.0, 1.65), rot=(0.0, 0.2588, 0.0, 0.9659), convention="world"),
        )
    return env_cfg


def main() -> None:
    arena_env = YamTabletopEnvironment().build(
        YamTabletopEnvironmentCfg(enable_cameras=CAMERAS_ON, object_jitter=0.0, episode_length_s=DEMO_DURATION + 2.0)
    )
    rig_timing = arena_env.env_cfg_callback
    arena_env.env_cfg_callback = lambda env_cfg: add_debug_sensors(rig_timing(env_cfg), overview=bool(args.screenshots))
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, env_spacing=3.0, device=args.device)).make_registered()
    base = env.unwrapped
    base.sim.set_camera_view(eye=EYE, target=TARGET)
    obs, _ = env.reset()
    policy = YamDemoPolicy(YamDemoPolicyCfg())
    policy.reset()
    print("[yam-arena] joints:", base.scene["left_arm"].joint_names)

    out_dir = Path(args.screenshots) if args.screenshots else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    print_every = max(1, round(0.25 / base.step_dt))
    shot_every = max(1, round(0.5 / base.step_dt))
    step = 0
    while app.is_running():
        obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
        step += 1
        if bool(terminated[0] or truncated[0]):  # the env has reset itself; restart the script
            policy.reset()
        t = float(base.episode_length_buf[0]) * base.step_dt
        if step % print_every == 0:
            for side in ("left", "right"):
                sensor = base.scene[f"{side}_contacts"]
                forces = sensor.data.net_forces_w.torch[0].norm(dim=-1)
                touching = {n: round(float(f), 1) for n, f in zip(sensor.body_names, forces) if f > 0.05}
                if touching:
                    print(f"[yam-arena] t={t:5.2f}s {side} arm contact N: {touching}")
        if out_dir and step % shot_every == 0:
            frames = {name: obs["camera_obs"][f"{name}_rgb"][0, ..., :3] for name in POLICY_CAMERAS}
            frames["overview_cam"] = base.scene["overview_cam"].data.output["rgb"].torch[0, ..., :3]
            for name, rgb in frames.items():
                cv2.imwrite(str(out_dir / f"{name}_{step:05d}.png"), cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR))
            print(f"[yam-arena] saved {', '.join(frames)} frames for step {step}")
        if args.steps and step >= args.steps:
            break
    sys.stdout.flush()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
