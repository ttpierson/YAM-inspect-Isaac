"""Spawn the bimanual YAM scene and move both arms through a demo motion.

    python scripts/view_scene.py                         # GUI (needs a display, e.g. the noVNC desktop)
    python scripts/view_scene.py --headless --enable_cameras --screenshots out/
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--screenshots", type=str, default=None, help="Directory to save overview camera PNGs into.")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many sim steps (0 = run until closed).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.screenshots:
    args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from yam_sim.assets import FINGER_OPEN  # noqa: E402
from yam_sim.scene import BimanualYamSceneCfg  # noqa: E402

# A raised "ready" pose: shoulder up, elbow bent, wrist pitched down toward the table.
READY = [0.0, 1.0, 1.0, -0.6, 0.0, 0.0]


def demo_targets(t: float, mirror: float) -> list[float]:
    """Ready pose plus a slow wave; `mirror` flips the base yaw for the other arm."""
    swing = 0.35 * math.sin(0.8 * t)
    pose = READY.copy()
    pose[0] += mirror * swing
    pose[1] += 0.25 * math.sin(0.5 * t)
    pose[5] = 0.8 * math.sin(1.1 * t)
    grip = FINGER_OPEN * (0.5 + 0.5 * math.sin(1.5 * t))
    return pose + [grip, grip]


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    sim.set_camera_view(eye=(-0.9, 0.9, 1.5), target=(0.3, 0.0, 0.8))
    scene_cfg = BimanualYamSceneCfg(num_envs=1, env_spacing=3.0)
    if not args.enable_cameras:
        scene_cfg.overview_cam = None
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    arms = {"left": (scene["left_arm"], 1.0), "right": (scene["right_arm"], -1.0)}
    print("[yam_sim] joints:", scene["left_arm"].joint_names)

    out_dir = Path(args.screenshots) if args.screenshots else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    while app.is_running():
        t = step * sim.get_physics_dt()
        for arm, mirror in arms.values():
            target = torch.tensor([demo_targets(t, mirror)], device=sim.device)
            arm.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        step += 1
        if out_dir and step % 120 == 0:
            import imageio.v3 as iio

            rgb = scene["overview_cam"].data.output["rgb"][0].cpu().numpy()
            path = out_dir / f"overview_{step:05d}.png"
            iio.imwrite(path, rgb)
            print(f"[yam_sim] saved {path}")
        if args.steps and step >= args.steps:
            break
    app.close()


if __name__ == "__main__":
    main()
