"""Spawn the bimanual YAM scene and loop the contact demo: the left arm picks and places
the cube, the right arm presses the plate. Link contact forces are printed as they happen.

    python scripts/view_scene.py                         # GUI (needs a display, e.g. the noVNC desktop)
    python scripts/view_scene.py --headless --enable_cameras --screenshots out/
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--screenshots", type=str, default=None, help="Directory to save overview, top and wrist camera PNGs into.")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many sim steps (0 = run until closed).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.screenshots:
    args.enable_cameras = True
app = AppLauncher(args).app

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from yam_sim.demo import make_drivers, reset_scene  # noqa: E402
from yam_sim.scene import PHYSICS_DT, RENDER_INTERVAL, BimanualYamSceneCfg, update_wrist_cams  # noqa: E402

CAMERAS = ("overview_cam", "top_cam", "left_cam", "right_cam")


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=RENDER_INTERVAL, device=args.device)
    )
    sim.set_camera_view(eye=(-0.9, 0.9, 1.5), target=(0.3, 0.0, 0.8))
    scene_cfg = BimanualYamSceneCfg(num_envs=1, env_spacing=3.0)
    if not args.enable_cameras:
        for name in CAMERAS:
            setattr(scene_cfg, name, None)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[yam_sim] joints:", scene["left_arm"].joint_names)

    out_dir = Path(args.screenshots) if args.screenshots else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    drivers = make_drivers(scene)
    episode = max(d.duration for d in drivers) + 2.0
    step, t = 0, 0.0
    while app.is_running():
        if t > episode:
            reset_scene(scene)
            drivers, t = make_drivers(scene), 0.0
        for driver in drivers:
            driver.step(t)
        scene.write_data_to_sim()
        update_wrist_cams(scene)
        sim.step(render=(step + 1) % RENDER_INTERVAL == 0)
        scene.update(PHYSICS_DT)
        step += 1
        t += PHYSICS_DT
        if step % 60 == 0:
            for side in ("left", "right"):
                sensor = scene[f"{side}_contacts"]
                forces = sensor.data.net_forces_w[0].norm(dim=-1)
                touching = {n: round(float(f), 1) for n, f in zip(sensor.body_names, forces) if f > 0.05}
                if touching:
                    print(f"[yam_sim] t={t:5.2f}s {side} arm contact N: {touching}")
        if out_dir and step % 120 == 0:
            import imageio.v3 as iio

            for name in CAMERAS:
                rgb = scene[name].data.output["rgb"][0].cpu().numpy()
                path = out_dir / f"{name}_{step:05d}.png"
                iio.imwrite(path, rgb)
            print(f"[yam_sim] saved {', '.join(CAMERAS)} frames for step {step}")
        if args.steps and step >= args.steps:
            break
    sys.stdout.flush()
    app.close()


if __name__ == "__main__":
    main()
