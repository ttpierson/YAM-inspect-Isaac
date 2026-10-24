"""Headless physics check for the bimanual YAM scene.

Runs one pass of the contact demo and reports whether the physics behaved:
joint limits and velocities stayed within the real motors' range, the left arm
actually lifted the cube (fingers collide with it), and the right arm's fingertips
stopped on the plate instead of passing through it.

    python scripts/diagnose_physics.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from yam_sim.demo import make_drivers  # noqa: E402
from yam_sim.scene import CUBE_POS, PHYSICS_DT, PLATE_HEIGHT, TABLE_HEIGHT, BimanualYamSceneCfg  # noqa: E402


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    cfg = BimanualYamSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.overview_cam = None
    scene = InteractiveScene(cfg)
    sim.reset()

    stage = omni.usd.get_context().get_stage()
    colliders = [
        str(p.GetPath()).split("/LeftArm/")[1].split("/")[0]
        for p in stage.Traverse(Usd.TraverseInstanceProxies())
        if "/LeftArm/" in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    print(f"[diag] links with colliders: {colliders}")

    drivers = make_drivers(scene)
    left, right = scene["left_arm"], scene["right_arm"]
    names = left.joint_names
    hard = left.data.joint_pos_limits[0]
    vel_limit = left.data.joint_vel_limits[0]
    over = torch.zeros(len(names), device=sim.device)
    peak_vel = torch.zeros(len(names), device=sim.device)
    lowest_tip = 9.0
    cube_peak = 0.0
    touched: dict[str, float] = {}
    steps = int((max(d.duration for d in drivers) + 1.0) / PHYSICS_DT)
    for step in range(steps):
        t = step * PHYSICS_DT
        for driver in drivers:
            driver.step(t)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(PHYSICS_DT)
        for arm in (left, right):
            q = arm.data.joint_pos[0]
            if not torch.isfinite(q).all():
                print(f"[diag] FAIL: NaN joint state at step {step}")
                return
            over = torch.maximum(over, torch.maximum(hard[:, 0] - q, q - hard[:, 1]))
            peak_vel = torch.maximum(peak_vel, arm.data.joint_vel[0].abs())
        # The pad tips sit 13 mm past the TCP along the approach axis.
        tcp_pos, tcp_quat = drivers[1].tcp_pose()
        tip = tcp_pos + quat_apply(tcp_quat, torch.tensor([[0.0, 0.0, 0.013]], device=sim.device))
        lowest_tip = min(lowest_tip, float(tip[0, 2]))
        cube_peak = max(cube_peak, float(scene["cube"].data.root_pos_w[0, 2]))
        for side in ("left", "right"):
            sensor = scene[f"{side}_contacts"]
            for name, f in zip(sensor.body_names, sensor.data.net_forces_w[0].norm(dim=-1)):
                if f > 0.05:
                    key = f"{side}/{name}"
                    touched[key] = max(touched.get(key, 0.0), float(f))

    print("[diag] joint      worst limit overshoot   peak |vel| / sim limit")
    for i, n in enumerate(names):
        print(f"[diag] {n:8s}   {max(0.0, float(over[i])):8.4f}            {peak_vel[i]:6.3f} / {vel_limit[i]:.3f}")
    print("[diag] peak contact force per link (N):", {k: round(v, 1) for k, v in sorted(touched.items())})
    lift = cube_peak - CUBE_POS[2]
    print(f"[diag] cube lifted {lift * 100:.1f} cm  -> {'PASS' if lift > 0.1 else 'FAIL'} (fingers grip the cube)")
    plate_top = TABLE_HEIGHT + PLATE_HEIGHT
    sink = plate_top - lowest_tip
    print(
        f"[diag] right fingertips lowest z={lowest_tip:.4f}, plate top {plate_top:.4f}, commanded 10 mm below it"
        f"  -> {'PASS' if sink < 0.004 else 'FAIL'} (penetration {max(0.0, sink) * 1000:.1f} mm)"
    )
    cube_final = scene["cube"].data.root_pos_w[0].tolist()
    print(f"[diag] cube final position {[round(v, 3) for v in cube_final]}")
    sys.stdout.flush()
    app.close()


if __name__ == "__main__":
    main()
