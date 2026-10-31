"""Headless physics check for the YAM tabletop in Isaac Lab Arena.

Runs one pass of the scripted contact demo (`yam_arena.arena.policies.YamDemoPolicy`) through
the `yam_tabletop` environment and reports whether the physics behaved: joint limits and
velocities stayed within the real motors' range, the left arm actually lifted the cube
(fingers collide with it), and the right arm's fingertips stopped on the plate instead of
passing through it.

    python scripts/diagnose_physics.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
app = AppLauncher(args).app

import sys  # noqa: E402

import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg  # noqa: E402

from yam_arena.arena.environment import YamTabletopEnvironment, YamTabletopEnvironmentCfg  # noqa: E402
from yam_arena.arena.policies import DEMO_DURATION, YamDemoPolicy, YamDemoPolicyCfg  # noqa: E402
from yam_arena.rig import CUBE_POS, PLATE_HEIGHT, TABLE_HEIGHT  # noqa: E402


def main() -> None:
    arena_env = YamTabletopEnvironment().build(YamTabletopEnvironmentCfg(object_jitter=0.0, episode_length_s=1e6))
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, env_spacing=3.0, device=args.device)).make_registered()
    base = env.unwrapped
    obs, _ = env.reset()
    policy = YamDemoPolicy(YamDemoPolicyCfg())
    policy.reset()

    left, right, cube = base.scene["left_arm"], base.scene["right_arm"], base.scene["cube"]
    names = left.joint_names
    hard = left.data.joint_pos_limits.torch[0]
    vel_limit = left.data.joint_vel_limits.torch[0]
    over = torch.zeros(len(names), device=base.device)
    peak_vel = torch.zeros(len(names), device=base.device)
    lowest_tip = 9.0
    cube_peak = 0.0
    right_ee = right.find_bodies("gripper")[0][0]
    # The pad tips sit 13 mm past the TCP, 147 mm along the gripper's approach axis.
    tip_offset = torch.tensor([[0.0, 0.0, 0.147]], device=base.device)
    for step in range(round((DEMO_DURATION + 1.0) / base.step_dt)):
        obs, *_ = env.step(policy.get_action(env, obs))
        for arm in (left, right):
            q = arm.data.joint_pos.torch[0]
            if not torch.isfinite(q).all():
                print(f"[diag] FAIL: NaN joint state at step {step}")
                return
            over = torch.maximum(over, torch.maximum(hard[:, 0] - q, q - hard[:, 1]))
            peak_vel = torch.maximum(peak_vel, arm.data.joint_vel.torch[0].abs())
        quat = right.data.body_quat_w.torch[:, right_ee]
        tip = right.data.body_pos_w.torch[:, right_ee] + quat_apply(quat, tip_offset)
        lowest_tip = min(lowest_tip, float(tip[0, 2]))
        cube_peak = max(cube_peak, float(cube.data.root_pos_w.torch[0, 2]))

    # Sampled once per 10 Hz control step, so short spikes between samples go unseen.
    print("[diag] joint      worst limit overshoot   peak |vel| / sim limit")
    for i, n in enumerate(names):
        print(f"[diag] {n:8s}   {max(0.0, float(over[i])):8.4f}            {peak_vel[i]:6.3f} / {vel_limit[i]:.3f}")
    lift = cube_peak - CUBE_POS[2]
    print(f"[diag] cube lifted {lift * 100:.1f} cm  -> {'PASS' if lift > 0.1 else 'FAIL'} (fingers grip the cube)")
    plate_top = TABLE_HEIGHT + PLATE_HEIGHT
    sink = plate_top - lowest_tip
    print(
        f"[diag] right fingertips lowest z={lowest_tip:.4f}, plate top {plate_top:.4f}, commanded 10 mm below it"
        f"  -> {'PASS' if sink < 0.004 else 'FAIL'} (penetration {max(0.0, sink) * 1000:.1f} mm)"
    )
    cube_final = (cube.data.root_pos_w.torch[0] - base.scene.env_origins[0]).tolist()
    print(f"[diag] cube final position {[round(v, 3) for v in cube_final]}")
    sys.stdout.flush()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
