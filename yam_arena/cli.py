"""`yam-arena`: the bimanual YAM tabletop in Isaac Lab Arena.

    yam-arena view [SCENE.usda]                # run the scene in the GUI; the arms hold their ready pose
    yam-arena edit SCENE.usda                  # edit the environment in Isaac Sim, then save it to SCENE.usda
    yam-arena eval EXPERIMENT.yaml [ARGS...]   # run an Arena Experiment (Arena's experiment runner)
    yam-arena rollout [ARGS...]                # run one policy in one environment (Arena's policy runner)

`eval` and `rollout` are Arena's own runners with the `yam_tabletop` environment registered;
everything after the command goes to them unchanged. For example:

    yam-arena eval experiments/yam_tabletop.yaml
    yam-arena rollout --viz kit --num_episodes 1 \\
        --policy_type yam_arena.arena.policies.YamDemoPolicy yam_tabletop

`edit` opens SCENE.usda if it exists (else the built-in scene) with physics stopped and adds
a "YAM scene" panel whose Save button writes the environment (everything but the arms and
cameras) back to the file. Edit with Isaac Sim's own tools: drag prims with the gizmo, add
shapes from Create or assets from the content browser, change lights and materials in the
Property panel. Press Play to let objects settle under physics and Save (playing or paused)
to keep them where they came to rest; Stop rewinds the scene to where it was when you
pressed Play. Edits made while paused reach physics only after Stop and Play. The arms
stand frozen in their ready pose for reference: they collide but do not move.

Saved files load in `yam-arena view SCENE.usda`, in the `yam_tabletop` environment
(`--scene SCENE.usda`) and in the `yam_arena` embodiment (`-E scene=SCENE.usda`). The GUI needs
a display: DISPLAY, or the instance's noVNC desktop on :1 when DISPLAY is unset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EYE, TARGET = (-0.9, 0.9, 1.5), (0.3, 0.0, 0.8)
SETTLE_S = 0.5


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(prog="yam-arena", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="run the scene in the GUI with the arms holding still")
    view.add_argument("scene", nargs="?", help="scene file saved by `yam-arena edit` (default: the built-in scene)")
    edit = commands.add_parser("edit", help="edit the environment and save it to a scene file")
    edit.add_argument("scene", help="scene file to save to; opened first if it already exists")
    for sub in (view, edit):
        AppLauncher.add_app_launcher_args(sub)
    commands.add_parser("eval", help="run an Arena Experiment YAML", add_help=False)
    commands.add_parser("rollout", help="run Arena's policy runner", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `yam-arena` console script."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("eval", "rollout"):
        from yam_arena.arena import runners

        if argv[0] == "eval":
            if len(argv) < 2 or argv[1].startswith("-"):
                sys.exit("usage: yam-arena eval EXPERIMENT.yaml [ARGS...]")
            runners.run_experiment(argv[1], argv[2:])
        else:
            runners.run_policy(argv[1:])
        return

    # Both must be set before Isaac Sim is imported: Kit asks about the EULA on stdin at import.
    if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X1").exists():
        os.environ["DISPLAY"] = ":1"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    args = _parser().parse_args(argv)
    args.enable_cameras = True  # the policy cameras are sensors; this lets them render
    if args.visualizer is None:
        args.visualizer = ["kit"]
    app = AppLauncher(args).app
    try:
        if args.command == "view":
            _view(app, args)
        else:
            _edit(app, args, Path(args.scene).resolve())
    finally:
        app.close()


def _make_env(args: argparse.Namespace, scene: str | None, **builder_kwargs):
    """The `yam_tabletop` Arena environment for one env, with the arms in their ready pose and objects at rest."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg

    from yam_arena.arena.environment import YamTabletopEnvironment, YamTabletopEnvironmentCfg
    from yam_arena.embodiment import READY_ACTION

    arena_env = YamTabletopEnvironment().build(
        YamTabletopEnvironmentCfg(enable_cameras=True, scene=scene, object_jitter=0.0, episode_length_s=1e6)
    )
    cfg = ArenaEnvBuilderCfg(num_envs=1, env_spacing=3.0, **{"device": args.device, **builder_kwargs})
    env = ArenaEnvBuilder(arena_env, cfg).make_registered()
    env.reset()
    hold = _hold_action(env, READY_ACTION)
    for _ in range(round(SETTLE_S / env.unwrapped.step_dt)):
        env.step(hold)
    env.unwrapped.sim.set_camera_view(eye=EYE, target=TARGET)
    return env, hold


def _hold_action(env, action):
    import torch

    return torch.tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)


def _view(app, args: argparse.Namespace) -> None:
    env, hold = _make_env(args, args.scene)
    print(f"[yam-arena] viewing {args.scene or 'the built-in scene'}; close the window to quit", flush=True)
    while app.is_running():
        env.step(hold)
    env.close()


def _hand_over_to_isaac_sim(env) -> None:
    """Hand the stage over to Isaac Sim: drop Isaac Lab's physics callbacks, freeze the arms, stop.

    Editing a physics prim (dragging an object, adding one) makes PhysX re-parse the stage,
    which invalidates the views Isaac Lab drives the arms and objects through; using them
    after that writes garbage poses. So once the arms are in the ready pose the editor stops
    using Isaac Lab's environment and only plays, pauses and stops the physics, with every arm
    link made kinematic at the pose PhysX wrote back to USD, so the arms hold still on their own.

    The simulation is left stopped rather than paused: Play from Stop parses the stage afresh,
    so physics sees every edit (resuming from a pause would keep simulating the old shapes).
    """
    import carb
    from pxr import PhysxSchema, Usd, UsdPhysics

    from yam_arena.arena.scene_file import ENV_NS

    scene = env.unwrapped.scene
    for entity in (*scene.articulations.values(), *scene.rigid_objects.values(), *scene.sensors.values()):
        entity._clear_callbacks()  # their timeline callbacks would rebuild the views on Play
    for arm in ("LeftArm", "RightArm"):
        for prim in Usd.PrimRange(scene.stage.GetPrimAtPath(f"{ENV_NS}/{arm}")):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateArticulationEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            if prim.IsA(UsdPhysics.Joint):  # PhysX rejects joints between kinematic links, noisily
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    # This first Stop keeps the settled poses; later ones rewind to where Play was pressed.
    sim = env.unwrapped.sim
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/resetOnStop", False)
    sim.stop()
    sim.render()
    settings.set_bool("/physics/resetOnStop", True)


def _edit(app, args: argparse.Namespace, scene_file: Path) -> None:
    import carb
    import omni.ui as ui
    import omni.usd

    from yam_arena.arena.scene_file import save_environment

    # PhysX writes body poses back to USD only on the CPU pipeline without Fabric, with
    # updateToUsd on (the Isaac Lab app files turn it off). That way objects settled with Play
    # are saved where they came to rest.
    args.device = "cpu"
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    env, _ = _make_env(args, str(scene_file) if scene_file.exists() else None, disable_fabric=True)
    _hand_over_to_isaac_sim(env)
    sim = env.unwrapped.sim

    # Isaac Lab drives the simulation itself rather than through Kit's toolbar, so the panel
    # has its own Play, Pause and Stop.
    window = ui.Window("YAM scene", width=460, height=190)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(f"Saves everything except the arms and cameras to\n{scene_file}", word_wrap=True, height=0)
            status = ui.Label(
                "Play lets objects settle; Save keeps them where they are. Stop rewinds to before Play.",
                word_wrap=True,
                height=0,
            )

            def save() -> None:
                count = save_environment(omni.usd.get_context().get_stage(), scene_file)
                status.text = f"Saved {count} top-level prims to {scene_file.name}."
                print(f"[yam-arena] saved {count} prims to {scene_file}", flush=True)

            with ui.HStack(spacing=8, height=32):
                ui.Button("Play", clicked_fn=sim.play)
                ui.Button("Pause", clicked_fn=sim.pause)
                ui.Button("Stop", clicked_fn=sim.stop)
            ui.Button("Save scene", clicked_fn=save, height=32)
    print(f"[yam-arena] editing {scene_file}; use the 'YAM scene' panel to save", flush=True)
    while app.is_running():
        if sim.is_playing():
            sim.step(render=True)
        else:
            sim.render()
