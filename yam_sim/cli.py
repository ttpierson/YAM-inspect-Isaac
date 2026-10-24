"""`yam-sim`: open the bimanual YAM scene in the Isaac Sim GUI.

    yam-sim view [SCENE.usda]   # run the scene; the arms hold their ready pose
    yam-sim edit SCENE.usda     # edit the environment in Isaac Sim, then save it to SCENE.usda

`edit` opens SCENE.usda if it exists (else the built-in scene) with physics stopped and adds
a "YAM scene" panel whose Save button writes the environment (everything but the arms and
cameras) back to the file. Edit with Isaac Sim's own tools: drag prims with the gizmo, add
shapes from Create or assets from the content browser, change lights and materials in the
Property panel. Press Play to let objects settle under physics and Save (playing or paused)
to keep them where they came to rest; Stop rewinds the scene to where it was when you
pressed Play. Edits made while paused reach physics only after Stop and Play. The arms
stand frozen in their ready pose for reference: they collide but do not move.

Saved files load in `yam-sim view SCENE.usda` and in the `yam_sim` embodiment
(`-E scene=SCENE.usda`). The GUI needs a display: DISPLAY, or the instance's noVNC desktop
on :1 when DISPLAY is unset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

EYE, TARGET = (-0.9, 0.9, 1.5), (0.3, 0.0, 0.8)
SETTLE_S = 0.5


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(prog="yam-sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="run the scene with the arms holding still")
    view.add_argument("scene", nargs="?", help="scene file saved by `yam-sim edit` (default: the built-in scene)")
    edit = commands.add_parser("edit", help="edit the environment and save it to a scene file")
    edit.add_argument("scene", help="scene file to save to; opened first if it already exists")
    for sub in (view, edit):
        AppLauncher.add_app_launcher_args(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `yam-sim` console script."""
    # Both must be set before Isaac Sim is imported: Kit asks about the EULA on stdin at import.
    if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X1").exists():
        os.environ["DISPLAY"] = ":1"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    args = _parser().parse_args(argv)
    args.enable_cameras = True  # the policy cameras are sensors; this lets them render
    app = AppLauncher(args).app
    try:
        if args.command == "view":
            _view(app, args.scene)
        else:
            _edit(app, Path(args.scene).resolve())
    finally:
        app.close()


def _settle(sim, scene, seconds: float) -> None:
    """Put the arms in the ready pose and the objects at their spawn poses, then let them settle."""
    from yam_sim.demo import reset_scene
    from yam_sim.scene import PHYSICS_DT, hold_arms, update_wrist_cams

    reset_scene(scene)
    for _ in range(round(seconds / PHYSICS_DT)):
        hold_arms(scene)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(PHYSICS_DT)
    update_wrist_cams(scene)


def _run(app, sim, scene) -> None:
    """Step physics with the arms held, rendering at 60 Hz; only render while paused."""
    from yam_sim.scene import PHYSICS_DT, RENDER_INTERVAL, hold_arms, update_wrist_cams

    step = 0
    while app.is_running():
        if not sim.is_playing():
            sim.render()
            continue
        hold_arms(scene)
        scene.write_data_to_sim()
        render = (step + 1) % RENDER_INTERVAL == 0
        if render:
            update_wrist_cams(scene)
        sim.step(render=render)
        scene.update(PHYSICS_DT)
        step += 1


def _hand_over_to_isaac_sim(sim, scene) -> None:
    """Hand the stage over to Isaac Sim: drop Isaac Lab's physics views, freeze the arms, stop.

    Editing a physics prim (dragging an object, adding one) makes PhysX re-parse the stage,
    which invalidates the views Isaac Lab drives the arms and objects through; using them
    after that writes garbage poses. So once the arms are in the ready pose the editor stops
    using Isaac Lab and lets Isaac Sim's own Play/Pause/Stop run the physics, with every arm
    link made kinematic at the pose PhysX wrote back to USD, so the arms hold still on their own.

    The timeline is left stopped rather than paused: Play from Stop parses the stage afresh,
    so physics sees every edit (resuming from a pause would keep simulating the old shapes).
    """
    import carb
    from pxr import PhysxSchema, Usd, UsdPhysics

    from yam_sim.scene import ENV_NS

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
    # Isaac Lab's own Stop handler would block in a render loop until Play is pressed.
    if sim._app_control_on_stop_handle is not None:
        sim._app_control_on_stop_handle.unsubscribe()
        sim._app_control_on_stop_handle = None
    # This first Stop keeps the settled poses; later ones rewind to where Play was pressed.
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/resetOnStop", False)
    sim.stop()
    sim.render()
    settings.set_bool("/physics/resetOnStop", True)


def _view(app, scene_file: str | None) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from yam_sim.scene import PHYSICS_DT, scene_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT))
    sim.set_camera_view(eye=EYE, target=TARGET)
    scene = InteractiveScene(scene_cfg(scene_file))
    sim.reset()
    _settle(sim, scene, SETTLE_S)
    print(f"[yam-sim] viewing {scene_file or 'the built-in scene'}; close the window to quit", flush=True)
    _run(app, sim, scene)


def _edit(app, scene_file: Path) -> None:
    import isaaclab.sim as sim_utils
    import omni.timeline
    import omni.ui as ui
    from isaaclab.scene import InteractiveScene

    from yam_sim.scene import PHYSICS_DT, save_environment, scene_cfg

    import carb

    # PhysX writes body poses back to USD only on the CPU pipeline without Fabric, with
    # updateToUsd on (the Isaac Lab app files turn it off). That way objects settled with Play
    # are saved where they came to rest.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, use_fabric=False, device="cpu"))
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    sim.set_camera_view(eye=EYE, target=TARGET)
    scene = InteractiveScene(scene_cfg(scene_file if scene_file.exists() else None))
    sim.reset()
    _settle(sim, scene, SETTLE_S)
    _hand_over_to_isaac_sim(sim, scene)

    # Isaac Lab hides the toolbar's Stop button, and its Play button while stopped, so the
    # panel has its own.
    timeline = omni.timeline.get_timeline_interface()
    window = ui.Window("YAM scene", width=460, height=190)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(
                f"Saves everything except the arms and cameras to\n{scene_file}", word_wrap=True, height=0
            )
            status = ui.Label(
                "Play lets objects settle; Save keeps them where they are. Stop rewinds to before Play.",
                word_wrap=True,
                height=0,
            )

            def save() -> None:
                count = save_environment(scene_file)
                status.text = f"Saved {count} top-level prims to {scene_file.name}."
                print(f"[yam-sim] saved {count} prims to {scene_file}", flush=True)

            with ui.HStack(spacing=8, height=32):
                ui.Button("Play", clicked_fn=timeline.play)
                ui.Button("Pause", clicked_fn=timeline.pause)
                ui.Button("Stop", clicked_fn=timeline.stop)
            ui.Button("Save scene", clicked_fn=save, height=32)
    print(f"[yam-sim] editing {scene_file}; use the 'YAM scene' panel to save", flush=True)
    while app.is_running():
        if sim.is_playing():
            sim.step(render=True)
        else:
            sim.render()
