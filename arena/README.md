# Arena tabletop

The bimanual YAM tabletop scene in [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena):
the same rig, scene, scene editor, scripted demo and Inspect Robots embodiment as `yam-sim` at the
repo root, with evaluations run by Arena. `launchable/setup.sh` installs it into the Arena venv,
`source <prefix>/IsaacLab-Arena/yam.env` to get there, and run the commands below from the repo root.

It shares that venv with [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena),
whose package and embodiment are called `yam_arena`, so this one is `yam_tabletop`.

Targets Arena 0.3.1 (Isaac Lab 3.0, Isaac Sim 6, Python 3.12). The URDF and meshes are read from
`assets/` at the repo root.

## Layout

| Path | What |
|---|---|
| `arena/yam_tabletop/rig.py` | Rig layout, cameras, timing and the 14-D `yam_arms` contract; imports no Isaac code |
| `arena/yam_tabletop/assets.py` | `YAM_CFG`, one arm as an Isaac Lab `ArticulationCfg` |
| `arena/yam_tabletop/arena/embodiment.py` | `yam_tabletop_bimanual`, the Arena embodiment: two arms, the rig's cameras, 14-D actions and state |
| `arena/yam_tabletop/arena/mdp.py` | The arm action term (joint targets, gripper 0..1, wrist cameras) and the joint-state observation |
| `arena/yam_tabletop/arena/environment.py` | `yam_tabletop`, the Arena environment: table, cube, plate (or a scene file) and the pick-and-place task |
| `arena/yam_tabletop/arena/policies.py` | Arena policies: `InspectRobotsPolicy` (any Inspect Robots policy) and `YamDemoPolicy` (scripted demo) |
| `arena/yam_tabletop/arena/runners.py` | Arena's experiment and policy runners with `yam_tabletop` registered |
| `arena/yam_tabletop/arena/scene_file.py` | Saving and loading scene files |
| `arena/yam_tabletop/cli.py` | The `yam-arena` command: `view`, `edit`, `eval`, `rollout` |
| `arena/yam_tabletop/embodiment.py` | `yam_tabletop`, the Arena environment as an Inspect Robots embodiment |
| `arena/experiments/yam_tabletop.yaml` | An Arena Experiment: the `agent` policy and the scripted demo |
| `arena/scripts/view_scene.py` | Loops the scripted demo, printing arm contact forces and optionally saving camera frames |
| `arena/scripts/diagnose_physics.py` | Runs the scripted demo and checks grasping, contact and joint limits |
| `arena/scripts/check_collision.py` | Measures, per link, how closely the simulated colliders follow the visual meshes |

## The Arena environment

`yam_tabletop` composes three Arena pieces:

- **Embodiment** `yam_tabletop_bimanual`: both arms on the near edge of a 0.75 m table, 0.6 m apart,
  reaching along +X. Each arm exposes `joint1`-`joint6` plus the finger joints
  `joint7`/`joint8` (0 = closed, -0.04695 m = open); at zero joint angles the gripper points
  along +X. The action is 14-D, `[left j1..j6, left gripper, right j1..j6, right gripper]`,
  absolute joint targets in rad and grippers 0 (closed) to 1 (open), at 10 Hz
  (`--control_hz`) on 240 Hz physics. The arms are gravity-compensated, as i2rt drives the real ones. Cameras (with
  `enable_cameras`): `top_cam` (overhead D435) and `left_cam`/`right_cam` (wrist D405s), 640x360 RGB.
- **Scene**: the table, a red 4 cm cube in front of the left arm and a white plate in front of
  the right, ground and dome light; or the environment of a scene file (`--scene`). Each
  reset puts the arms in their ready pose and shifts the cube and plate by up to 3 cm
  (`--object_jitter`).
- **Task**: Arena's `PickAndPlaceTask`. An episode succeeds once `--pick_up_object` (cube)
  has been lifted and rests on `--destination` (plate), and fails if it falls off the table.
  Metrics: success rate and object-moved rate.

## Evaluate with Arena

Inside the Arena environment (`source <prefix>/IsaacLab-Arena/yam.env`), `yam-arena eval` runs an
Arena Experiment YAML and `yam-arena rollout` runs Arena's policy runner. Both are Arena's own
runners with `yam_tabletop` registered; their flags pass through unchanged.

```bash
yam-arena eval arena/experiments/yam_tabletop.yaml                 # every Run in the file
yam-arena eval arena/experiments/yam_tabletop.yaml shared.rollout_limit.num_episodes=10

# One policy, one environment; --viz kit shows it on DISPLAY (the noVNC desktop)
yam-arena rollout --viz kit --num_episodes 1 \
    --policy_type yam_tabletop.arena.policies.YamDemoPolicy yam_tabletop
yam-arena rollout --enable_cameras --num_episodes 5 \
    --policy_type yam_tabletop.arena.policies.InspectRobotsPolicy --inspect_policy agent \
    yam_tabletop --task_description "put the red cube on the plate"
```

Arena writes per-episode results, metrics and an HTML report under `outputs/<timestamp>/`.

`InspectRobotsPolicy` runs any registered Inspect Robots policy (`--inspect_policy agent`,
`molmoact2`, `gr00t`, ...; factory arguments as JSON in `--policy_args`). It sees exactly
what the real rig's `yam_arms` embodiment produces (cameras resized to 224x224, 14-D
`joint_pos`), so a policy written for the real arms is scored by Arena's task metrics
unchanged. It drives one environment at a time. `YamDemoPolicy` is a scripted motion that
exercises contact: the left arm picks the cube and sets it down 10 cm nearer the arms, the right
arm presses its closed fingers onto the plate. `python arena/scripts/diagnose_physics.py` runs it
once and checks that the cube was lifted, the fingers stopped on the plate, and the joints
stayed within limits.
`python arena/scripts/view_scene.py` loops it in the GUI and prints each arm link's contact force;
`--viz none --screenshots out/ --steps 120` runs headless and saves overview, top and wrist frames instead.

## View and edit the scene

Both commands open the Isaac Sim GUI on the `yam_tabletop` environment. Over ssh with no `DISPLAY`, they use the instance's noVNC desktop on `:1`.

```bash
yam-arena view                       # the built-in scene; the arms hold their ready pose
yam-arena view scenes/kitchen.usda   # a scene saved by `yam-arena edit`
yam-arena edit scenes/kitchen.usda   # opens the file if it exists, else the built-in scene
```

`edit` opens the scene with physics stopped and adds a "YAM scene" panel with Play,
Pause, Stop and Save buttons. Use the panel's buttons, not the toolbar's: Isaac Lab drives
the simulation itself, not through Kit's timeline. Edit with Isaac Sim's own tools: move prims with the gizmo, add shapes from
Create, drag assets in from the content browser, change lights and materials in the
Property panel. The arms stand frozen in their ready pose for reference (they collide
but don't move). To let objects settle onto the table, press Play, then Save (playing
or paused). Stop rewinds to where things were when you pressed Play. Edits made while
paused only reach physics after Stop and Play.

Under the hood, the editor hands the stage over to Isaac Sim once the arms are posed.
Editing a physics prim makes PhysX re-read the stage, which breaks Isaac Lab's physics
handles, so Isaac Lab doesn't drive anything while you edit.

Save writes the *environment* (everything except the two arms and the policy cameras,
which always come from the embodiment) to the file, as one self-contained USD under
`/Environment`. It is flattened, so assets dragged in from a library are copied in rather
than referenced. On load the file becomes the environment's Arena background, and every
free rigid body in it an object that resets to its saved pose, named after its prim
(lower-cased). The task picks up `--pick_up_object` and puts it on `--destination`, both
named that way (default `cube` and `plate`); without both, the environment runs with no
task. The editor is the part of the port that leans most on Isaac Lab internals; if Play
misbehaves, `yam-arena view` and the runners are unaffected.

## Inspect Robots

`yam_tabletop` has the same observation and action contract as the real rig's `yam_arms`
embodiment from [inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam),
so policies written for the real arms (`molmoact2`, `gr00t`, `agent`) run against it
unchanged:

| | |
|---|---|
| Cameras | `top_cam` (overhead D435), `left_cam` and `right_cam` (D405 on each wrist), RGB 224x224 |
| State `joint_pos` | 14-D: `[left j0..j5, left gripper, right j0..j5, right gripper]`, joints in rad, grippers 0 closed to 1 open |
| Action | the same 14-D layout as absolute joint targets, at 10 Hz |

The simulation is the Arena `yam_tabletop` environment, and its task judges the episode:
the step that puts the cube on the plate ends it with termination reason `success`, which
the `success_at_end` scorer counts; knocking the cube off the table ends it with
`object_dropped`.

```bash
inspect-robots run --embodiment yam_tabletop --policy agent --scorer success_at_end \
    --instruction "put the red cube on the plate"
inspect-robots run --embodiment yam_tabletop -E headless=false ...   # render on DISPLAY (noVNC desktop)
```

On a machine set up for the real rig, make `yam_tabletop` its simulated counterpart so
`--sim` swaps it in for `yam_arms`. Add it to `[defaults]` in
`~/.config/inspect-robots/config.ini`:

```ini
[defaults]
embodiment = yam_arms
sim_embodiment = yam_tabletop
```

Other `-E` options: `scene` (a file from `yam-arena edit`), `pick_up_object` and
`destination` (its objects for the task), `episode_length_s`, `control_hz` (default 10),
`cam_height`, `cam_width`, `device`. Every reset shifts the cube and plate by up to 3 cm; a seeded reset does it
reproducibly.
