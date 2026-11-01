"""Scene files: the *environment* of the YAM tabletop, saved by `yam-arena edit`.

The arms and the policy cameras always come from the embodiment. Everything else (table,
objects, lights, ground, anything added in the editor) is the environment. A scene file holds
it as one self-contained USD under the default prim ``/Environment``, in env-local
coordinates, and the Arena environment loads it as a background whose free rigid bodies reset
to where the file put them.

Needs a running Isaac Sim app (``pxr``).
"""

from __future__ import annotations

import re
from pathlib import Path

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

ROOT = "/Environment"
# Where the Arena environment spawns a loaded scene file, under each env.
BACKGROUND_NAME = "environment"
ENV_NS = "/World/envs/env_0"
# Prims under ENV_NS that come from the embodiment, never from a scene file.
EMBODIMENT_PRIMS = {"LeftArm", "RightArm", "TopCam", "LeftWristCam", "RightWristCam"}
# Prims under /World that Isaac Lab generates for every scene: the env tree, the per-env
# collision filter groups and debug markers (stale copies would filter or draw the wrong things).
GENERATED_PRIMS = {"envs", "collisions", "Visuals"}


def _environment_roots(stage: Usd.Stage) -> dict[Sdf.Path, Sdf.Path]:
    """Top-level environment prims of an Arena stage, mapped to their path in a scene file.

    A scene file loaded as the background contributes its children, so saving it again does
    not nest it one level deeper.
    """
    roots: dict[Sdf.Path, Sdf.Path] = {}
    world = stage.GetPrimAtPath("/World")
    env = stage.GetPrimAtPath(ENV_NS)
    children = [p for p in world.GetChildren() if p.GetName() not in GENERATED_PRIMS] if world else []
    for prim in env.GetChildren() if env else []:
        if prim.GetName() == BACKGROUND_NAME:
            children += list(prim.GetChildren())
        elif prim.GetName() not in EMBODIMENT_PRIMS:
            children.append(prim)
    for prim in children:
        roots[prim.GetPath()] = Sdf.Path(ROOT).AppendChild(prim.GetName())
    return roots


def _remap(path: Sdf.Path, roots: dict[Sdf.Path, Sdf.Path]) -> Sdf.Path:
    for src, dst in roots.items():
        if path.HasPrefix(src):
            return path.ReplacePrefix(src, dst)
    return path


def _remap_targets(layer: Sdf.Layer, roots: dict[Sdf.Path, Sdf.Path]) -> None:
    """Point relationships and connections (material bindings, joints) at the moved prims."""

    def remap_list(paths: Sdf.PathListOp) -> None:
        for field in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
            items = getattr(paths, field)
            if items:
                setattr(paths, field, [_remap(p, roots) for p in items])

    def visit(path: Sdf.Path) -> None:
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec):
            remap_list(spec.targetPathList)
        elif isinstance(spec, Sdf.AttributeSpec):
            remap_list(spec.connectionPathList)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)


def save_environment(stage: Usd.Stage, path: str | Path) -> int:
    """Write the environment of an Arena stage to a scene file at `path`; return the prim count.

    The stage is flattened first, so the file is self-contained: assets dragged in from a
    library are copied in, not referenced. Every free rigid body gets a contact report, so a
    task can sense what it rests on.
    """
    flat = stage.Flatten()
    roots = _environment_roots(Usd.Stage.Open(flat))
    out = Sdf.Layer.CreateAnonymous(".usda")
    root = Sdf.CreatePrimInLayer(out, ROOT)
    root.specifier, root.typeName = Sdf.SpecifierDef, "Xform"
    for src, dst in roots.items():
        Sdf.CopySpec(flat, src, out, dst)
    _remap_targets(out, roots)
    out.defaultPrim = ROOT.lstrip("/")
    out.pseudoRoot.SetInfo("upAxis", "Z")
    out.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    saved = Usd.Stage.Open(out)
    for prim in free_rigid_bodies(saved):
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.Export(str(path))
    return len(roots)


def free_rigid_bodies(stage: Usd.Stage) -> list[Usd.Prim]:
    """Rigid bodies that are not part of a larger rigid body or articulation, and not kinematic."""
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
            continue
        parent = prim.GetParent()
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI) and not parent.HasAPI(UsdPhysics.ArticulationRootAPI):
            parent = parent.GetParent()
        if not parent:
            bodies.append(prim)
    return bodies


def object_names(path: str | Path) -> dict[str, str]:
    """The free rigid bodies of a scene file, by lower-cased prim name, as paths relative to ROOT."""
    stage = Usd.Stage.Open(str(path))
    names: dict[str, str] = {}
    for prim in free_rigid_bodies(stage):
        name = re.sub(r"\W", "_", prim.GetName()).lower()
        while name in names:
            name += "_"
        names[name] = str(prim.GetPath().MakeRelativePath(Sdf.Path(ROOT)))
    return names
