"""Check that each YAM link's imported collision geometry matches its visual mesh.

Spawns one arm at zero joint angles and probes the colliders PhysX actually simulates
with raycasts. For each link it samples points on the visual surface and casts a ray
inward along the surface normal, starting `--standoff` outside it, keeping only hits on
that link's own collider. The signed error is how much sooner the ray meets the collider
than the link's visual surface:

- positive: collider bulges out past the mesh (e.g. convex hulls bridging a concavity)
- negative: collider is inset from the mesh
- miss:     no collider of this link anywhere along the ray

    python scripts/check_collision.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--samples", type=int, default=3000, help="Visual surface points probed per link.")
parser.add_argument("--standoff", type=float, default=0.05, help="Ray start distance outside the surface (m).")
parser.add_argument("--tol", type=float, default=0.002, help="Error within this many metres counts as matching.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import carb  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
import trimesh  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from yam_sim.assets import YAM_CFG  # noqa: E402

ROOT = "/World/Yam"


def world_mesh(prim) -> trimesh.Trimesh:
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces, start = [], 0
    for c in counts:  # fan-triangulate polygons
        faces += [[idx[start], idx[start + k], idx[start + k + 1]] for k in range(1, c - 1)]
        start += c
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    pts = np.c_[np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64), np.ones(len(mesh.GetPointsAttr().Get()))]
    return trimesh.Trimesh((pts @ xf)[:, :3], np.array(faces), process=True)  # USD: row vectors


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 240, enable_scene_query_support=True))
    cfg = YAM_CFG.replace(prim_path=ROOT)
    cfg.init_state = cfg.init_state.replace(joint_pos={".*": 0.0})
    robot = Articulation(cfg)
    sim.reset()
    sim.step()

    stage = omni.usd.get_context().get_stage()
    visuals = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path.startswith(ROOT + "/") and "/visuals/" in path and prim.IsA(UsdGeom.Mesh):
            visuals.setdefault(path.split("/")[3], []).append(world_mesh(prim))

    sq = omni.physx.get_physx_scene_query_interface()
    rng = np.random.default_rng(0)
    print(f"\n{'link':10s} {'match':>6s} {'bulge':>6s} {'inset':>6s} {'miss':>6s}   err p5/p50/p95 mm   other-link hits")
    for link in sorted(visuals):
        vis = trimesh.util.concatenate(visuals[link])
        pts, face = trimesh.sample.sample_surface(vis, args.samples, seed=int(rng.integers(1 << 31)))
        normals = vis.face_normals[face]
        own = f"/{link}/collisions/"
        errs, others, origins = [], {}, []
        for p, n in zip(pts, normals):
            hits = []

            def report(h, hits=hits):
                hits.append((h.distance, h.collision))
                return True

            origin = p + n * args.standoff
            sq.raycast_all(carb.Float3(*origin), carb.Float3(*(-n)), 2 * args.standoff, report)
            mine = [d for d, c in hits if own in c]
            for d, c in hits:
                if own not in c and d < args.standoff:
                    other = c.split("/")[3]
                    others[other] = others.get(other, 0) + 1
            origins.append(origin)
            errs.append(min(mine) if mine else np.nan)
        # Reference: where the same ray first meets this link's visual surface. In a slot or
        # pocket that is the opposite wall, not the sampled point, so compare against it.
        origins = np.array(origins)
        dirs = -normals
        vis_hit = np.full(len(origins), args.standoff)
        locs, ray_idx, _ = vis.ray.intersects_location(origins, dirs, multiple_hits=True)
        if len(ray_idx):
            d = np.einsum("ij,ij->i", locs - origins[ray_idx], dirs[ray_idx])
            np.minimum.at(vis_hit, ray_idx, np.where(d > 1e-6, d, np.inf))
        inside = vis.contains(origins)  # ray starts inside the link's own solid: no reference
        errs = (vis_hit - np.array(errs))[~inside]
        hit = errs[~np.isnan(errs)]
        frac = lambda m: f"{100 * m.sum() / len(errs):5.1f}%"  # noqa: E731
        pct = np.percentile(hit * 1000, [5, 50, 95]) if len(hit) else [np.nan] * 3
        print(
            f"{link:10s} {frac(np.abs(hit) <= args.tol)} {frac(hit > args.tol)} {frac(hit < -args.tol)}"
            f" {frac(np.isnan(errs))}   {pct[0]:6.1f}/{pct[1]:5.1f}/{pct[2]:5.1f}     {others or ''}"
            + (f"  ({inside.sum()} rays skipped: start inside link)" if inside.any() else ""),
            flush=True,
        )
    del robot


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # Kit's app.close() can hang after a headless run
