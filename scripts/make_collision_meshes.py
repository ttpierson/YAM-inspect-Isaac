"""Convex-decompose each YAM link mesh into collision hulls with CoACD.

PhysX's built-in convex decomposition (V-HACD at default settings) fills the links'
pockets and finger recesses, leaving the collider several millimetres proud of the
mesh. CoACD follows concave surfaces much more closely. Its hulls are written to
`assets/yam/collision/<link>/` in the visual mesh's own frame, and `make_sim_urdf.py`
turns each one into a <collision> element. Hulls are capped at 60 vertices so PhysX's
64-vertex hull limit never simplifies them.

Needs `coacd` and `trimesh` (not part of Isaac Lab; any Python env will do):

    pip install coacd==1.0.7 trimesh==4.5.1
    python scripts/make_collision_meshes.py
    python scripts/make_sim_urdf.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import coacd
import trimesh

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "yam"
OUT_DIR = ASSET_DIR / "collision"

# CoACD concavity threshold (lower = tighter fit, more hulls). 0.03 is a little tighter
# than CoACD's default 0.05; `check_collision.py` measures the result in Isaac Sim.
THRESHOLD = 0.03


def main() -> None:
    coacd.set_log_level("error")
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    for stl in sorted((ASSET_DIR / "assets").glob("*.stl")):
        mesh = trimesh.load(stl)
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=THRESHOLD, max_ch_vertex=60, seed=0)
        link_dir = OUT_DIR / stl.stem
        link_dir.mkdir(parents=True)
        for i, (vertices, faces) in enumerate(parts):
            trimesh.Trimesh(vertices, faces).export(link_dir / f"{stl.stem}_{i:03d}.stl")
        print(f"{stl.stem}: {len(parts)} hulls")


if __name__ == "__main__":
    main()
