"""Build a simulation-ready YAM URDF from I2RT's upstream `yam.urdf`.

Upstream ships visual meshes only. Physics engines need collision geometry, so
this gives every link one <collision> per convex hull from
`make_collision_meshes.py`, at the link's visual origin (the hulls share the visual
mesh's frame), and recolors links to match the real arm.

    python scripts/make_sim_urdf.py   # writes assets/yam/yam_sim.urdf
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "yam"

# Real YAM: black anodized links, silver gripper tips.
DARK = "0.12 0.12 0.13 1"
METAL = "0.70 0.70 0.72 1"
LINK_COLORS = {"tip_left": METAL, "tip_right": METAL}


def main() -> None:
    tree = ET.parse(ASSET_DIR / "yam.urdf")
    robot = tree.getroot()
    for link in robot.iter("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        material = visual.find("material")
        if material is not None:
            material.set("name", f"{link.get('name')}_mat")
            material.find("color").set("rgba", LINK_COLORS.get(link.get("name"), DARK))
        if link.find("collision") is not None:
            continue
        hulls = sorted((ASSET_DIR / "collision" / link.get("name")).glob("*.stl"))
        if not hulls:
            raise FileNotFoundError(f"no collision hulls for {link.get('name')}; run make_collision_meshes.py")
        for hull in hulls:
            collision = ET.SubElement(link, "collision")
            collision.append(copy.deepcopy(visual.find("origin")))
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "mesh", filename=str(hull.relative_to(ASSET_DIR)), scale="1 1 1")
    ET.indent(tree)
    out = ASSET_DIR / "yam_sim.urdf"
    tree.write(out, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
