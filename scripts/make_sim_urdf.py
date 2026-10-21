"""Build a simulation-ready YAM URDF from I2RT's upstream `yam.urdf`.

Upstream ships visual meshes only. Physics engines need collision geometry, so
this copies every <visual> into a matching <collision> (the importer then
convex-decomposes the meshes) and recolors links to match the real arm.

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
        if link.find("collision") is None:
            collision = ET.SubElement(link, "collision")
            collision.append(copy.deepcopy(visual.find("origin")))
            collision.append(copy.deepcopy(visual.find("geometry")))
    ET.indent(tree)
    out = ASSET_DIR / "yam_sim.urdf"
    tree.write(out, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
