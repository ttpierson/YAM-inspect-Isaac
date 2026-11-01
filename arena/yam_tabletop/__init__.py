"""The bimanual I2RT YAM tabletop for Isaac Lab Arena; the assets are shared with yam_sim at the repo root."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
YAM_URDF = REPO_ROOT / "assets" / "yam" / "yam_sim.urdf"
