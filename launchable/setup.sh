#!/bin/bash
# Brev launchable setup: turn a bare Ubuntu GPU VM into the YAM Isaac Lab Arena workstation.
#
# Installs Isaac Lab Arena 0.3.1 with its locked stack (Isaac Lab 3.0 from Arena's submodule,
# Isaac Sim 6, Python 3.12, all in Arena's uv venv), inspect-robots with its `agent` policy
# plugin, and this repo (`yam-arena` command, `yam_tabletop` Arena environment, `yam_arena`
# embodiment), then starts a browser desktop (Xvfb + xfce + noVNC on port 6080) as a
# systemd service.
#
# Run as the login user (with sudo) or as root; safe to re-run, finished steps are skipped.
# Paste it as the launchable's setup script, or run it from a checkout:
#     bash launchable/setup.sh
#
# Environment overrides:
#   YAM_PREFIX          install root (default /data if writable, else $HOME)
#   YAM_REPO_REF        branch or tag of this repo to clone when not run from a checkout (arena)
#   ARENA_REF           Isaac Lab Arena branch or tag (release/0.3.1)
#   INSPECT_ROBOTS_REF  inspect-robots tag or branch (v0.60.0)
#   YAM_VNC_PASSWORD    desktop password (default: random, saved to ~/yam-arena-desktop-password)
#
# Deploying this accepts the NVIDIA Omniverse license for Isaac Sim on the VM's behalf
# (OMNI_KIT_ACCEPT_EULA=YES): https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html
set -euo pipefail

# Brev may run setup scripts as root; install for the login user instead.
if [ "$(id -u)" = 0 ] && id ubuntu >/dev/null 2>&1; then
  if [ -f "$0" ]; then
    install -m 755 "$0" /tmp/yam-arena-setup.sh
  else  # piped in (`curl ... | bash`): fetch a copy the login user can run
    curl -fsSL https://raw.githubusercontent.com/ttpierson/YAM-inspect-Isaac/arena/launchable/setup.sh -o /tmp/yam-arena-setup.sh
    chmod 755 /tmp/yam-arena-setup.sh
  fi
  exec sudo -u ubuntu -H --preserve-env=YAM_PREFIX,YAM_REPO_REF,ARENA_REF,INSPECT_ROBOTS_REF,YAM_VNC_PASSWORD bash /tmp/yam-arena-setup.sh "$@"
fi

# The Arena launchable lives on the `arena` branch; `main` is the Isaac Lab launchable.
REPO_URL=https://github.com/ttpierson/YAM-inspect-Isaac.git
ARENA_REF=${ARENA_REF:-release/0.3.1}
INSPECT_ROBOTS_REF=${INSPECT_ROBOTS_REF:-v0.60.0}
if [ -z "${YAM_PREFIX:-}" ]; then
  if [ -d /data ] && [ -w /data ]; then YAM_PREFIX=/data; else YAM_PREFIX=$HOME; fi
fi
ARENA=$YAM_PREFIX/IsaacLab-Arena
VENV=$ARENA/.venv
LOG=$YAM_PREFIX/yam-arena-setup.log
mkdir -p "$YAM_PREFIX"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [yam-arena-setup] $*"; }
step "installing into $YAM_PREFIX (log: $LOG)"

# This repo: use the checkout the script lives in, else clone it.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)
if [ -n "$HERE" ] && [ -f "$HERE/../pyproject.toml" ] && grep -q '^name = "yam-arena"' "$HERE/../pyproject.toml"; then
  REPO=$(cd "$HERE/.." && pwd)
elif grep -qs '^name = "yam-arena"' "$HOME/YAM-inspect-Isaac/pyproject.toml"; then  # Brev's clone, if on this branch
  REPO=$HOME/YAM-inspect-Isaac
else
  REPO=$YAM_PREFIX/YAM-inspect-Arena
  [ -d "$REPO/.git" ] || git clone --branch "${YAM_REPO_REF:-arena}" "$REPO_URL" "$REPO"
fi

step "system packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl cmake build-essential libgl1-mesa-dev libglu1-mesa libvulkan1 \
  libx11-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev \
  xvfb xfce4 xfce4-terminal dbus-x11 x11vnc novnc websockify x11-xserver-utils
# A locked screen would hide the desktop behind a password prompt nobody set.
sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y xfce4-screensaver light-locker || true

step "uv"
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=$YAM_PREFIX/.uv-cache

step "Isaac Lab Arena $ARENA_REF (Isaac Lab 3.0, Isaac Sim 6)"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
[ -d "$ARENA/.git" ] || git clone --depth 1 --branch "$ARENA_REF" https://github.com/isaac-sim/IsaacLab-Arena.git "$ARENA"
# Arena's submodules use SSH URLs, and a fresh VM has no GitHub key.
git -C "$ARENA" -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --depth 1
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import isaaclab_arena, isaaclab" 2>/dev/null; then
  # Arena's lockfile pins the whole stack; --frozen installs it as locked without re-checking it
  # against pyproject.toml (release/0.3.1's lockfile fails --locked). --inexact keeps the
  # packages installed below on re-runs.
  (cd "$ARENA" && uv sync --frozen --inexact)
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# What follows installs into Arena's venv but must not move Arena's pins (numpy, torch, ...).
CONSTRAINTS=$YAM_PREFIX/arena-constraints.txt
uv pip freeze | grep -v -e '^-e ' -e ' @ ' > "$CONSTRAINTS"

step "inspect-robots $INSPECT_ROBOTS_REF"
# A full clone: inspect-robots takes its version from the git tags (hatch-vcs).
[ -d "$YAM_PREFIX/inspect-robots/.git" ] ||
  git clone --branch "$INSPECT_ROBOTS_REF" https://github.com/robocurve/inspect-robots.git "$YAM_PREFIX/inspect-robots"
uv pip install -c "$CONSTRAINTS" -e "$YAM_PREFIX/inspect-robots" -e "$YAM_PREFIX/inspect-robots/plugins/inspect-robots-agent"

step "yam-arena ($REPO)"
uv pip install -c "$CONSTRAINTS" -e "$REPO"
inspect-robots list embodiments | grep yam_arena

step "shell"
grep -q "# yam-arena launchable" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<RC

# yam-arena launchable
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
source $VENV/bin/activate
cd $REPO
RC

step "desktop service"
PASSWORD_FILE=$HOME/yam-arena-desktop-password
if [ ! -f "$HOME/.vnc/passwd" ] || [ ! -f "$PASSWORD_FILE" ]; then
  mkdir -p "$HOME/.vnc"
  password=${YAM_VNC_PASSWORD:-$(head -c 6 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 8)}
  x11vnc -storepasswd "$password" "$HOME/.vnc/passwd" >/dev/null
  echo "$password" > "$PASSWORD_FILE"
  chmod 600 "$HOME/.vnc/passwd" "$PASSWORD_FILE"
fi
sudo tee /etc/systemd/system/yam-arena-desktop.service >/dev/null <<UNIT
[Unit]
Description=YAM sim desktop (Xvfb :1 + xfce, noVNC on port 6080)
After=network.target

[Service]
User=$(id -un)
Environment=HOME=$HOME
ExecStart=$REPO/launchable/desktop.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
# The Isaac Lab launchable's desktop uses the same display (:1) and port (6080).
if [ -f /etc/systemd/system/yam-desktop.service ]; then
  sudo systemctl disable --now yam-desktop.service || true
fi
sudo systemctl enable yam-arena-desktop.service
sudo systemctl restart yam-arena-desktop.service

step "done"
cat <<DONE
Desktop: forward port 6080 (Brev secure link, or \`brev port-forward <instance> -p 6080:6080\`)
and open http://localhost:6080/vnc.html. Password: $(cat "$PASSWORD_FILE" 2>/dev/null || echo "see $PASSWORD_FILE")

In a terminal on the instance:
    yam-arena view                                   # the scene, arms holding still
    yam-arena edit ~/scene.usda                      # edit and save the environment
    python scripts/view_scene.py                     # loop the scripted contact demo
    yam-arena eval experiments/yam_tabletop.yaml     # Arena evaluation: success rate and metrics
    inspect-robots run --embodiment yam_arena --policy agent --scorer success_at_end \\
        --instruction "put the red cube on the plate"
DONE
