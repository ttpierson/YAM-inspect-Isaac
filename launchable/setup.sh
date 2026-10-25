#!/bin/bash
# Brev launchable setup: turn a bare Ubuntu GPU VM into the YAM sim workstation.
#
# Installs Isaac Sim 5.1 + Isaac Lab 2.3.2 (uv venv, Python 3.11), inspect-robots with its
# `agent` policy plugin, and this repo (`yam-sim` command, `yam_sim` embodiment), then
# starts a browser desktop (Xvfb + xfce + noVNC on port 6080) as a systemd service.
#
# Run as the login user (with sudo) or as root; safe to re-run, finished steps are skipped.
# Paste it as the launchable's setup script, or run it from a checkout:
#     bash launchable/setup.sh
#
# Environment overrides:
#   YAM_PREFIX          install root (default /data if writable, else $HOME)
#   YAM_REPO_REF        branch or tag of this repo to clone when not run from a checkout (main)
#   INSPECT_ROBOTS_REF  inspect-robots tag or branch (v0.60.0)
#   YAM_VNC_PASSWORD    desktop password (default: random, saved to ~/yam-desktop-password)
#
# Deploying this accepts the NVIDIA Omniverse license for Isaac Sim on the VM's behalf
# (OMNI_KIT_ACCEPT_EULA=YES): https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html
set -euo pipefail

# Brev may run setup scripts as root; install for the login user instead.
if [ "$(id -u)" = 0 ] && id ubuntu >/dev/null 2>&1; then
  if [ -f "$0" ]; then
    install -m 755 "$0" /tmp/yam-setup.sh
  else  # piped in (`curl ... | bash`): fetch a copy the login user can run
    curl -fsSL https://raw.githubusercontent.com/ttpierson/YAM-inspect-Isaac/main/launchable/setup.sh -o /tmp/yam-setup.sh
    chmod 755 /tmp/yam-setup.sh
  fi
  exec sudo -u ubuntu -H --preserve-env=YAM_PREFIX,YAM_REPO_REF,INSPECT_ROBOTS_REF,YAM_VNC_PASSWORD bash /tmp/yam-setup.sh "$@"
fi

REPO_URL=https://github.com/ttpierson/YAM-inspect-Isaac.git
ISAACLAB_REF=v2.3.2
ISAACSIM_VERSION=5.1.0
INSPECT_ROBOTS_REF=${INSPECT_ROBOTS_REF:-v0.60.0}
if [ -z "${YAM_PREFIX:-}" ]; then
  if [ -d /data ] && [ -w /data ]; then YAM_PREFIX=/data; else YAM_PREFIX=$HOME; fi
fi
VENV=$YAM_PREFIX/env_isaaclab
LOG=$YAM_PREFIX/yam-setup.log
mkdir -p "$YAM_PREFIX"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [yam-setup] $*"; }
step "installing into $YAM_PREFIX (log: $LOG)"

# This repo: use the checkout the script lives in, else clone it.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)
if [ -n "$HERE" ] && [ -f "$HERE/../pyproject.toml" ] && grep -q '^name = "yam-sim"' "$HERE/../pyproject.toml"; then
  REPO=$(cd "$HERE/.." && pwd)
elif [ -f "$HOME/YAM-inspect-Isaac/pyproject.toml" ]; then  # where Brev clones the launchable's repo
  REPO=$HOME/YAM-inspect-Isaac
else
  REPO=$YAM_PREFIX/YAM-inspect-Isaac
  [ -d "$REPO/.git" ] || git clone --branch "${YAM_REPO_REF:-main}" "$REPO_URL" "$REPO"
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

step "Isaac Sim $ISAACSIM_VERSION"
[ -x "$VENV/bin/python" ] || uv venv --python 3.11 --seed "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
if ! python -c "import isaacsim" 2>/dev/null; then
  uv pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
  uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
fi

step "Isaac Lab $ISAACLAB_REF"
[ -d "$YAM_PREFIX/IsaacLab/.git" ] ||
  git clone --depth 1 --branch "$ISAACLAB_REF" https://github.com/isaac-sim/IsaacLab.git "$YAM_PREFIX/IsaacLab"
if ! python -c "import isaaclab" 2>/dev/null; then
  # flatdict 4.0.1 (an Isaac Lab dependency) does not build with setuptools 81+.
  echo "setuptools<81" > "$YAM_PREFIX/build-constraints.txt"
  uv pip install --build-constraint "$YAM_PREFIX/build-constraints.txt" flatdict==4.0.1
  # `none`: the Isaac Lab extensions only, no RL training frameworks.
  (cd "$YAM_PREFIX/IsaacLab" && ./isaaclab.sh --install none)
fi

step "inspect-robots $INSPECT_ROBOTS_REF"
# A full clone: inspect-robots takes its version from the git tags (hatch-vcs).
[ -d "$YAM_PREFIX/inspect-robots/.git" ] ||
  git clone --branch "$INSPECT_ROBOTS_REF" https://github.com/robocurve/inspect-robots.git "$YAM_PREFIX/inspect-robots"
uv pip install -e "$YAM_PREFIX/inspect-robots" -e "$YAM_PREFIX/inspect-robots/plugins/inspect-robots-agent"

step "yam-sim ($REPO)"
uv pip install -e "$REPO"
inspect-robots list embodiments | grep yam_sim

step "shell"
grep -q "# yam-sim launchable" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<RC

# yam-sim launchable
export OMNI_KIT_ACCEPT_EULA=YES
source $VENV/bin/activate
cd $REPO
RC

step "desktop service"
PASSWORD_FILE=$HOME/yam-desktop-password
if [ ! -f "$HOME/.vnc/passwd" ]; then
  mkdir -p "$HOME/.vnc"
  password=${YAM_VNC_PASSWORD:-$(head -c 6 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 8)}
  x11vnc -storepasswd "$password" "$HOME/.vnc/passwd" >/dev/null
  echo "$password" > "$PASSWORD_FILE"
  chmod 600 "$HOME/.vnc/passwd" "$PASSWORD_FILE"
fi
sudo tee /etc/systemd/system/yam-desktop.service >/dev/null <<UNIT
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
sudo systemctl enable yam-desktop.service
sudo systemctl restart yam-desktop.service

step "done"
cat <<DONE
Desktop: forward port 6080 (Brev secure link, or \`brev port-forward <instance> -p 6080:6080\`)
and open http://localhost:6080/vnc.html. Password: $(cat "$PASSWORD_FILE" 2>/dev/null || echo "see $PASSWORD_FILE")

In a terminal on the instance:
    yam-sim view                                   # the scene, arms holding still
    yam-sim edit ~/scene.usda                      # edit and save the environment
    inspect-robots run --embodiment yam_sim --policy agent --instruction "put the red cube on the plate"
DONE
