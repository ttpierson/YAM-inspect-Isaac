#!/bin/bash
# Brev launchable setup: turn a bare Ubuntu GPU VM into the YAM sim workstation.
#
# Installs Isaac Sim 5.1 + Isaac Lab 2.3.2 (uv venv, Python 3.11), inspect-robots with its
# `agent` policy plugin, and this repo (`yam-sim` command, `yam_sim` embodiment). Beside them,
# in Arena's own venv (Isaac Sim 6, Isaac Lab 3.0, Python 3.12): Isaac Lab Arena 0.3.1,
# yam-isaac-lab-arena (`yam_arena` embodiment, `yam_scripted` demo policy), Dream-1 (`dream`
# policy), and this repo's Arena scene (`yam-arena` command, `yam_tabletop` embodiment).
# Then starts a browser desktop (Xvfb + xfce + noVNC on port 6080) and the trial reports for
# ~/yam-eval (port 8300) as systemd services, and plays the scripted demo once. New shells
# start in the Arena venv.
#
# Run as the login user (with sudo) or as root; safe to re-run, finished steps are skipped.
# Paste it as the launchable's setup script, or run it from a checkout:
#     bash launchable/setup.sh
#
# Environment overrides:
#   YAM_PREFIX          install root (default /data if writable, else $HOME)
#   YAM_REPO_REF        branch or tag of this repo to clone when not run from a checkout (main)
#   INSPECT_ROBOTS_REF  inspect-robots tag or branch (v1.0.0)
#   YAM_VNC_PASSWORD    desktop password (default: random, saved to ~/yam-desktop-password)
#   YAM_ARENA_STACK_REF yam-isaac-lab-arena commit or tag to install (see below)
#   DREAM_REF           inspect-robots-dream commit or tag to install (see below)
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
  exec sudo -u ubuntu -H --preserve-env=YAM_PREFIX,YAM_REPO_REF,INSPECT_ROBOTS_REF,YAM_VNC_PASSWORD,YAM_ARENA_STACK_REF,DREAM_REF bash /tmp/yam-setup.sh "$@"
fi

REPO_URL=https://github.com/ttpierson/YAM-inspect-Isaac.git
ISAACLAB_REF=v2.3.2
ISAACSIM_VERSION=5.1.0
INSPECT_ROBOTS_REF=${INSPECT_ROBOTS_REF:-v1.0.0}
YAM_ARENA_STACK_URL=https://github.com/kairosiann/yam-isaac-lab-arena.git
YAM_ARENA_STACK_REF=${YAM_ARENA_STACK_REF:-0059d02a1a91bc6834c1d0846022891df0c78730}
DREAM_REF=${DREAM_REF:-b5e936a201cec66c54c297e522c920d98445e339}
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
git -C "$YAM_PREFIX/inspect-robots" fetch --tags origin
git -C "$YAM_PREFIX/inspect-robots" checkout --detach "$INSPECT_ROBOTS_REF"
uv pip install -e "$YAM_PREFIX/inspect-robots" -e "$YAM_PREFIX/inspect-robots/plugins/inspect-robots-agent"

step "yam-sim ($REPO)"
uv pip install -e "$REPO"
inspect-robots list embodiments | grep yam_sim

step "Isaac Lab Arena, YAM embodiment, Dream-1 and scripted demo"
ARENA=$YAM_PREFIX/IsaacLab-Arena
ARENA_STACK=$YAM_PREFIX/yam-isaac-lab-arena
[ -d "$ARENA_STACK/.git" ] || git clone "$YAM_ARENA_STACK_URL" "$ARENA_STACK"
git -C "$ARENA_STACK" fetch --tags origin
git -C "$ARENA_STACK" checkout --detach "$YAM_ARENA_STACK_REF"
# Its setup.sh builds Arena's own venv, so it runs outside the Isaac Lab one.
(
  deactivate
  export ACCEPT_EULA=Y
  ARENA_DIR=$ARENA DREAM_SPEC="git+https://github.com/kairosiann/inspect-robots-dream@$DREAM_REF" \
    bash "$ARENA_STACK/setup.sh"
)
# Arena 0.3.1 still points at the staging bucket, which no longer has its 6.1 assets.
sed -i 's|^ARENA_NUCLEUS_DIR: str = .*|ARENA_NUCLEUS_DIR: str = ISAACLAB_NUCLEUS_DIR|' \
  "$ARENA/isaaclab_arena/assets/nucleus.py"

# Where the guides expect them, also when installing under /data.
for dir in IsaacLab-Arena yam-isaac-lab-arena; do
  [ -e "$HOME/$dir" ] || ln -s "$YAM_PREFIX/$dir" "$HOME/$dir"
done
[ -e "$HOME/my_task.py" ] || cp "$ARENA_STACK/examples/stack_cubes.py" "$HOME/my_task.py"

step "Arena tabletop scene ($REPO/arena)"
uv pip install --python "$ARENA/.venv/bin/python" --override "$ARENA_STACK/overrides.txt" -e "$REPO/arena"
ARENA_EMBODIMENTS=$("$ARENA/.venv/bin/inspect-robots" list embodiments)
ARENA_POLICIES=$("$ARENA/.venv/bin/inspect-robots" list policies)
for name in yam_arena yam_tabletop; do grep -w "$name" <<<"$ARENA_EMBODIMENTS"; done
for name in dream yam_scripted; do grep -w "$name" <<<"$ARENA_POLICIES"; done

step "shell"
# Arena is the default; `source $VENV/bin/activate` switches to Isaac Lab and yam-sim. A VM set up
# before Arena was the default has the old block, and this one comes after it, so this one wins.
grep -q "# yam launchable" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<RC

# yam launchable
source $ARENA/yam.env
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

step "trial reports service"
# `view` reads the logs directly in this one directory, so runs log to ~/yam-eval itself. On all
# interfaces: Brev's secure-link proxy connects to the VM's address, not loopback. It has no
# password, so open 8300 as a secure link only, not as a public TCP port.
mkdir -p "$HOME/yam-eval"
sudo tee /etc/systemd/system/yam-reports.service >/dev/null <<UNIT
[Unit]
Description=YAM trial reports for ~/yam-eval (port 8300)
After=network.target

[Service]
User=$(id -un)
Environment=HOME=$HOME
Environment=PATH=$ARENA/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$ARENA/.venv/bin/inspect-robots view $HOME/yam-eval --serve --host 0.0.0.0 --port 8300
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable yam-reports.service
sudo systemctl restart yam-reports.service

step "first run: the scripted demo"
# Isaac Sim's first start compiles its shaders, which takes minutes; do it here rather than in the
# user's first run, and leave a report to open. The install is usable without it, so don't fail.
if ! ls "$HOME"/yam-eval/*.json >/dev/null 2>&1; then
  (
    deactivate
    # shellcheck disable=SC1091
    source "$ARENA/yam.env"
    timeout 1800 inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E scene=station -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "put apple into the bowl" --scorer success_at_end \
      --max-steps 600 --store-frames --log-dir "$HOME/yam-eval" </dev/null
  ) || echo "[yam-setup] warning: the scripted demo failed; the install itself finished. See the output above."
fi

step "done"
cat <<DONE
Desktop: the 6080 secure link (or \`brev port-forward <instance> -p 6080:6080\`, then
http://localhost:6080/vnc.html). Password: $(cat "$PASSWORD_FILE" 2>/dev/null || echo "see $PASSWORD_FILE")
Trial reports: the 8300 secure link (or \`brev port-forward <instance> -p 8300:8300\`).

New shells are in the Arena venv. Run the scripted demo; -E headless=false shows it on the desktop:
    inspect-robots run --embodiment yam_arena --policy yam_scripted \\
        -E scene=station -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \\
        --instruction "put apple into the bowl" --scorer success_at_end \\
        --max-steps 600 --store-frames --log-dir ~/yam-eval
For Dream-1, save a key from https://platform.dreamlabs.ai/api-keys in ~/.dreamlabs/credentials,
then use --policy dream and --max-steps 2400. Keep --log-dir ~/yam-eval to see runs in the reports.

Also here:
    yam-arena view                                 # your Arena tabletop scene
    yam-arena eval arena/experiments/yam_tabletop.yaml
    source $VENV/bin/activate                      # switch to Isaac Lab, for yam-sim
DONE
