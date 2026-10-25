#!/bin/bash
# The launchable's browser desktop, run in the foreground by yam-desktop.service.
#
# A virtual X display (Xvfb :1, 1920x1080) with an xfce session, shared over VNC and served
# by noVNC on localhost:6080. Isaac Sim still renders on the GPU (Vulkan) and presents into
# this display; NVIDIA's own Xorg captures as all black over VNC, hence Xvfb.
set -u
export DISPLAY=:1

rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
Xvfb :1 -screen 0 1920x1080x24 +extension GLX +extension RANDR -nolisten tcp &
for _ in $(seq 50); do [ -S /tmp/.X11-unix/X1 ] && break; sleep 0.2; done

dbus-launch --exit-with-session startxfce4 &
sleep 4
xfconf-query -c xfwm4 -p /general/use_compositing -s false 2>/dev/null
xset s off -dpms 2>/dev/null

x11vnc -display :1 -forever -shared -rfbauth "$HOME/.vnc/passwd" -localhost -rfbport 5900 -quiet &
websockify --web /usr/share/novnc 127.0.0.1:6080 localhost:5900 &

# If any part dies, exit so systemd restarts the whole desktop.
wait -n
kill 0
