#!/bin/bash
# desktop: VNC desktop on :1 (port 5901) running $DESKTOP_APP; anything else runs as is.
set -e
VIVADO_SETTINGS=$(ls -d /tools/Xilinx/*/Vivado/settings64.sh /tools/Xilinx/Vivado/*/settings64.sh 2>/dev/null | sort | tail -1 || true)
[ -n "$VIVADO_SETTINGS" ] && source "$VIVADO_SETTINGS"
if [ "${1:-}" = desktop ]; then
    shift
    mkdir -p ~/.vnc
    printf '%s\n' "${VNC_PASSWORD:?VNC_PASSWORD is required}" | vncpasswd -f > ~/.vnc/passwd
    chmod 600 ~/.vnc/passwd
    cat > ~/.vnc/xstartup <<XS
#!/bin/sh
xsetroot -solid '#2b2b2b'
openbox-session &
xterm -geometry 100x30+10+10 -title shell &
${DESKTOP_APP:-xterm}
XS
    chmod +x ~/.vnc/xstartup
    vncserver :1 -geometry "${VNC_GEOMETRY:-1680x1050}" -depth 24 -localhost no -SecurityTypes VncAuth
    exec tail -F ~/.vnc/*.log
fi
exec "$@"
