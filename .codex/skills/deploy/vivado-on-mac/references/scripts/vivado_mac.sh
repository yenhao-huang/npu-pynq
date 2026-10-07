#!/bin/bash
# AMD Vivado (Linux x86-64) on an Apple silicon Mac through Docker Desktop +
# Rosetta, with the GUI shown by macOS Screen Sharing (VNC).
#
#   vivado_mac.sh check                      Docker engine, amd64 emulation, memory, disk, image, Vivado
#   vivado_mac.sh build-image                build the npu-vivado image (no Vivado inside)
#   vivado_mac.sh install-gui INSTALLER.bin  run AMD's Linux installer in a VNC desktop;
#                                            install to /tools/Xilinx (kept in a Docker volume)
#   vivado_mac.sh gui [--fresh]              open the Vivado UI on the npu_matrix project
#                                            (created with build_overlay.tcl --elaborate-only)
#   vivado_mac.sh build [TCL ARGS]           full batch build: BIT/HWH in build/vivado/<design>/artifacts
#   vivado_mac.sh deploy [RUN ARGS]          send that overlay + packages to the PYNQ-Z1 and run
#   vivado_mac.sh stop                       stop the VNC container
set -euo pipefail
# Docker Desktop's helpers (docker-credential-desktop, ...) live in the app bundle.
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

REPO=$(cd "$(dirname "$0")/../../../../../.." && pwd)
SKILL=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=npu-vivado:22.04
VOLUME=npu-vivado-tools
NAME=npu-vivado-gui
PORT=${VIVADO_VNC_PORT:-5901}
CONFIG="$HOME/.config/npu-vivado"
DEPLOY="$REPO/.codex/skills/deploy/deploy-pynq-macos/references/scripts/pynq_deploy.sh"
BUILD_DIR=build/vivado/npu_matrix_16x16

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }

# macOS has no timeout(1), and the docker CLI ignores SIGALRM: kill it instead.
with_timeout() {
    local seconds=$1; shift
    "$@" &
    local pid=$!
    ( sleep "$seconds"; kill -9 "$pid" 2>/dev/null ) &
    local watchdog=$!
    local rc=0
    wait "$pid" || rc=$?
    kill "$watchdog" 2>/dev/null || true
    return "$rc"
}

engine() {
    with_timeout 10 docker info --format '{{.NCPU}} {{.MemTotal}}' 2>/dev/null | grep -v '^0 ' ||
        die "Docker engine is not running: start Docker Desktop (Settings > General: 'Use Rosetta for x86_64/amd64 emulation on Apple Silicon' on)"
}

vnc_password() {
    mkdir -p "$CONFIG"
    if [ ! -s "$CONFIG/vnc-password" ]; then
        LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 8 > "$CONFIG/vnc-password"
        chmod 600 "$CONFIG/vnc-password"
    fi
    cat "$CONFIG/vnc-password"
}

vivado_installed() {
    docker run --rm --platform linux/amd64 -v "$VOLUME:/tools/Xilinx" --entrypoint bash "$IMAGE" -c \
        'ls -d /tools/Xilinx/*/Vivado/bin/vivado /tools/Xilinx/Vivado/*/bin/vivado 2>/dev/null | sort | tail -1'
}

cmd_check() {
    local info cpus mem
    info=$(engine)
    cpus=${info%% *}; mem=${info#* }
    say "docker     $(docker version --format '{{.Server.Version}}') ${cpus} CPUs, $((mem / 1024 / 1024 / 1024)) GiB"
    [ $((mem / 1024 / 1024 / 1024)) -ge 8 ] || say "warning: give Docker >= 8 GiB (Settings > Resources) for synthesis"
    local arch
    arch=$(docker run --rm --platform linux/amd64 ubuntu:22.04 uname -m 2>&1 | tail -1)
    [ "$arch" = x86_64 ] || die "amd64 containers do not run ($arch); enable Rosetta in Docker Desktop"
    say "amd64      $arch (Rosetta)"
    say "disk       $(df -h "$HOME" | awk 'NR==2 {print $4}') free (Vivado needs ~60-100 GB in the volume)"
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then say "image      $IMAGE"; else say "image      missing: vivado_mac.sh build-image"; return 0; fi
    local v
    v=$(vivado_installed || true)
    if [ -n "$v" ]; then say "vivado     $v (volume $VOLUME)"; else say "vivado     not installed: vivado_mac.sh install-gui <AMD Linux installer .bin>"; fi
}

cmd_build_image() {
    engine >/dev/null
    docker build --platform linux/amd64 -t "$IMAGE" "$SKILL/docker"
}

start_desktop() {
    # $1: command the desktop runs; remaining args: extra docker run options.
    local app=$1; shift
    local pass
    pass=$(vnc_password)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --platform linux/amd64 --shm-size=2g \
        -p "127.0.0.1:$PORT:5901" -e VNC_PASSWORD="$pass" -e DESKTOP_APP="$app" \
        -v "$VOLUME:/tools/Xilinx" -v "$REPO:/work" "$@" "$IMAGE" desktop >/dev/null
    local i
    for i in $(seq 1 60); do
        docker logs "$NAME" 2>&1 | grep -q -i "vnc" && nc -z 127.0.0.1 "$PORT" 2>/dev/null && break
        sleep 1
    done
    say "desktop    vnc://localhost:$PORT (password in $CONFIG/vnc-password)"
    open "vnc://:$pass@localhost:$PORT"
}

cmd_install_gui() {
    local installer=${1:?usage: vivado_mac.sh install-gui <AMD Linux self-extracting web installer .bin>}
    [ -f "$installer" ] || die "no such file: $installer"
    engine >/dev/null
    docker image inspect "$IMAGE" >/dev/null 2>&1 || cmd_build_image
    local dir base
    dir=$(cd "$(dirname "$installer")" && pwd); base=$(basename "$installer")
    say "In the installer: sign in with your AMD account, choose Vivado (ML Standard is enough"
    say "for the xc7z020), select only Zynq-7000 devices, and install to /tools/Xilinx."
    start_desktop "bash /installer/$base" -v "$dir:/installer:ro"
}

ensure_project() {
    local xpr="$REPO/$BUILD_DIR/project/npu_matrix.xpr"
    if [ -f "$xpr" ] && [ "${1:-}" != --fresh ]; then return 0; fi
    say "project    creating with build_overlay.tcl --elaborate-only"
    docker run --rm --platform linux/amd64 -v "$VOLUME:/tools/Xilinx" -v "$REPO:/work" "$IMAGE" \
        bash -lc 'vivado -mode batch -nojournal -nolog -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl -tclargs --elaborate-only'
}

cmd_gui() {
    engine >/dev/null
    [ -n "$(vivado_installed || true)" ] || die "Vivado is not installed: vivado_mac.sh install-gui <installer.bin>"
    ensure_project "${1:-}"
    start_desktop "vivado -nojournal -nolog /work/$BUILD_DIR/project/npu_matrix.xpr"
}

cmd_build() {
    engine >/dev/null
    [ -n "$(vivado_installed || true)" ] || die "Vivado is not installed"
    if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
        die "build_overlay.tcl only publishes artifacts from a clean tree; commit or stash first"
    fi
    docker run --rm --platform linux/amd64 -v "$VOLUME:/tools/Xilinx" -v "$REPO:/work" "$IMAGE" \
        bash -lc "vivado -mode batch -nojournal -nolog -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl $([ $# -gt 0 ] && printf -- '-tclargs %s' "$*")"
    ls -la "$REPO/$BUILD_DIR/artifacts"
}

cmd_deploy() {
    local artifacts="$REPO/$BUILD_DIR/artifacts"
    [ -f "$artifacts/npu_matrix.bit" ] || die "no overlay in $artifacts: run vivado_mac.sh build"
    "$DEPLOY" preflight
    "$DEPLOY" overlay --dir "$artifacts"
    [ -d "$REPO/build/deploy/export" ] || "$DEPLOY" export
    "$DEPLOY" run "$@"
}

cmd_stop() {
    docker rm -f "$NAME" >/dev/null 2>&1 && say "stopped $NAME" || say "$NAME is not running"
}

[ $# -gt 0 ] || { sed -n '2,15p' "$0"; exit 2; }
command=$1; shift
case "$command" in
    check) cmd_check ;;
    build-image) cmd_build_image ;;
    install-gui) cmd_install_gui "$@" ;;
    gui) cmd_gui "$@" ;;
    build) cmd_build "$@" ;;
    deploy) cmd_deploy "$@" ;;
    stop) cmd_stop ;;
    -h|--help) sed -n '2,15p' "$0" ;;
    *) die "unknown command $command" ;;
esac
