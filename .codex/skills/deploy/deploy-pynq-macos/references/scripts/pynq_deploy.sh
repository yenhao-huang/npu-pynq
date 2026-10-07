#!/bin/bash
# Deploy NPU overlays and compiled models to the PYNQ-Z1 from macOS, and run
# them there. The macOS counterpart of the exp-board workflow's board job.
#
#   pynq_deploy.sh [--dry-run] [--host IP] [--user NAME] <command> [options]
#
# Commands
#   preflight                   network route, SSH key login, board venv and sudo rule
#   overlay  --run ID | --release TAG | --dir DIR
#                               fetch npu_matrix.bit/.hwh into build/deploy/overlay
#   export                      build SmolLM2, Qwen3 and ResNet-18 packages on this Mac
#   run      [--script PATH] [--timeout MIN]
#                               copy overlay + packages + src/ + exp/ to the board,
#                               run PATH there (default: the #119 model run), fetch results
#   all      (--run ID | --release TAG | --dir DIR) [run options]
#                               preflight, overlay, export (if missing), run
#
# Works with the macOS system bash (3.2). Vivado stays on the self-hosted
# runner: overlays come from its artifacts or a Release.
set -euo pipefail

HOST=${PYNQ_BOARD_HOST:-192.168.2.99}
USER_NAME=${PYNQ_BOARD_USER:-xilinx}
DRY=0
REPO=$(cd "$(dirname "$0")/../../../../../.." && pwd)
DEPLOY="$REPO/build/deploy"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
          -o StrictHostKeyChecking=accept-new)
BOARD_PY=/usr/local/share/pynq-venv/bin/python3

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
run() {
    if [ "$DRY" = 1 ]; then printf 'dry-run: %s\n' "$*"; else "$@"; fi
}
target() { printf '%s@%s' "$USER_NAME" "$HOST"; }

cmd_preflight() {
    for tool in ssh scp tar route; do
        command -v "$tool" >/dev/null || die "$tool is not on PATH"
    done
    say "board      $(target)"
    # A direct cable means the route to the board leaves through a non-default
    # interface with an address on the board's /24.
    local iface addr
    iface=$(route -n get "$HOST" 2>/dev/null | awk '/interface:/ {print $2}')
    addr=$(ipconfig getifaddr "${iface:-none}" 2>/dev/null || true)
    say "route      ${iface:-none} (local address ${addr:-none})"
    if [ "${addr%.*}" != "${HOST%.*}" ]; then
        cat >&2 <<EOF
error: no interface on ${HOST%.*}.0/24, so ${HOST} is not on a direct link.
  Connect the board's Ethernet port to this Mac (USB/Thunderbolt adapter), then in
  System Settings > Network > <adapter> > Details > TCP/IP set
  Configure IPv4: Manually, IP address ${HOST%.*}.1, Subnet mask 255.255.255.0.
EOF
        [ "$DRY" = 1 ] && return 0
        exit 1
    fi
    [ "$DRY" = 1 ] && { say "dry-run: ssh $(target) <board checks>"; return 0; }
    if ! ssh "${SSH_OPTS[@]}" "$(target)" true 2>/dev/null; then
        die "key-based SSH to $(target) failed; install a key once with: ssh-copy-id $(target)"
    fi
    ssh "${SSH_OPTS[@]}" "$(target)" bash -s <<EOF
set -u
test -r /etc/profile.d/xrt_setup.sh || { echo "missing /etc/profile.d/xrt_setup.sh"; exit 1; }
test -r /etc/profile.d/pynq_venv.sh || { echo "missing /etc/profile.d/pynq_venv.sh"; exit 1; }
test -x $BOARD_PY || { echo "missing the PYNQ venv python3"; exit 1; }
sudo -n -l $BOARD_PY >/dev/null 2>&1 || { echo "\$(id -un) cannot run sudo -n $BOARD_PY; add a sudoers rule"; exit 1; }
avail=\$(df -Pk "\$HOME" | awk 'NR==2 {print \$4}')
test "\$avail" -gt 1048576 || { echo "less than 1 GiB free in \$HOME"; exit 1; }
echo "board ready: \$(uname -m), \$(grep MemTotal /proc/meminfo | tr -s ' '), \${avail} KiB free"
EOF
    say "PASS: preflight"
}

cmd_overlay() {
    local src="" kind=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --run|--release|--dir) kind=${1#--}; src=$2; shift 2 ;;
            *) die "overlay: unknown option $1" ;;
        esac
    done
    [ -n "$kind" ] || die "overlay needs --run ID, --release TAG or --dir DIR"
    local out="$DEPLOY/overlay/artifacts" tmp="$DEPLOY/overlay-download"
    run rm -rf "$DEPLOY/overlay" "$tmp"
    run mkdir -p "$out" "$tmp"
    case "$kind" in
        run) run gh run download "$src" -R yenhao-huang/npu-pynq -p 'exp-overlay-*' -D "$tmp" ;;
        release) run gh release download "$src" -R yenhao-huang/npu-pynq -p 'npu_matrix.*' -D "$tmp" ;;
        dir) run cp -R "$src"/. "$tmp"/ ;;
    esac
    [ "$DRY" = 1 ] && return 0
    local bit hwh
    bit=$(find "$tmp" -name 'npu_matrix.bit' | head -1)
    hwh=$(find "$tmp" -name 'npu_matrix.hwh' | head -1)
    [ -n "$bit" ] && [ -n "$hwh" ] || die "no npu_matrix.bit/.hwh pair in the $kind source"
    cp "$bit" "$hwh" "$out/"
    find "$tmp" -name 'npu_matrix.manifest.json' -exec cp {} "$out/" \; 2>/dev/null || true
    rm -rf "$tmp"
    say "overlay    $out ($(shasum -a 256 "$out/npu_matrix.bit" | cut -c1-16)...)"
}

cmd_export() {
    local out="$DEPLOY/export"
    run rm -rf "$out"
    run bash -c "cd '$REPO' && bash exp/1007_sw_stack/export_all.sh '$out'"
}

cmd_run() {
    local script=exp/1007_sw_stack/board/run_models.sh timeout=300
    while [ $# -gt 0 ]; do
        case "$1" in
            --script) script=$2; shift 2 ;;
            --timeout) timeout=$2; shift 2 ;;
            *) die "run: unknown option $1" ;;
        esac
    done
    [ "$DRY" = 1 ] || [ -f "$DEPLOY/overlay/artifacts/npu_matrix.bit" ] || die "no overlay; run the overlay command first"
    [ "$DRY" = 1 ] || [ -d "$DEPLOY/export" ] || die "no packages; run the export command first"
    local id remote payload results
    id="mac-$(date +%Y%m%d-%H%M%S)"
    remote="/home/$USER_NAME/npu_exp/$id"
    payload="$DEPLOY/payload-$id.tgz"
    results="$DEPLOY/results/$id"
    say "job        $id -> $(target):$remote, script $script"
    # Same layout as the exp-board workflow: src/, exp/, overlay/, export/.
    run tar -czf "$payload" -C "$REPO" src exp -C "$DEPLOY" overlay export
    run ssh "${SSH_OPTS[@]}" "$(target)" "mkdir -p '$remote'"
    run scp "${SSH_OPTS[@]}" "$payload" "$(target):$remote/payload.tgz"
    # Detached on the board: a dropped SSH session costs one poll, not the run.
    local job encoded
    job=$(cat <<JOB
set -u
cd '$remote'
tar -xzf payload.tgz && rm payload.tgz
mkdir -p results
code=0
bash '$script' '$remote' > results/board.log 2>&1 || code=\$?
sudo -n chmod -R a+rwX results || true
echo \$code > exit.tmp && mv exit.tmp exit
JOB
)
    # Base64 keeps every \$ for the board; nothing is expanded on the way.
    encoded=$(printf '%s\n' "$job" | base64 | tr -d '\n')
    run ssh "${SSH_OPTS[@]}" "$(target)" "cd '$remote' && echo $encoded | base64 -d > job.sh && nohup setsid bash job.sh > job.log 2>&1 < /dev/null & echo started"
    [ "$DRY" = 1 ] && { say "dry-run: poll $remote/exit for up to $timeout min, fetch results to $results"; return 0; }
    local deadline status="" last=0
    deadline=$(( $(date +%s) + timeout * 60 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        sleep 20
        status=$(ssh "${SSH_OPTS[@]}" "$(target)" "cat '$remote/exit' 2>/dev/null || echo running" 2>/dev/null || echo unreachable)
        [ "$status" = running ] || [ "$status" = unreachable ] || break
        if [ $(( $(date +%s) - last )) -ge 120 ]; then
            last=$(date +%s)
            ssh "${SSH_OPTS[@]}" "$(target)" "tail -n 3 '$remote/results/board.log'" 2>/dev/null | sed 's/^/  board: /' || true
        fi
    done
    mkdir -p "$results"
    ssh "${SSH_OPTS[@]}" "$(target)" "cd '$remote' && tar -czf - results" | tar -xzf - -C "$results"
    rm -f "$payload"
    tail -n 40 "$results/results/board.log" || true
    say "results    $results/results"
    case "$status" in
        0) say "PASS: board job" ;;
        running|unreachable|"") die "board job did not finish within $timeout minutes" ;;
        *) die "board job failed (exit $status)" ;;
    esac
}

cmd_all() {
    local overlay_args=() run_args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --run|--release|--dir) overlay_args=("$1" "$2"); shift 2 ;;
            --script|--timeout) run_args+=("$1" "$2"); shift 2 ;;
            *) die "all: unknown option $1" ;;
        esac
    done
    cmd_preflight
    if [ ${#overlay_args[@]} -gt 0 ]; then cmd_overlay "${overlay_args[@]}"; fi
    if [ "$DRY" = 1 ] || [ ! -d "$DEPLOY/export" ]; then cmd_export; fi
    cmd_run ${run_args[@]+"${run_args[@]}"}
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --host) HOST=$2; shift 2 ;;
        --user) USER_NAME=$2; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { sed -n '2,21p' "$0"; exit 2; }
command=$1; shift
case "$command" in
    preflight|overlay|export|run|all) "cmd_$command" "$@" ;;
    *) die "unknown command $command" ;;
esac
