#!/usr/bin/env bash
# Run a release archive on the PYNQ-Z1 and bring back what it printed.
#
#   .github/release/board_check.sh <archive.tar.gz> <evidence.json> <src.runtime.cli arguments...>
#
# The archive is what src/deploy/deploy.sh ARCHIVE=... wrote: one directory,
# <name>/, the same tree the example guides send. It replaces
# ~/jupyter_notebooks/<name> on the board, and there the guide's own command
# runs as root with the PYNQ environment:
#
#   sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli <arguments>
#
# The arguments must make the runtime write board.json in that directory
# (classifier --evidence board.json, llm --json board.json); it is copied to
# <evidence.json>. An inference can outlast an idle ssh session, so the run is
# detached on the board and polled with short connections.
#
# BOARD overrides the target (default xilinx@192.168.2.99); TIMEOUT_MINUTES
# bounds the run (default 180).
set -euo pipefail

archive=${1:?usage: board_check.sh <archive.tar.gz> <evidence.json> <arguments...>}
evidence=${2:?usage: board_check.sh <archive.tar.gz> <evidence.json> <arguments...>}
shift 2
board=${BOARD:-xilinx@192.168.2.99}
timeout=${TIMEOUT_MINUTES:-180}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

name=$(tar -tzf "$archive" | sed -n '1s|/.*||p')
case "$name" in
  ''|.|..|*[!A-Za-z0-9._-]*) echo "archive must hold one named directory, got '$name'" >&2; exit 1 ;;
esac
remote="jupyter_notebooks/$name"
command=$(printf '%q ' "$@")

echo "$archive -> $board:$remote"
# The previous run left root-owned files (the runtime builds its libraries as
# root); the chmod the sudo rule allows makes them removable.
ssh "${ssh_opts[@]}" "$board" "if [ -e '$remote' ]; then sudo -n chmod -R a+rwX '$remote'; fi; rm -rf '$remote' && mkdir -p jupyter_notebooks && tar -xzmf - -C jupyter_notebooks" < "$archive"

job="set -u
cd \"\$HOME/$remote\"
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
code=0
sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B -m src.runtime.cli $command || code=\$?
if [ -f board.json ]; then sudo -n chmod 0644 board.json || code=1; fi
echo \$code > exit.tmp && mv exit.tmp exit"
encoded=$(printf '%s\n' "$job" | base64 | tr -d '\n')
ssh "${ssh_opts[@]}" "$board" "cd '$remote' && echo $encoded | base64 -d > job.sh && { nohup setsid bash job.sh > board.log 2>&1 < /dev/null & } && echo started"

deadline=$(( $(date +%s) + timeout * 60 ))
status=
failures=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 30
  if ! answer=$(ssh "${ssh_opts[@]}" "$board" "cat '$remote/exit' 2>/dev/null || echo running"); then
    failures=$((failures + 1))
    echo "poll failed; $failures in a row"
    [ "$failures" -lt 20 ] || { echo "lost the board for $failures polls" >&2; exit 1; }
    continue
  fi
  failures=0
  if [ "$answer" != running ]; then status=$answer; break; fi
done

ssh "${ssh_opts[@]}" "$board" "tail -n 40 '$remote/board.log'" || true
[ -n "$status" ] || { echo "the board run did not finish within $timeout minutes" >&2; exit 1; }
[ "$status" = 0 ] || { echo "the board run failed (exit $status)" >&2; exit 1; }
mkdir -p "$(dirname "$evidence")"
scp "${ssh_opts[@]}" "$board:$remote/board.json" "$evidence"
echo "PASS: $name on $board; evidence at $evidence"
