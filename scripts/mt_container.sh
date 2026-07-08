#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${MULTITI_CONTAINER:-eeg-report-audit}"
WORKDIR_IN_CONTAINER="${MULTITI_WORKDIR_IN_CONTAINER:-/workspace/eeg_report_multiagent_v1}"
HOST_UID="${MULTITI_HOST_UID:-$(id -u)}"
HOST_GID="${MULTITI_HOST_GID:-$(id -g)}"
USER_SPEC="${HOST_UID}:${HOST_GID}"

usage() {
  cat <<EOF
Usage: scripts/mt_container.sh <command> [args...]

Commands:
  status                 Show container/workdir/python status.
  shell                  Open an interactive shell as host uid/gid.
  shell-root             Open an interactive shell as root.
  run <cmd>              Run a command in the repo as host uid/gid.
  run-env <cmd>          Run a command after sourcing .env as host uid/gid.
  run-root <cmd>         Run a command in the repo as root.
  pytest [args...]       Run pytest in the container.
  compile                Compile src/eeg_report_multiagent.
  fix-perms [paths...]   Chown generated paths back to host uid/gid. Defaults to common artifact/cache paths.

Env overrides:
  MULTITI_CONTAINER              default: eeg-report-audit
  MULTITI_WORKDIR_IN_CONTAINER   default: /workspace/eeg_report_multiagent_v1
  MULTITI_HOST_UID/GID           default: current host uid/gid
EOF
}

check_container() {
  docker inspect "$CONTAINER" >/dev/null 2>&1 || {
    echo "Container not found: $CONTAINER" >&2
    echo "Start it first or set MULTITI_CONTAINER." >&2
    exit 2
  }
}

run_bash() {
  local user_flag="$1"
  shift
  docker exec -i ${user_flag:+-u "$user_flag"} "$CONTAINER" bash -lc "cd '$WORKDIR_IN_CONTAINER' && $*"
}

case "${1:-}" in
  status)
    check_container
    docker ps --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Mounts}}'
    docker exec -i "$CONTAINER" bash -lc "cd '$WORKDIR_IN_CONTAINER' && pwd && python3 --version && python3 -m pytest --version && id && test -f .env && echo '.env: present' || echo '.env: missing'"
    ;;
  shell)
    check_container
    docker exec -it -u "$USER_SPEC" "$CONTAINER" bash -lc "cd '$WORKDIR_IN_CONTAINER' && exec bash"
    ;;
  shell-root)
    check_container
    docker exec -it "$CONTAINER" bash -lc "cd '$WORKDIR_IN_CONTAINER' && exec bash"
    ;;
  run)
    check_container
    shift
    if [ "$#" -eq 0 ]; then usage; exit 2; fi
    run_bash "$USER_SPEC" "$*"
    ;;
  run-env)
    check_container
    shift
    if [ "$#" -eq 0 ]; then usage; exit 2; fi
    run_bash "$USER_SPEC" "set -a; [ -f .env ] && source .env; set +a; $*"
    ;;
  run-root)
    check_container
    shift
    if [ "$#" -eq 0 ]; then usage; exit 2; fi
    run_bash "" "$*"
    ;;
  pytest)
    check_container
    shift || true
    run_bash "$USER_SPEC" "python3 -m pytest ${*:-}"
    ;;
  compile)
    check_container
    run_bash "$USER_SPEC" "python3 -m compileall -q src/eeg_report_multiagent"
    ;;
  fix-perms)
    check_container
    shift || true
    if [ "$#" -eq 0 ]; then
      set -- artifacts analysis_artifacts .pytest_cache scripts/__pycache__ src/eeg_report_multiagent/__pycache__ tests/__pycache__
    fi
    quoted_paths=""
    for p in "$@"; do
      quoted_paths="$quoted_paths '$WORKDIR_IN_CONTAINER/$p'"
    done
    docker exec -i "$CONTAINER" bash -lc "chown -R '$HOST_UID:$HOST_GID' $quoted_paths 2>/dev/null || true"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 2
    ;;
esac
