#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the 1P1D fork contributors. Fork of MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (their recipes, harnesses and image; see fork README for credits).
# 1p1d/start-pair.sh — two-Spark prefill/decode pair orchestrator.
#
# Run ON box 1 (prefill / kv_producer). This script drives box 2 (decode /
# kv_consumer) over ssh — the GLM-kit head-orchestrates-worker pattern:
#
#   start-pair.sh start      P (box 1) -> gate -> D (box 2, ssh) -> gate -> router -> gate
#   start-pair.sh stop       router -> D (ssh) -> P
#   start-pair.sh status     P/D/router endpoints + container states
#   start-pair.sh restart    stop, then start
#   start-pair.sh dry-run    preflight + resolved commands, touches nothing
#
# Boot order is deliberate: the decode pool connects to the prefill pool's
# NIXL side channel at engine init, so P must be healthy before D launches.
# Cold start: ~11 min per box (first launch adds the ~27 GB PLE table build).
#
# ssh note (GLM-kit pattern): launch commands run detached (setsid+nohup) on
# each box — if this script or its ssh connection drops mid-launch, the
# containers keep launching; just re-run `start-pair.sh status` and then
# `start` again to re-attach the gates (start.sh archives and relaunches
# idempotently).
#
# Upstream integration: start.sh/stop.sh/download.sh are used UNMODIFIED.
# start-pair exports 1p1d/env.P (box 1) / 1p1d/env.D (box 2) into the
# launcher environment — upstream's rule "environment wins over .env" does
# the rest. Keep env.P/env.D/env.pair byte-identical on both boxes.
set -euo pipefail

MODE="${1:-start}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.pair
source "$SCRIPT_DIR/env.pair"

log()  { printf '\033[1;36m[1p1d]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[1p1d] WARN:\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[1p1d] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }
usage(){ sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ "${2:-}" = "-h" ] || [ "${2:-}" = "--help" ] && usage 0
case "$MODE" in start|stop|status|restart|dry-run) ;; *) usage 1 ;; esac

# ---------------------------------------------------------------------------
# Helpers

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null; }

dssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$D_SSH" "$@"; }

container_up() { # $1=name  — "up" if docker ps lists it as running
  local where="$1" name="$2"
  if [ "$where" = local ]; then
    docker ps --filter "name=^/$name\$" --format '{{.Status}}' | grep -q '^Up'
  else
    dssh "docker ps --filter 'name=^/$name\$' --format '{{.Status}}' | grep -q '^Up'"
  fi
}

container_status() { # $1=where $2=name — human status or "absent"
  local s
  if [ "$1" = local ]; then
    s=$(docker ps -a --filter "name=^/$2\$" --format '{{.Status}}' | head -1)
  else
    s=$(dssh "docker ps -a --filter 'name=^/$2\$' --format '{{.Status}}' | head -1")
  fi
  echo "${s:-absent}"
}

crash_tail() { # $1=where $2=name — print last error lines from container logs
  local get='docker logs --tail 400 "$NAME" 2>&1 | grep -E "ValueError|RuntimeError|TimeoutError|torch.*Error|CUDA out of memory|NIXL|nixl" | tail -5'
  if [ "$1" = local ]; then
    NAME="$2" bash -c "$get" || true
    docker inspect --format 'OOMKilled={{.State.OOMKilled}}' "$2" 2>/dev/null || true
  else
    dssh "NAME=$2; $get" || true
    dssh "docker inspect --format 'OOMKilled={{.State.OOMKilled}}' '$2'" 2>/dev/null || true
  fi
}

wait_gate() { # $1=url $2=label $3=timeout_s $4=where $5=container
  local url="$1" label="$2" timeout="$3" where="$4" name="$5" t0 now code
  t0=$(date +%s); log "gating $label: polling $url (timeout ${timeout}s)"
  while :; do
    code=$(http_code "$url")
    if [ "$code" = 200 ]; then
      log "$label READY in $(( $(date +%s) - t0 ))s"
      return 0
    fi
    if ! container_up "$where" "$name"; then
      warn "$label container '$name' is not running:"
      container_status "$where" "$name"
      crash_tail "$where" "$name"
      fail "$label died during readiness gate"
    fi
    now=$(( $(date +%s) - t0 ))
    if [ $(( now % 60 )) -lt 16 ]; then
      log "$label waiting... ${now}s (http=$code)"
    fi
    [ "$now" -ge "$timeout" ] && { warn "$label readiness timed out after ${now}s"; crash_tail "$where" "$name"; fail "$label not ready after ${timeout}s"; }
    sleep 15
  done
}

export_env_launch() { # $1=envfile $2=repodir $3=logname — detached upstream start.sh
  printf '( set -a; . "%s/1p1d/env.%s"; cd "%s" && setsid nohup ./start.sh > logs/1p1d-launch-%s.log 2>&1 < /dev/null & )' \
    "$2" "$1" "$2" "$3"
}

# ---------------------------------------------------------------------------
# Preflight (read-only; used by dry-run, and by start before launching)

preflight() {
  command -v docker >/dev/null || fail "docker not found (box 1)"
  command -v curl   >/dev/null || fail "curl not found (box 1)"
  [ -d "$P_REPO" ] && [ -f "$P_REPO/start.sh" ] || fail "P repo not found: $P_REPO"
  [ -d "$P_REPO/1p1d" ] || fail "1p1d/ missing in $P_REPO (is this a fork clone?)"
  dssh "true" || fail "ssh to box 2 ($D_SSH) failed (BatchMode key auth required)"
  dssh "[ -d '$D_REPO' ] && [ -f '$D_REPO/start.sh' ]" \
    || fail "D repo not found on box 2: $D_REPO (clone the fork there too)"
  dssh "[ -f '$D_REPO/1p1d/env.D' ]" \
    || fail "1p1d/env.D missing in $D_REPO — keep identical fork clones on both boxes"
  # engine env sanity: the two seam ends must agree on the shared knobs
  dssh "grep -q 'KV_TRANSFER_ROLE=kv_consumer' '$D_REPO/1p1d/env.D'" \
    || fail "box 2 env.D does not pin KV_TRANSFER_ROLE=kv_consumer"
  grep -q 'KV_TRANSFER_ROLE=kv_producer' "$SCRIPT_DIR/env.P" \
    || fail "env.P does not pin KV_TRANSFER_ROLE=kv_producer"
  log "preflight OK: repos, ssh, seam-role pins"
}

ensure_image_and_weights() { # $1=where $2=repodir — pull image + download.sh if needed
  local where="$1" repo="$2" cmd
  if [ "$where" = local ]; then
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "pulling $IMAGE (box 1)"; docker pull "$IMAGE"; }
    if ! (cd "$repo" && bash -c 'set -a; . ./.env 2>/dev/null; f="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4"; [ -e "$f/refs/main" ]'); then
      log "checkpoint missing on box 1 — running ./download.sh (resumable, ~99 GiB first time)"
      (cd "$repo" && ./download.sh)
    fi
  else
    dssh "docker image inspect '$IMAGE' >/dev/null 2>&1 || docker pull '$IMAGE'"
    if ! dssh "cd '$repo' && f=\"\${HF_HOME:-\$HOME/.cache/huggingface}/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4\"; [ -e \"\$f/refs/main\" ]"; then
      log "checkpoint missing on box 2 — running ./download.sh there (resumable)"
      dssh "cd '$repo' && ./download.sh"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Actions

do_start() {
  preflight
  ensure_image_and_weights local "$P_REPO"

  log "launching P (box 1, kv_producer) — detached, log: $P_REPO/logs/1p1d-launch-P.log"
  bash -c "$(export_env_launch P "$P_REPO" P)"
  wait_gate "$P_HEALTH" "P (prefill)" "$GATE_TIMEOUT_S" local "$P_CONTAINER"

  log "launching D (box 2, kv_consumer) — detached via ssh, log: $D_REPO/logs/1p1d-launch-D.log"
  dssh "mkdir -p '$D_REPO/logs'; $(export_env_launch D "$D_REPO" D)"
  wait_gate "$D_HEALTH" "D (decode)" "$GATE_TIMEOUT_S" remote "$D_CONTAINER"

  log "launching router (box 1, :$ROUTER_PORT)"
  bash "$SCRIPT_DIR/start-router.sh"
  wait_gate "$ROUTER_HEALTH/v1/models" "router" 120 local "$ROUTER_CONTAINER"

  log "PAIR UP:"
  log "  router    :$ROUTER_PORT        <- point clients here"
  log "  prefill   :$P_HEALTH (box 1, kv_producer)"
  log "  decode    :$D_HEALTH (box 2, kv_consumer)"
  log "  numbers?  bash 1p1d/report.sh   (paste the block into a GitHub issue)"
}

do_stop() {
  log "stopping router"
  docker rm -f "$ROUTER_CONTAINER" >/dev/null 2>&1 && log "  router removed" || warn "  router was not running"
  log "stopping D (box 2)"
  dssh "cd '$D_REPO' && ./stop.sh" && log "  D stopped" || warn "  D stop returned nonzero (check $D_REPO/logs/)"
  log "stopping P (box 1)"
  (cd "$P_REPO" && ./stop.sh) && log "  P stopped"
  log "PAIR DOWN"
}

do_status() {
  local up
  printf '%-22s %-28s %s\n' "node" "endpoint" "state"
  printf '%-22s %-28s %s\n' "P  kv_producer"   "$P_HEALTH"   "http=$(http_code "$P_HEALTH/health")  $(container_status local "$P_CONTAINER")"
  printf '%-22s %-28s %s\n' "D  kv_consumer"   "$D_HEALTH"   "http=$(http_code "$D_HEALTH/health")  $(container_status remote "$D_CONTAINER")"
  printf '%-22s %-28s %s\n' "router"           ":$ROUTER_PORT" "http=$(http_code "$ROUTER_HEALTH/v1/models")  $(container_status local "$ROUTER_CONTAINER")"
  if [ "$(http_code "$ROUTER_HEALTH/v1/router/stats")" = 200 ]; then
    echo; curl -s --max-time 5 "$ROUTER_HEALTH/v1/router/stats"; echo
  fi
}

case "$MODE" in
  start)   do_start ;;
  stop)    do_stop ;;
  status)  do_status ;;
  restart) do_stop; echo; do_start ;;
  dry-run)
    preflight
    echo
    log "DRY RUN — resolved commands (nothing executed):"
    echo "  [box 1] $(export_env_launch P "$P_REPO" P)"
    echo "  [gate ] curl $P_HEALTH/health   until 200 (<= ${GATE_TIMEOUT_S}s)"
    echo "  [box 2] ssh $D_SSH \"mkdir -p '$D_REPO/logs'; $(export_env_launch D "$D_REPO" D)\""
    echo "  [gate ] curl $D_HEALTH/health   until 200 (<= ${GATE_TIMEOUT_S}s)"
    echo "  [box 1] bash $SCRIPT_DIR/start-router.sh"
    echo "  [gate ] curl $ROUTER_HEALTH/v1/models until 200"
    echo
    log "read-only checks:"
    docker image inspect "$IMAGE" >/dev/null 2>&1 && log "box 1 image present: $IMAGE" || warn "box 1 image missing: $IMAGE (start would pull)"
    dssh "docker image inspect '$IMAGE' >/dev/null 2>&1" && log "box 2 image present: $IMAGE" || warn "box 2 image missing: $IMAGE (start would pull)"
    (cd "$P_REPO" && [ -e "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/refs/main" ] && log "box 1 checkpoint snapshot present" || warn "box 1 checkpoint missing (start runs ./download.sh)")
    dssh "[ -e \"\${HF_HOME:-\$HOME/.cache/huggingface}/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/refs/main\" ]" && log "box 2 checkpoint snapshot present" || warn "box 2 checkpoint missing (start runs ./download.sh)"
    log "dry-run complete"
    ;;
esac
