#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the 1P1D fork contributors. Fork of MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (their recipes, harnesses and image; see fork README for credits).
# 1p1d/start-router.sh — PD router container on box 1 (runs WITH the P pool).
#
# The router runs IN the required vLLM image (fastapi/uvicorn/httpx ship in
# it — zero extra deps), on the host network, restart=unless-stopped, so it
# survives reboots and comes back as soon as docker does. Any GATTLING_*
# variable exported in the caller's environment is passed into the container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.pair
source "$SCRIPT_DIR/env.pair"

log()   { printf '\033[1;36m[1p1d]\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m[1p1d] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || fail "docker not found"
[ -f "$SCRIPT_DIR/pd_router.py" ] || fail "$SCRIPT_DIR/pd_router.py missing"

# Passthrough caller GATTLING_* knobs first; our three pins go last (win).
EXTRA_ENV=()
while IFS='=' read -r k v; do EXTRA_ENV+=("-e" "$k=$v"); done \
  < <(env | grep -E '^GATTLING_[A-Z_0-9]+=' || true)

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
docker rm -f "$ROUTER_CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$ROUTER_CONTAINER" \
  --restart unless-stopped --network host \
  --entrypoint python3 \
  -w /vllm-workspace \
  -v "$SCRIPT_DIR/pd_router.py:/r/pd_router.py:ro" \
  "${EXTRA_ENV[@]}" \
  -e "GATTLING_P_URL=$P_HEALTH" \
  -e "GATTLING_D_URL=$D_FABRIC_URL" \
  -e "GATTLING_P_HOST=$P_SIDECHANNEL_IP" \
  "$IMAGE" /r/pd_router.py

log "router '$ROUTER_CONTAINER' launching on :$ROUTER_PORT (logs: docker logs -f $ROUTER_CONTAINER)"
