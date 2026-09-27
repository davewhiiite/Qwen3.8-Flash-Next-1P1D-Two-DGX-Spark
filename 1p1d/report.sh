#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the 1P1D fork contributors. Fork of MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (their recipes, harnesses and image; see fork README for credits).
# 1p1d/report.sh — one-paste feedback dump ("post this as a GitHub issue").
#
# Requires: pair UP (start-pair.sh start) and an IDLE rig (any other traffic
# lands in the same engine steps and pollutes the numbers).
#
#   bash 1p1d/report.sh                 # run mixed.py via the router, print the issue block
#   BENCH_SKIP=1 bash 1p1d/report.sh    # fingerprint only, no bench run
#
# What you get: a fenced markdown block with your mixed.py money-shot cells
# (ITL p50/p95/p99 during the 64K prefill window, in-window aggregate tok/s),
# the rig fingerprint (image digest, KV pools, context, MTP, seam env), and
# our published reference row to compare against. Paste it into a new issue —
# that IS the feedback loop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.pair
source "$SCRIPT_DIR/env.pair"

log()  { printf '\033[1;36m[1p1d]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[1p1d] WARN:\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[1p1d] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

ISSUES_URL="${REPORT_ISSUES_URL:-https://github.com/davewhiiite/Qwen3.8-Flash-Next-1P1D-Two-DGX-Spark/issues}"
MODEL_ID_DEFAULT="models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4"

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null; }

[ "$(http_code "$P_HEALTH/health")" = 200 ] || fail "P not healthy at $P_HEALTH — run start-pair.sh start first"
[ "$(http_code "$D_HEALTH/health")" = 200 ] || fail "D not healthy at $D_HEALTH — run start-pair.sh start first"

# ---- fingerprint ----------------------------------------------------------
# (subshell: env.P/env.D are engine knobs, don't leak them)
snap() { ( set -a; . "$SCRIPT_DIR/env.P"; . "$SCRIPT_DIR/env.D"
  printf 'IMAGE=%s\nMODEL_LEN=%s\nMTP=%s\nKV_DTYPE=%s\nSSM_DTYPE=%s\nMAX_SEQS=%s\nYARN=%s\nMAX_BATCHED=%s\n' \
    "$IMAGE" "$MAX_MODEL_LEN" "$MTP_NUM_SPECULATIVE_TOKENS" "$KV_CACHE_DTYPE" \
    "$MAMBA_SSM_CACHE_DTYPE" "$MAX_NUM_SEQS" "$YARN" "$MAX_NUM_BATCHED_TOKENS" ); }
eval "$(snap | sed 's/^/export /')"

img_ref=$(docker inspect --format '{{with index .RepoDigests 0}}{{.}}{{else}}{{.Id}}{{end}}' "$IMAGE" 2>/dev/null | tr -d '\n')
img_ref=${img_ref:-"$IMAGE (local)"}
kv_p()   { docker logs "$P_CONTAINER" 2>&1 | grep -oE 'Available KV cache memory: [0-9.]+ [GM]iB' | tail -1; }
kv_d()   { ssh -o BatchMode=yes "$D_SSH" "docker logs $D_CONTAINER 2>&1 | grep -oE 'Available KV cache memory: [0-9.]+ [GM]iB' | tail -1"; }
ckpt() { local d="${HF_HOME:-$HOME/.cache/huggingface}/hub/$MODEL_ID_DEFAULT"; ls -d "$d/snapshots/"* 2>/dev/null | tail -1 | xargs -r basename; }
ckpt_d() { ssh -o BatchMode=yes "$D_SSH" "ls -d \${HF_HOME:-\$HOME/.cache/huggingface}/hub/$MODEL_ID_DEFAULT/snapshots/* 2>/dev/null | tail -1 | xargs -r basename"; }

# ---- bench ----------------------------------------------------------------
OUT="logs/1p1d-report.jsonl"
mkdir -p logs
if [ "${BENCH_SKIP:-0}" != 1 ]; then
  log "running 1p1d/bench/mixed.py against the router (2 decoders + 64K prefill; ~2-4 min, rig must be idle)"
  ( cd "$P_REPO" && python3 1p1d/bench/mixed.py --tag report --out logs/1p1d-report.jsonl --note "report.sh $(date -Iseconds)" )
else
  log "BENCH_SKIP=1 — reusing last row in $OUT if present"
fi
[ -f "$OUT" ] || fail "no bench row at $OUT (run without BENCH_SKIP)"
ROW=$(tail -1 "$OUT")

# ---- format ---------------------------------------------------------------
export REPORT_ROW="$ROW" REPORT_IMG="$img_ref" REPORT_KV_P="$(kv_p)" REPORT_KV_D="$(kv_d)" \
       REPORT_CKPT_P="$(ckpt)" REPORT_CKPT_D="$(ckpt_d)" REPORT_P_URL="$P_HEALTH" \
       REPORT_D_URL="$D_HEALTH" REPORT_ROUTER_PORT="$ROUTER_PORT" REPORT_ISSUES_URL="$ISSUES_URL"

python3 <<'PY'
import json, sys, os, datetime
row = json.loads(os.environ["REPORT_ROW"])
img = os.environ["REPORT_IMG"]; kv_p = os.environ["REPORT_KV_P"]; kv_d = os.environ["REPORT_KV_D"]
ckpt = os.environ["REPORT_CKPT_P"]; ckpt_d = os.environ["REPORT_CKPT_D"]
p_url = os.environ["REPORT_P_URL"]; d_url = os.environ["REPORT_D_URL"]
rport = os.environ["REPORT_ROUTER_PORT"]; issues = os.environ["REPORT_ISSUES_URL"]
snap = {k: os.environ.get(k, "") for k in ("IMAGE", "MODEL_LEN", "MTP", "KV_DTYPE",
       "SSM_DTYPE", "MAX_SEQS", "YARN", "MAX_BATCHED")}

dur = [s for s in (row.get("itl_during_prefill") or []) if s]
def med(key):
    vals = [s[key] for s in dur if s] or [None]
    return sorted(vals)[len(vals)//2]
p50, p95, p99 = med("p50_ms"), med("p95_ms"), med("p99_ms")
tps = row.get("decode_tps_in_window"); ttft = row.get("big_ttft_s")
err = [e for e in (row.get("errors") or []) if e]

print()
print("=" * 78)
print("ONE-PASTE GITHUB ISSUE — copy everything between the fences")
print("=" * 78)
print("```markdown")
print("## 1P1D results report")
print()
print(f"- date: {datetime.date.today().isoformat()}")
print(f"- rig: 2x NVIDIA DGX Spark (GB10), Qwen3.8-Flash-Next-NVFP4")
print(f"- image: `{img}`")
print(f"- checkpoint snapshot: `{ckpt}` (P) / `{ckpt_d}` (D)")
print(f"- engine: MAX_MODEL_LEN={snap.get('MODEL_LEN')} · MTP-{snap.get('MTP')} · KV {snap.get('KV_DTYPE')} · "
      f"SSM {snap.get('SSM_DTYPE')} · MAX_NUM_SEQS={snap.get('MAX_SEQS')} · YaRN={snap.get('YARN')} · "
      f"MAX_NUM_BATCHED_TOKENS={snap.get('MAX_BATCHED')}")
print(f"- seam: NixlConnector pull-mode, UCX RC over roceP2p1s0f1 (soft-RoCE), `--device /dev/infiniband`")
print(f"- KV pools: P \"{kv_p or 'n/a'}\" · D \"{kv_d or 'n/a'}\"")
print(f"- endpoints: prefill {p_url} · decode {d_url} · router :{rport}")
print()
print("### `1p1d/bench/mixed.py` — 2 decoders + one 64K prefill")
print()
print("| metric | this run | our published C (reference) |")
print("|---|---|---|")
print(f"| ITL p50 during prefill (median of streams) | {p50} ms | 71.8 ms |")
print(f"| ITL p95 during prefill | {p95} ms | 80.2 ms |")
print(f"| ITL p99 during prefill | {p99} ms | 91.5 ms |")
print(f"| decode aggregate in window | {tps} tok/s | 19.59 tok/s |")
print(f"| big-prompt TTFT | {ttft} s | 33.39 s |")
print()
print("Context (their published numbers, single box): 1,057/1,111/1,400 ms p50/p95/p99, ~5.8 tok/s.")
print("Decode-only parity anchor: structured 65-67 (x1) / 306-334 (x8) tok/s.")
if err:
    print()
    print(f"_errors_: {', '.join(err)}")
print()
print(f"_run `bash 1p1d/report.sh` on your pair to produce a block like this._")
print("```")
print("=" * 78)
print(f"Post at: {issues} (title: '1p1d results — <your rig notes>')")
print("=" * 78)
PY
