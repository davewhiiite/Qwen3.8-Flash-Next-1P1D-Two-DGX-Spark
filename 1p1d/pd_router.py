#!/usr/bin/env python3
# GATTLING PD router — conditional decode-direct routing (P2).
#   client -> :8200
#     short/cached prompts  -> decode pool :8100 DIRECT (no seam, no wire hop;
#                              D-side prefix cache makes repeat turns ~free)
#     long/uncached prompts -> prefill pool :8000 (Spark pair, KV handles)
#                              -> decode pool :8100 (NIXL/UCX pull, decodes)
#     failures              -> prefill-pool passthrough (never worse than P
#                              alone; handoff first on D failure since P-side
#                              handoff may still work)
#
# Routing rule (Ch5 §5.5.1 conditional disaggregation): tokenized prompt length
# <= GATTLING_D_TOK_MAX -> decode-direct. After ANY handoff turn the decode pool
# owns the full transferred KV as its own blocks (prefix caching is ON on both
# pools), so sessions converge to decode-side cache hits without routing state.
import asyncio
import hashlib
import json
import logging
import os
import time
from collections import deque

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# FN 1P1D pool (P8, 2026-09-26): env-overridable so pool swaps (27B rollback,
# future D-pool moves) never need an edit. Defaults = the live FN pair.
PREFILL_URL = os.environ.get("GATTLING_P_URL", "http://127.0.0.1:8000")  # prefill pool (kv_producer)
DECODE_URL = os.environ.get("GATTLING_D_URL", "http://127.0.0.1:8100")   # decode pool (kv_consumer)
PREFILL_HOST = os.environ.get("GATTLING_P_HOST", "127.0.0.1")           # P's NIXL side-channel host (LAN)
MODEL = os.environ.get("GATTLING_MODEL", "qwen3.8-flash-next")
ROUTER_MODE = os.environ.get("GATTLING_ROUTER_MODE", "auto")  # auto|always_handoff
D_TOK_MAX = int(os.environ.get("GATTLING_D_TOK_MAX", "8192"))
P_INFLIGHT = int(os.environ.get("GATTLING_P_INFLIGHT", "2"))
TOKENIZE_TIMEOUT = float(os.environ.get("GATTLING_TOKENIZE_TIMEOUT", "5"))
PREFILL_SEM = asyncio.Semaphore(P_INFLIGHT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pd-router")
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="gattling PD router")
client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0))

STATS = {
    "bypassed": 0, "handoff": 0, "passthrough": 0,
    "cache_hit_tokens": 0,
    "tokens_by_route": {"bypassed": 0, "handoff": 0},
}
RECENT = deque(maxlen=200)
# session key -> (total_tokens, monotonic ts) of the last successful route.
# A recent route means D provably owns the full prefix (handoff transfers give
# D the KV; bypasses served it), so a small delta can safely bypass too.
SESSION_MEM = {}
SESSION_TTL = 600.0
DELTA_TOK_MAX = int(os.environ.get("GATTLING_D_DELTA_TOK", "2048"))
BYPASS_TOTAL_MAX = int(os.environ.get("GATTLING_D_BYPASS_TOTAL_MAX", "32768"))
_METRICS_CACHE = {"t": 0.0, "hit_rate": None}


def _session_key(body: dict) -> str:
    """Conversation identity under full-history resend. messages[:-1] hashes
    NEVER match across turns (each turn appends assistant+user), so key on the
    first user message: stable turn-to-turn, distinct per task."""
    msgs = body.get("messages") or []
    first_user = next((m for m in msgs if m.get("role") == "user"), None)
    h = hashlib.sha256()
    h.update((body.get("model") or MODEL).encode())
    h.update(json.dumps(first_user.get("content") if first_user else msgs,
                        sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _tool_token_estimate(body: dict) -> int:
    """Rough chars/3 overestimate for serialized tool schemas (ASCII JSON)."""
    tools = body.get("tools")
    if not tools:
        return 0
    return len(json.dumps(tools, default=str)) // 3


async def _count_tokens(body: dict) -> int | None:
    """Token count from the decode pool's own tokenizer + chat template."""
    try:
        r = await client.post(
            DECODE_URL + "/tokenize",
            json={"model": MODEL, "messages": body.get("messages") or [],
                  "chat_template_kwargs": body.get("chat_template_kwargs") or {}},
            timeout=TOKENIZE_TIMEOUT)
        r.raise_for_status()
        n = r.json().get("count")
        return int(n) if n is not None else None
    except Exception as e:
        log.warning("tokenize failed (%s); defaulting to handoff", e)
        return None


def _record(route: str, tokens: int, cached: int, ttft: float | None, key: str):
    STATS[route] = STATS.get(route, 0) + 1
    STATS["cache_hit_tokens"] += cached
    if route in STATS["tokens_by_route"]:
        STATS["tokens_by_route"][route] += tokens
    if tokens > 0:
        SESSION_MEM[key] = (tokens, time.monotonic(), route)
    RECENT.append({"t": round(time.time(), 1), "route": route, "tokens": tokens,
                   "cached_tokens": cached, "ttft_s": round(ttft, 2) if ttft else None,
                   "session": key})
    log.info("route=%s tokens=%d cached=%d ttft=%s session=%s",
             route, tokens, cached, f"{ttft:.2f}s" if ttft else "-", key)


def _cached_from_usage(usage) -> int:
    try:
        return int(usage["prompt_tokens_details"]["cached_tokens"])
    except (KeyError, TypeError, ValueError):
        return 0


def _sum_remote_tokens(handles: dict) -> int:
    v = handles.get("remote_num_tokens")
    if isinstance(v, list):
        try:
            return sum(int(x) for x in v)
        except (TypeError, ValueError):
            return 0
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


async def _passthrough(body: dict, stream: bool):
    """Fallback: serve the request from the prefill pool directly."""
    log.warning("falling back to prefill-pool local serving")
    if stream:
        return StreamingResponse(
            _relay_stream(PREFILL_URL + "/v1/chat/completions", body,
                          tag="passthrough"),
            media_type="text/event-stream")
    r = await client.post(PREFILL_URL + "/v1/chat/completions", json=body)
    return JSONResponse(r.json())


async def _serve_decode_direct(body: dict):
    """Serve from the decode pool without touching the seam. Response or None."""
    try:
        return await client.post(DECODE_URL + "/v1/chat/completions", json=body)
    except Exception as e:
        log.warning("decode-direct failed (%s); falling through to handoff", e)
        return None


async def _decode_healthy() -> bool:
    try:
        r = await client.get(DECODE_URL + "/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


async def _relay_stream(url: str, body: dict, tag: str = "", route: str | None = None,
                        tokens: int = 0, session: str = "-"):
    n_chunks = 0
    first_chunks = []
    t0 = time.time()
    ttft = None
    usage = None
    async with client.stream("POST", url, json=body) as r:
        if r.status_code != 200:
            text = (await r.aread()).decode(errors="replace")[:500]
            log.error("[%s] HTTP %d: %s", tag, r.status_code, text)
            raise RuntimeError(f"HTTP {r.status_code}")
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                n_chunks += 1
                if ttft is None:
                    ttft = time.time() - t0
                if len(first_chunks) < 3:
                    first_chunks.append(line[:220])
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                except (ValueError, IndexError):
                    pass
            yield f"{line}\n\n"
    cached = _cached_from_usage(usage) if usage else 0
    if route:
        _record(route, tokens, cached, ttft, session)
    log.info("[%s] relay done: chunks=%d ttft=%.2fs cached=%d first=%s",
             tag, n_chunks, ttft or 0, cached, " | ".join(first_chunks))


@app.get("/v1/models")
async def models():
    r = await client.get(DECODE_URL + "/v1/models")
    return JSONResponse(r.json())


@app.get("/v1/router/stats")
async def stats():
    return JSONResponse({"mode": ROUTER_MODE, "d_tok_max": D_TOK_MAX,
                         "p_inflight": P_INFLIGHT,
                         "sessions_tracked": len(SESSION_MEM),
                         "d_prefix_cache_hit_rate": await _d_prefix_cache_hit_rate(),
                         **STATS, "recent": list(RECENT)[-50:]})


async def _d_prefix_cache_hit_rate() -> float | None:
    """Scrape the decode pool's prometheus prefix-cache counters. This vLLM
    build emits no prompt_tokens_details in usage, so hit-rate lives here."""
    if time.monotonic() - _METRICS_CACHE["t"] < 10:
        return _METRICS_CACHE["hit_rate"]
    try:
        r = await client.get(DECODE_URL + "/metrics", timeout=3)
        queries = hits = None
        for line in r.text.splitlines():
            if line.startswith("vllm:prefix_cache_queries_total"):
                queries = float(line.split()[-1])
            elif line.startswith("vllm:prefix_cache_hits_total"):
                hits = float(line.split()[-1])
        if queries:
            _METRICS_CACHE.update(t=time.monotonic(), hit_rate=hits / queries)
    except Exception:
        pass
    return _METRICS_CACHE["hit_rate"]


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    stream = body.get("stream", False)
    key = _session_key(body)
    log.info("incoming: stream=%s msgs=%d tools=%d approx_chars=%d",
             stream, len(body.get("messages") or []), len(body.get("tools") or []),
             len(json.dumps(body.get("messages") or [])))
    t_start = time.time()

    # --- conditional routing: short prompts go decode-direct, skip the seam ---
    if ROUTER_MODE == "auto" and "kv_transfer_params" not in body:
        n_tok = await _count_tokens(body)
        if n_tok is not None:
            n_tok += _tool_token_estimate(body)
            mem = SESSION_MEM.get(key)
            fresh = mem is not None and (time.monotonic() - mem[1]) < SESSION_TTL
            delta = (n_tok - mem[0]) if (fresh and mem) else None
            # Rule 1: small total -> unconditional bypass (worst case: one D
            # cold prefill <= D_TOK_MAX, bounded). Rule 2: D-owned prefix with
            # small delta -> bypass. Empirical law on this build: each pool
            # caches only what it prefilled itself — handoff-transferred KV is
            # NOT reusable on D (21.6s re-prefill observed), so delta-bypass is
            # only safe after a BYPASS turn. After a handoff turn, keep handing
            # off: P's cache hit makes the repeat cheap (~3s).
            d_owned = fresh and mem and mem[2] == "bypassed"
            if n_tok <= D_TOK_MAX or (d_owned and delta is not None
                                      and delta <= DELTA_TOK_MAX
                                      and n_tok <= BYPASS_TOTAL_MAX):
                log.info("bypass decision: total=%d delta=%s d_owned=%s",
                         n_tok, delta, bool(d_owned))
                if stream:
                    if not await _decode_healthy():
                        log.warning("decode pool unhealthy; routing to handoff")
                    else:
                        so = dict(body.get("stream_options") or {})
                        so["include_usage"] = True
                        sbody = {**body, "stream_options": so}
                        return StreamingResponse(
                            _relay_stream(DECODE_URL + "/v1/chat/completions",
                                          sbody, tag="bypass", route="bypassed",
                                          tokens=n_tok, session=key),
                            media_type="text/event-stream")
                else:
                    r = await _serve_decode_direct(body)
                    if r is not None and r.status_code == 200:
                        data = r.json()
                        _record("bypassed", n_tok,
                                _cached_from_usage(data.get("usage")),
                                None, key)  # non-stream latency incl. generation: not TTFT
                        return JSONResponse(data)
                    if r is not None:
                        log.warning("decode-direct HTTP %d; falling to handoff",
                                    r.status_code)
                # D direct failed -> fall through to handoff (P path may still work)
        # tokenize failed -> today's behavior (handoff)

    # --- handoff: ask the producer to prefill and hold blocks ---
    async with PREFILL_SEM:
        pf = dict(body)
        pf["max_tokens"] = 1
        pf["stream"] = False
        pf.pop("stream_options", None)
        pf["kv_transfer_params"] = {"do_remote_decode": True}
        try:
            r = await client.post(PREFILL_URL + "/v1/chat/completions", json=pf)
            r.raise_for_status()
            handles = r.json().get("kv_transfer_params")
            if not handles or not handles.get("remote_block_ids"):
                STATS["passthrough"] += 1
                return await _passthrough(body, stream)
        except Exception as e:
            log.error("prefill handoff failed: %s", e)
            STATS["passthrough"] += 1
            return await _passthrough(body, stream)

    htok = _sum_remote_tokens(handles)
    handles["remote_host"] = PREFILL_HOST  # D must reach P over the LAN
    log.info("handoff ok: engine=%s blocks=%d tokens=%s port=%s",
             handles.get("remote_engine_id"),
             len(handles.get("remote_block_ids", [])),
             handles.get("remote_num_tokens"), handles.get("remote_port"))

    # --- decode on the consumer, relay to client ---
    dec = dict(body)
    dec["kv_transfer_params"] = handles
    try:
        if stream:
            return StreamingResponse(
                _relay_stream(DECODE_URL + "/v1/chat/completions", dec,
                              tag="handoff", route="handoff",
                              tokens=htok, session=key),
                media_type="text/event-stream")
        r = await client.post(DECODE_URL + "/v1/chat/completions", json=dec)
        resp = r.json()
        _record("handoff", htok, _cached_from_usage(resp.get("usage")),
                None, key)
        return JSONResponse(resp)
    except Exception as e:
        log.error("decode failed after prefill (P may hold blocks): %s", e)
        # Re-serve locally so the user gets an answer. Note: the prefilled
        # blocks on P stay held (remote_blocks_expiry_time null) until P is
        # restarted or a matching pull completes — watch KV usage.
        STATS["passthrough"] += 1
        return await _passthrough(body, stream)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="warning")
