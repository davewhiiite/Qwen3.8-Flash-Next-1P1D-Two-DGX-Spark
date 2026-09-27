# Two DGX Sparks, one model, disaggregated

**Qwen3.8-Flash-Next served prefill/decode-split across a pair of NVIDIA DGX Spark (GB10) boxes —
with parity on both prefill and decode, and the colocated-iterative-latency problem solved
structurally: during a 64K prefill, decoders stream at 80 ms ITL instead of freezing for seconds.**

This is a fork of
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
(their single-box and TP2 dual-box recipes, their harnesses, their image and checkpoint
conventions — see [their README](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#readme)
and give them the stars). We added `1p1d/`: one pool per box, KV + GDN conv state shipped
over the CX-7 fabric (NIXL/UCX), and a small conditional router in front. Everything else in
this repo is theirs, unmodified except for four hunks in `start.sh` (the `KV_TRANSFER_ROLE`
knob) and one `--device /dev/infiniband` flag.

Scripts only — a few hundred KB. Weights come from HuggingFace via their sha256-verified
`download.sh`; the image comes from Docker Hub; the ~27 GB PLE table builds locally on first
launch. Clone → download → `1p1d/start-pair.sh start` → serving.

---

## Why: the number that matters

`1p1d/bench/mixed.py` (their harness): two streams decoding an essay; mid-stream, a 64K
prompt lands. All arms: matched flags, same image digest, same checkpoint snapshot, same
harness. **A** = their single-Spark recipe. **B** = their TP2 dual-Spark kit. **C** = this
(1P1D).

| ITL during the 64K prefill | A (single) | B (TP2 dual) | **C (1P1D)** |
|---|---|---|---|
| p50 | 1,030 ms | 2,469 ms | **71.8 ms** |
| p95 | 1,073 ms | 2,915 ms | **80.2 ms** |
| p99 | 1,250 ms | 3,108 ms | **91.5 ms** |
| decode aggregate in window | 2.23 tok/s | 1.3 tok/s | **19.6 tok/s** |
| % of its own quiet pace | 8% | 5% | ~70% |

The colocated collapse reproduces exactly as they published it (p50 1,057 / p95 1,111 /
p99 1,400 ms) — their baseline is honest. The surprise: **TP2 is the *worst* place to
colocate prefill with decode** — the prefill's allreduce chain stalls both ranks, so B's
decoders hiccup 2.7× worse than single-box. And it holds when the cache is warm (reps that
prefill in 1–2 s still trip colocated decode to 500–1,800 ms; C never leaves the 70s).

## Prefill and decode: not nerfed

Cold TTFT ladder (unique prompts, client-side, identical harness for all arms) — C tracks A
within run-to-run noise at every depth:

| ctx | A single | B TP2 | **C 1P1D** |
|---|---|---|---|
| 8K | 5.07 s | 5.07 s | **4.14 s** |
| 32K | 14.87 s | 11.02 s | 16.54 s |
| 64K | 30.22 s | 22.47 s | 33.14 s |
| 128K | 65.84 s | 48.35 s | 68.14 s |
| 260K | 148.18 s | 104.88 s | **146.93 s** |

Decode sweep (their `structured.py`, 3 reps) — C ≡ A at every level, both workloads. The
whole seam costs zero on decode:

| streams (agg tok/s, structured) | A | **C** | B |
|---|---|---|---|
| 1 | 66.7 | **65–67** | 66.7–67.1 |
| 8 | 297–351 | **306–334** | **353–357** |

And the daily-driver shape — 8 concurrent users with fat prompts (8K ctx, >8192 tokens
route over the seam): **C 78.8 t/s (TPOT 110 ms) > B 60.6 > A 54.1**. Every one of those
prefills happened on the other box. That's the point.

## Where C still loses (read this before you build it)

| Cell | B (TP2) | C (1P1D) |
|---|---|---|
| Pure-decode aggregate @8 (prose) | 161–167 t/s | 120–122 t/s — **B +35%** |
| Pure-decode aggregate @8 (structured) | 353–357 t/s | 306–334 t/s — **B +11%** |
| Prefill at depth | 22.5 s @64K / 48.4 s @128K | 33.1 / 68.1 s — **B ~1.4× faster** |
| KV pool depth | 3.65M tok | ~0.94M tok/side (≈7 full 128K sessions — `MAX_NUM_SEQS=8`-bound either way) |

If you only ever run pure decode, TP2 wins. If prompts and streams ever share a box,
1P1D wins. Full tables: [`1p1d/RESULTS.md`](1p1d/RESULTS.md); narrative:
[`1p1d/REPORT.md`](1p1d/REPORT.md).

## Correctness

Needle tests (`1p1d/bench/fn_needle.py`, bench_poc protocol): **3/3 at 32K and 128K on
every path** — bypass, P→D handoff, and through the router — with **MTP-3 speculative
decoding left ON across the seam**. The 27B playbook's two hybrid-arch gaps (connector
blocked hybrid-KV-manager boot; MTP refused the transfer) are upstreamed away in this
image's vLLM: `NixlConnector` subclasses `SupportsHMA` and ships the GDN conv-state format.

## Reproduce from zero

Prereqs: 2× DGX Spark on the same LAN with working ssh **box1 → box2** (BatchMode/key auth),
Docker, ~99 GiB free per box (or HF-cache rsync), the recipe's prerequisites from
[their README](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#readme).

```bash
# 1. clone the fork on BOTH boxes
git clone https://github.com/davewhiiite/Qwen3.8-Flash-Next-1P1D-Two-DGX-Spark.git
cd Qwen3.8-Flash-Next-1P1D-Two-DGX-Spark

# 2. weights (their downloader, sha256-verified, resumable)
./download.sh                       # ~99 GiB, on both boxes

# 3. pin YOUR rig (box 1 and box 2 keep identical copies)
$EDITOR 1p1d/env.pair               # ssh pin, repo paths, LAN/fabric IPs
$EDITOR 1p1d/env.P 1p1d/env.D       # the four *_IP pins per box (see comments)
#    (no .env needed — start-pair.sh synthesizes one per box from env.P/env.D
#     on first start, role pin included, so supervisor relaunches stay correct)

# 4. inspect, then launch (P gates healthy first — D needs P's NIXL side
#    channel at init; cold start ~11 min/box, first launch adds the ~27 GB
#    PLE table build)
bash 1p1d/start-pair.sh dry-run
bash 1p1d/start-pair.sh start       # also: status | stop | restart

# 5. prove it
python3 1p1d/bench/fn_needle.py                     # 3/3 @ 32K,128K, bypass+handoff
python3 1p1d/bench/structured.py --streams 1 2 4 8  # decode parity vs your A rig
python3 1p1d/bench/mixed.py --tag mypair            # the money shot
bash 1p1d/report.sh                                 # one-paste issue block -> post it
```

Point clients at the **router** (box 1, `:8200`): short/cached prompts decode-direct on the
decode pool (prefix-cache hits, no wire hop), long prompts prefill → handoff → decode.
`GET /v1/router/stats` shows the route mix.

Measured stranger-run: _pending — timed on the release candidate before tagging._

## Gotchas (each one cost us a crash or a re-run)

1. **`--device /dev/infiniband` is not optional on the RC transport.** The recipe's docker
   run omits it; UCX can't see the HCA and NIXL init dies (the VRAM-drop you'll chase for
   an hour). Our env.P/env.D add it. Root cause of the "RC crash": RDMA without the device
   node.
2. **RC ≈ 1.1 GB/s vs TCP ≈ 0.58 GB/s.** Worth the flag, but see #3.
3. **The Spark fabric port is soft-RoCE** — no RDMA offload engine, so the plan-scale
   "~23 GB/s line rate" is fiction. Ships cost ~0.44 s @32K, ~2.2 s @128K, hidden inside
   chunked-prefill noise at ≤64K.
4. **Boot order: P before D, always.** The decode pool connects to the prefill pool's NIXL
   side channel at engine init; boot D first and it fails. `start-pair.sh` enforces this.
5. **Matched flags across the seam = the KV contract.** Same image digest, same checkpoint
   snapshot, same chat template on both ends, or you're back in silent-corruption territory
   (the 27B PoC's whole failure class). `env.P`/`env.D` differ only in role/port/IP pins.
6. **YaRN: off/262K for comparison runs, on/524K for serving.** All tables above are
   YaRN-off/262K for matched-flags comparability; serving config turns it on.
7. **`enable_thinking` kwarg.** The template reasons by default; the needle protocol uses
   `chat_template_kwargs: {"enable_thinking": false}` so the token budget lands in
   `.content`. Your agent harness probably wants the kwarg too.
8. **Salt-collision honesty rule.** Any harness that salts prompts per-rep to defeat the
   prefix cache can collide and silently hand you cache hits labeled "cold". Salt uniquely
   per rep; re-verify suspiciously-fast cold cells.

## What's in `1p1d/`

| file | role |
|---|---|
| `start-pair.sh` | the orchestrator: run on box 1, ssh-drives box 2 (head/worker pattern, detached launches survive ssh drops, per-node readiness gates) |
| `env.pair` | your rig pins (hosts, paths, IPs) |
| `env.P` / `env.D` | per-pool engine env — the 4-hunk `KV_TRANSFER_ROLE` knob + seam pins |
| `pd_router.py` | conditional decode-direct router (:8200) — runs in the required image, zero extra deps |
| `start-router.sh` | router container launcher (host net, restart=unless-stopped) |
| `bench/` | `mixed.py`, `structured.py` (router-default ports), `fn_needle.py` + `bench_poc.py` protocol |
| `report.sh` | fingerprints your rig, runs the money-shot once, prints a one-paste GitHub issue block — post it, that's the whole feedback loop |
| `RESULTS.md` / `REPORT.md` | full A/B/C tables + honest losses / the narrative |

## Credits & lineage

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)** — the
  recipes (single-box and TP2 dual-kit), the image, the download/verify machinery, the bench
  harnesses (`mixed.py`, `structured.py`, `decodebench.py` conventions), the needle protocol,
  sparkDash monitoring conventions. This fork exists because their rig is good enough to
  disaggregate.
- **The 27B playbook** — our earlier 1P1D PoC on the hybrid GDN/QSA 27B class found the
  seam bugs (hybrid boot, MTP-at-seam, TCP-vs-RC tradeoffs) that this run confirms are
  fixed upstream in this image class.
- AGPL-3.0-or-later, same as upstream; upstream file headers preserved; our new files are
  ours, credited to the fork.

## Upstream PR status

The `KV_TRANSFER_ROLE` knob (empty = stock behavior) and the `--device /dev/infiniband` fix
will be offered upstream as a small backwards-compatible PR **after** this fork is stable —
not before. Don't hold your breath; do run the report.
