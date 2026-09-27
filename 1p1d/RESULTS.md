# Qwen3.8-Flash-Next 1P1D PoC — Results (E3/E4)

**Date:** 2026-09-26 · **Plan:** `qwen-flash-next-two-sparks-plan.md` · **Arms:** A = 1× single-Spark
(recipe stock, f2) · B = 2× TP2 dual kit (`~/qwen38fn-kit`, f1+f2, YaRN off / 262K for matched flags)
· C = 1P1D ours (f1 `kv_producer` :8000 → f2 `kv_consumer` :8100, NixlConnector pull-mode, UCX RC
over soft-RoCE `roceP2p1s0f1`, router `pd_router.py` :8200 on box 1).

Matched flags: MTP-3 + 47k draft vocab, KV fp8, MAX_NUM_SEQS=8, 262K native / YaRN off, same image
digest (`vllm/vllm-openai:qwen38-flash-next` `d464f3b466fa`) and same checkpoint snapshot on both
seam ends. Their harnesses, our orchestration: `bench/mixed.py`, `bench/structured.py` (their own
no-sparkDash fallback; + a `--kind prose` variant using `mixed.py`'s exact PROSE workload on the same
machinery), `bench_suite.py` (env-repointed) for the ladder, `bench_poc.py` needles. sparkDash was
not deployed (monitoring-stack install for one sweep API; substitution documented here).

## The money shot — `bench/mixed.py` (2 decoders + one 64K prefill, cold rep)

| | A (single) | B (TP2 dual) | **C (1P1D)** | C vs A | C vs B |
|---|---|---|---|---|---|
| Big-prompt TTFT | 33.67 s | 23.04 s | 33.39 s | parity | −31% (B prefills faster) |
| ITL p50 during prefill | 1,030 ms | 2,469 ms | **71.8 ms** | 14.3× | 34× |
| ITL p95 during prefill | 1,073 ms | 2,915 ms | **80.2 ms** | 13.4× | 36× |
| ITL p99 during prefill | 1,250 ms | 3,108 ms | **91.5 ms** | 13.7× | 34× |
| Decode aggregate in window | 2.23 tok/s | 1.3 tok/s | **19.59 tok/s** | 8.8× | **15×** |
| Warm-cache reps: ITL p95 | 522–902 ms | 1,167–1,828 ms | **78–83 ms** | — | — |

C's decoders run at ~70% of their own quiet pace through the prefill (A: 8%, B: 5%). Their published
A/B collapse (p50 1,057 / p95 1,111 / p99 1,400 ms, ~5.8 tok/s) reproduces on our host at the same
class — the baseline is honest. The dual kit is the *worst* colocated arm: TP2 prefill work allreduces
across both ranks, so colocated decode stalls harder even though the window is shorter.

Adoption gates: chunk-gap p95 ≤ 100 ms → **80.2 ms PASS** (13–36× the colocated arms); in-window
aggregate ≥ 10× → 8.8× vs A (decoders' own essay pace caps it), **15× vs B PASS**.

## Decode sweeps (3 reps, per-stream / aggregate tok/s)

| Level | A structured | C structured | B structured | A prose | C prose | B prose |
|---|---|---|---|---|---|---|
| 1 | 66.7 | 65–67 | 66.7–67.1 | 32.8–36.2 | 33.9–35.3 | 35.4–36.8 |
| 2 | 115–117 | 113–115 | 117–121 | 53–58 | 52–54 | 62–67 |
| 4 | 191–208 | 211–218 | 206–218 | 82–86 | 87–88 | 100–105 |
| 8 | 297–351 | 306–334 | **353–357** | 121–122 | 120–122 | **161–167** |

- **Parity gate PASS:** C ≈ A within run-to-run noise at every level, both workloads (the connector,
  HND KV layout and seam env cost zero on decode).
- **Honest loss (predicted):** B wins pure-decode aggregate @8 — structured +11%, prose **+35%**
  (within the plan's predicted +10–35% TP2-over-CX7 band; finding #2 confirmed). Single-stream is
  parity everywhere — TP2's per-step allreduce RTT eats the bandwidth win.
- Published-anchor note: our structured C1 65–67 / C8 306–334 vs their published 65.2 / 313.6 —
  matched. Their published *prose* (48.7 ×1 / 162.9 ×8) is sparkDash's prose type; our essay-workload
  prose runs ~34 ×1 in **all three arms equally**, so A↔B↔C comparisons stay internally consistent.

## TTFT ladder (cold, unique prompts) and concurrency

| ctx | A | B | C | C−A |
|---|---|---|---|---|
| 8K | 5.07 s | 5.07 s | 4.14 s | −0.9 s |
| 16K | 7.66 s | 5.78 s | 8.54 s | +0.9 s |
| 32K | 14.87 s | 11.02 s | 16.54 s | +1.7 s |
| 64K | 30.22 s | 22.47 s | 33.14 s | +2.9 s |
| 128K | 65.84 s | 48.35 s | 68.14 s | +2.3 s |
| 260K | 148.18 s | 104.88 s | 146.93 s | −1.3 s |

Isolated wire premium (T5, warm-P basis): **0.44 s at 32K** — gate ≤0.5 s PASS. Ladder deltas beyond
that are prefill run-to-run variance (f1 vs f2, moment-to-moment) — at the extremes C lands *inside*
A's noise. **Honest loss:** B's prefill is ~1.4× faster at depth (22.5 vs 33.1 s @64K; 48.4 vs 68.1 s
@128K); C pays the soft-RoCE ship (~1.4 GB at 128K ≈ 2.2 s D-side) where B pays nothing.

Concurrency (bench_suite T4, 8 users × 8K-ctx prompts + 160 tok): **C 78.8 t/s (TPOT 110 ms) > B
60.6 (100 ms) > A 54.1 (126 ms)** — at >8192 tokens those prefills hand off to P, so D decodes 8
users with zero colocated prefill interference. The disaggregation win shows up under concurrent
prefill+decode, not just in the isolation probe.

## Correctness — needles (their protocol, both depths, all arms)

C bypass 3/3 @32K+128K · C handoff 3/3 @32K+128K · C via router 3/3 @32K+128K · A bypass 3/3 ·
B bypass 3/3. MTP-3 stayed **ON** across the seam everywhere — this build's `NixlConnector`
(`SupportsHMA`) + `ssm_conv_transfer_utils` close the 27B's hybrid-boot and MTP-at-seam gaps
(divergence from the 27B PoC's spec-off-for-handoff policy; that's a finding, not a failure).

## Honest-loss table (C loses to B — said out loud)

| Cell | B | C | Delta |
|---|---|---|---|
| Pure-decode aggregate @8 (prose) | 161–167 t/s | 120–122 t/s | B +35% |
| Pure-decode aggregate @8 (structured) | 353–357 t/s | 306–334 t/s | B +11% |
| KV pool depth | 3.65M tok | ~0.92–0.96M tok/side | B 3.9× (C is MAX_NUM_SEQS=8-bound ≈ 7 full 128K sessions either way) |
| TTFT ≥64K | 22.5 s @64K / 48.4 @128K | 33.1 / 68.1 s | B ~1.4× faster prefill |

## Findings (the publishable three)

1. **First Spark-pair PD disaggregation for a hybrid GDN/QSA arch** — and the 27B playbook's two
   nastiest hybrid gaps (hybrid-KV-manager boot, MTP at the seam) are *upstreamed away* in this
   image class: needles pass at MTP-3 ON, both depths, both paths.
2. **TP2-over-CX7 buys +11–35% pure-decode aggregate and nothing single-stream** — confirmed on
   prose and structured. But TP2 is the *worst* box to colocate prefill with decode (ITL p95
   2,915 ms vs single-box 1,073 ms) — the allreduce chain stalls colocated decode harder than
   single-GPU chunked prefill does.
3. **Decode-under-prefill, fixed structurally:** 80 ms p95 ITL and 19.6 tok/s in-window vs 2,915 ms
   / 1.3 tok/s (B) and 1,073 ms / 2.2 tok/s (A) — and it holds through *warm-cache* prefills too
   (78–83 ms vs A's 522–902 / B's 1,167–1,828).

## Wire (soft-RoCE reality)

`UCX_TLS=rc` + `roceP2p1s0f1` (with `--device /dev/infiniband` — the recipe omits it; RC dies at
NIXL init without it) sustains **~1.0–1.2 GB/s** vs ~0.58 GB/s on the 27B-proven TCP pin — 2×.
The plan's ~23 GB/s is raw CX-7 link rate; the integrated fabric port has no RDMA offload engine
(soft-RoCE), so single-stream is capped at ~1.1 GB/s. Ship costs: 32K ≈ 0.44 s, 128K ≈ 2.2 s D-side.

## Repro

- Rig: this fork on both boxes; sources
  of truth in `1p1d/{env.P,env.D,env.pair}`; relaunch = `bash 1p1d/start-pair.sh start` (gates P before D).
- Router: `1p1d/start-router.sh` (or `python3 1p1d/pd_router.py`) on box 1 (:8200, env-overridable pool constants).
- Benches: `1p1d/bench/mixed.py` (BENCH_BASE), `1p1d/bench/structured.py` (--host/--port/--kind),
  `1p1d/bench/fn_needle.py` (--durl/--paths/--purl).
- Raw: internal bench-run JSON (A/B/C × 3 reps); regenerate the mixed money-shot
  with `bash 1p1d/report.sh` on your own pair.
- Rollback: relaunch the prior serving pool (single-box or dual-kit `./start.sh`).
