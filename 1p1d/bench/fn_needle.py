#!/usr/bin/env python3
"""GATTLING FN 1P1D needle test (qwen-flash-next-two-sparks-plan.md §8, E2).

Gate: 3/3 needles at 32K and 128K, on BOTH paths:
  bypass   plain request straight to D — D-only correctness incl. PLE mmap
           under decode. Arms A/B (single box / dual kit) run bypass only,
           pointed at the box under test via --durl.
  handoff  P(max_tokens=1, do_remote_decode) -> handles -> D pulls KV+conv
           state over the CX-7 seam and decodes.

Protocol: bench_poc.py's exact needle protocol (same needle sentences, same
25/50/75% injection, same question, PASS = answer code present). Adaptation:
enable_thinking=false so the 96-token budget lands in .content (the 27B rig
solved this with a no-think template; the recipe note prefers the kwarg).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from bench_poc import NEEDLE_ANSWERS, build_prompt  # noqa: E402

def _pair_pins():
    """Endpoints from 1p1d/env.pair (single source of truth); env FN_* wins."""
    import os, re
    pins = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "env.pair")
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r'([A-Z_0-9]+)="(.*)"', line.strip())
            if m:
                pins[m.group(1)] = m.group(2)
    return pins

_pins = _pair_pins()
P_URL = os.environ.get("FN_P_URL", _pins.get("P_HEALTH", "http://127.0.0.1:8000"))
D_URL = os.environ.get("FN_D_URL", _pins.get("D_HEALTH", "http://127.0.0.1:8100"))
CHAT = "/v1/chat/completions"
MODEL = os.environ.get("FN_MODEL", _pins.get("SERVED_MODEL_NAME", "qwen3.8-flash-next"))
REMOTE_HOST = os.environ.get("FN_P_HOST", _pins.get("P_SIDECHANNEL_IP", "127.0.0.1"))
BASE = {"model": MODEL, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
QUESTION = ("What is the magic number, the fallback code, and the release tag "
            "hidden in the text? Answer in one short line.")


def post(url, body, timeout=900):
    req = urllib.request.Request(
        url, json.dumps(dict(BASE, **body)).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_collect(url, body, timeout=900):
    body = dict(BASE, **body, stream=True)
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text = None, ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            piece = (choices[0].get("delta") or {}).get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                text += piece
    return ttft, text


def run_path(path, ctx, seed):
    """One needle cell. Returns dict(pass, needles, ttft, text)."""
    prompt = build_prompt(ctx, needle=True, seed=seed)
    msgs = [{"role": "user", "content": prompt + "\n\n" + QUESTION}]
    if path == "bypass":
        ttft, text = stream_collect(D_URL + CHAT,
                                    {"messages": msgs, "max_tokens": 96})
    else:  # handoff
        t0 = time.time()
        pf = post(P_URL + CHAT, {"messages": msgs, "max_tokens": 1,
                                 "kv_transfer_params": {"do_remote_decode": True}})
        t_pf = round(time.time() - t0, 1)
        h = pf.get("kv_transfer_params")
        if not h:
            return {"pass": False, "error": "no kv_transfer_params",
                    "p_head": str(pf)[:200]}
        h = dict(h)
        h["remote_host"] = REMOTE_HOST
        ttft, text = stream_collect(
            D_URL + CHAT, {"messages": msgs, "max_tokens": 96,
                           "kv_transfer_params": h})
        print(f"    [handoff] P prefill {t_pf}s, remote_num_tokens="
              f"{h.get('remote_num_tokens')}, D-side ttft {round(ttft, 2)}s")
    found = {n: (n in text) for n in NEEDLE_ANSWERS}
    return {"pass": all(found.values()), "needles": found, "ttft": ttft,
            "text": text}


def main():
    global P_URL, D_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32768,131072")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--durl", default=D_URL,
                    help="decode/bypass endpoint; arms A/B point this at the box under test")
    ap.add_argument("--purl", default=P_URL, help="prefill endpoint for the handoff path")
    ap.add_argument("--paths", default="bypass,handoff",
                    help="comma list; arm A/B single-box runs use bypass only")
    args = ap.parse_args()
    P_URL, D_URL = args.purl, args.durl
    ctxs = [int(x) for x in args.ctx.split(",")]
    paths = args.paths.split(",")
    ok = True
    for i, ctx in enumerate(ctxs):
        print(f"\n== ctx={ctx:,} ==")
        outs = {}
        for path in paths:
            res = run_path(path, ctx, args.seed + i * 7 + (0 if path == "bypass" else 3))
            outs[path] = res
            status = "PASS" if res.get("pass") else "FAIL"
            print(f"  {path:<8} {status}  needles={res.get('needles', {})}")
            if not res.get("pass"):
                ok = False
                print(f"    output: {res.get('text', res)!r}")
        if outs.get("bypass", {}).get("text") and outs.get("handoff", {}).get("text"):
            same = outs["bypass"]["text"] == outs["handoff"]["text"]
            print(f"  cross-path greedy match: {'IDENTICAL' if same else 'DIFFER'}")
            if not same:
                print(f"    bypass:  {outs['bypass']['text'][:100]!r}")
                print(f"    handoff: {outs['handoff']['text'][:100]!r}")
    print("\nE2 NEEDLE GATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
