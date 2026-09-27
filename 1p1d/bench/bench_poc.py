#!/usr/bin/env python3
"""
bench_poc.py — single-file benchmark harness for a disaggregated LLM serving PoC.

Measures:
  - TTFT / prefill throughput
  - decode throughput
  - long-context needle retrieval
  - greedy-output matching between two vLLM OpenAI-compatible endpoints
    (a "prefill pool" and a "decode pool" of a prefill/decode disaggregation setup)

Stdlib only (urllib.request, json, time, argparse, random, statistics).
Deterministic prompt generation (seeded random).

Works against any OpenAI-compatible /v1/chat/completions endpoint with
stream=true (vLLM). SSE parsing: read the response line by line; data lines
start with "data: "; payload "data: [DONE]" ends the stream; each JSON chunk
has choices[0].delta.content (may be absent/empty — skip empties).
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Fixed word list (200+ common English words) for deterministic prompt build.
# ---------------------------------------------------------------------------
WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "she", "or", "an",
    "will", "my", "one", "all", "would", "there", "is", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "into", "your", "take",
    "people", "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after", "use",
    "two", "how", "our", "work", "first", "well", "way", "even", "new", "want",
    "because", "any", "these", "give", "day", "most", "us", "find", "long", "great",
    "world", "still", "large", "very", "small", "old", "right", "much", "big", "high",
    "each", "place", "next", "little", "own", "same", "here", "before", "house", "water",
    "state", "need", "land", "different", "house", "move", "kind", "home", "most", "us",
    "while", "school", "under", "life", "name", "river", "city", "tree", "street", "food",
    "rock", "night", "light", "room", "parent", "child", "side", "world", "head", "story",
    "fish", "car", "bird", "sun", "mountain", "minute", "snow", "month", "weather", "book",
    "eye", "body", "animal", "hour", "paper", "dog", "horse", "boat", "dog", "tree",
    "star", "door", "color", "month", "week", "season", "summer", "winter", "spring", "autumn",
    "morning", "afternoon", "evening", "hour", "minute", "second", "day", "week", "month", "year",
    "table", "chair", "window", "wall", "floor", "roof", "garden", "field", "farm", "forest",
    "lake", "ocean", "sea", "island", "beach", "cliff", "valley", "hill", "road", "bridge",
    "train", "plane", "ship", "truck", "bicycle", "engine", "wheel", "metal", "wood", "stone",
    "glass", "paper", "cloth", "leather", "plastic", "cotton", "wool", "silk", "steel", "iron",
    "gold", "silver", "copper", "bronze", "coal", "oil", "gas", "water", "air", "fire",
    "earth", "sky", "cloud", "rain", "wind", "storm", "thunder", "lightning", "fog", "mist",
    "snow", "ice", "frost", "dew", "heat", "cold", "warm", "cool", "hot", "cold",
    "light", "dark", "bright", "dim", "clear", "cloudy", "sunny", "rainy", "windy", "snowy",
    "fast", "slow", "quick", "rapid", "swift", "steady", "constant", "regular", "frequent", "rare",
    "common", "usual", "normal", "ordinary", "typical", "average", "middle", "center", "edge", "corner",
    "top", "bottom", "front", "back", "left", "right", "above", "below", "inside", "outside",
    "near", "far", "close", "distant", "adjacent", "opposite", "across", "through", "around", "between",
    "among", "within", "without", "during", "before", "after", "since", "until", "while", "when",
    "where", "why", "how", "what", "who", "whom", "whose", "which", "that", "this",
    "those", "these", "such", "same", "other", "another", "either", "neither", "both", "each",
    "every", "all", "some", "any", "no", "none", "few", "many", "much", "little",
    "more", "less", "most", "least", "enough", "plenty", "several", "several", "couple", "pair",
    "group", "set", "lot", "mass", "heap", "pile", "stack", "row", "line", "circle",
    "square", "triangle", "rectangle", "oval", "curve", "angle", "point", "line", "shape", "form",
    "size", "length", "width", "height", "depth", "weight", "volume", "area", "space", "room",
    "room", "hall", "room", "cell", "box", "case", "bag", "pack", "bundle", "bundle",
    "piece", "part", "bit", "chunk", "slice", "drop", "grain", "spark", "flame", "ember",
    "smoke", "steam", "vapor", "mist", "fog", "haze", "dust", "sand", "soil", "mud",
    "clay", "dirt", "earth", "ground", "floor", "base", "bottom", "top", "peak", "summit",
    "ridge", "slope", "bank", "shore", "coast", "beach", "bank", "edge", "border", "boundary",
    "limit", "end", "start", "begin", "finish", "stop", "pause", "rest", "break", "gap",
    "hole", "crack", "cut", "break", "split", "tear", "rip", "break", "damage", "damage",
    "repair", "fix", "mend", "patch", "seal", "close", "open", "shut", "lock", "unlock",
    "bind", "tie", "fasten", "attach", "connect", "join", "link", "couple", "pair", "match",
    "fit", "suit", "agree", "disagree", "differ", "vary", "change", "shift", "move", "turn",
    "shift", "slide", "roll", "spin", "twist", "bend", "fold", "crease", "crumple", "flatten",
    "press", "push", "pull", "drag", "lift", "drop", "throw", "catch", "hold", "keep",
    "save", "store", "hide", "find", "seek", "search", "look", "watch", "see", "view",
    "glance", "stare", "peep", "peek", "spot", "notice", "observe", "notice", "note", "mark",
    "sign", "signal", "symbol", "mark", "label", "tag", "name", "title", "word", "letter",
    "number", "figure", "digit", "sum", "total", "count", "tally", "score", "rate", "ratio",
    "share", "part", "portion", "piece", "slice", "bit", "bit", "bit", "bit", "bit",
]

# Needle sentences (identical every time — required for the needle test).
NEEDLE_1 = "Project records note that the magic number is QX-7F3Q."
NEEDLE_2 = "Project records note that the fallback code is BR-91LK."
NEEDLE_3 = "Project records note that the release tag is v4-ORNG."

NEEDLE_ANSWERS = ["QX-7F3Q", "BR-91LK", "v4-ORNG"]


def build_prompt(n_tokens, needle=True, seed=1234):
    """Synthesize a prompt of approximately n_tokens.

    Approximate tokens as words * 1.35 (no tokenizer API call).
    If needle=True, inject 3 needle sentences at ~25%, ~50%, ~75% of the text.
    Deterministic: seeded random.
    """
    rng = random.Random(seed)
    n_words = max(1, int(round(n_tokens / 1.35)))

    # Build pseudo-sentences from the fixed word list.
    sentences = []
    words_used = 0
    while words_used < n_words:
        # Sentence length 8-16 words.
        sent_len = rng.randint(8, 16)
        sent_words = [rng.choice(WORDS) for _ in range(sent_len)]
        # Capitalize first word, end with period.
        sent_words[0] = sent_words[0].capitalize()
        sentence = " ".join(sent_words) + "."
        sentences.append(sentence)
        words_used += sent_len

    text = " ".join(sentences)

    if needle:
        # Inject needles at ~25%, ~50%, ~75% of the text (by character position).
        # Split text into 4 chunks and insert needles after chunks 1, 2, 3.
        total_len = len(text)
        # Build a list of (position, needle) sorted by position.
        positions = [
            (int(total_len * 0.25), NEEDLE_1),
            (int(total_len * 0.50), NEEDLE_2),
            (int(total_len * 0.75), NEEDLE_3),
        ]
        # Insert in reverse order so earlier positions aren't shifted.
        for pos, needle_text in reversed(positions):
            # Clamp position to valid range.
            pos = max(0, min(pos, len(text)))
            text = text[:pos] + " " + needle_text + " " + text[pos:]

    return text


def chat_stream(base_url, model, user_text, max_tokens, temperature):
    """POST to {base_url}/v1/chat/completions with stream=true.

    Returns (ttft_seconds, full_text, total_gen_seconds, n_delta_tokens).
      - ttft = time until FIRST non-empty delta content arrives
      - total_gen = time until stream ends

    SSE parsing: read line by line; data lines start with "data: ";
    "data: [DONE]" ends the stream; each JSON chunk has
    choices[0].delta.content (may be absent/empty — skip empties).
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Gates measure engine mechanics (prefill/decode rate), not reasoning:
        # with --reasoning-parser qwen3 the model thinks by default and burns
        # the max_tokens budget inside the think block. Disable thinking so
        # the output budget goes to the measured content.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    # Authorization: Bearer dummy only if base_url contains "none".
    if "none" in base_url:
        headers["Authorization"] = "Bearer dummy"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    t_start = time.monotonic()
    ttft = None
    full_parts = []
    n_tok = 0
    total_gen = None

    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.URLError as e:
        print(f"server unreachable at {base_url}: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"server unreachable at {base_url}: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                # Malformed chunk — skip and continue.
                continue
            try:
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta")
                if not delta:
                    continue
                # Count BOTH reasoning and content deltas: with
                # --reasoning-parser the model streams delta.reasoning
                # first (vLLM >=0.29; older builds used
                # reasoning_content), and skipping it would measure
                # 0 tok/s.
                content = (delta.get("content") or "") + (delta.get("reasoning") or "") + (delta.get("reasoning_content") or "")
                if content == "":
                    continue
            except (AttributeError, IndexError, TypeError):
                # Malformed chunk structure — skip and continue.
                continue
            if ttft is None:
                ttft = time.monotonic() - t_start
            full_parts.append(content)
            n_tok += 1
    finally:
        total_gen = time.monotonic() - t_start
        try:
            resp.close()
        except Exception:
            pass

    full_text = "".join(full_parts)
    if ttft is None:
        # No non-empty content arrived; use total_gen as a fallback.
        ttft = total_gen
    return ttft, full_text, total_gen, n_tok


def fmt(x, nd=3):
    """Format a number with thousands separators and fixed decimals."""
    if x is None:
        return "n/a"
    return f"{x:,.{nd}f}"


def run_ttft(decode_url, model, ctxs, runs):
    """Mode ttft: for each ctx, run --runs chat_stream calls with max_tokens=8.

    Report per-ctx: TTFT min/avg/max (s), implied prefill tok/s = n_tokens/TTFT_avg.
    """
    results = {"mode": "ttft", "ctxs": {}}
    print(f"\n=== TTFT / Prefill throughput (decode url: {decode_url}) ===")
    for ctx in ctxs:
        prompt = build_prompt(ctx, needle=False)
        n_tokens = int(round(len(prompt.split()) * 1.35))
        ttfts = []
        for _ in range(runs):
            ttft, _, _, _ = chat_stream(decode_url, model, prompt, max_tokens=8, temperature=0)
            ttfts.append(ttft)
        ttft_min = min(ttfts)
        ttft_avg = statistics.mean(ttfts)
        ttft_max = max(ttfts)
        prefill_tps = n_tokens / ttft_avg if ttft_avg > 0 else 0.0
        results["ctxs"][str(ctx)] = {
            "n_tokens": n_tokens,
            "ttft_min": ttft_min,
            "ttft_avg": ttft_avg,
            "ttft_max": ttft_max,
            "prefill_tok_s": prefill_tps,
        }
        print(
            f"  ctx={ctx:>7,}  n_tokens={n_tokens:>7,}  "
            f"TTFT min={fmt(ttft_min)}s  avg={fmt(ttft_avg)}s  max={fmt(ttft_max)}s  "
            f"prefill={fmt(prefill_tps, 1)} tok/s"
        )
    return results


def run_decode(decode_url, model, runs):
    """Mode decode: prompt ~256 tokens, max_tokens=256, --runs.

    Report decode tok/s = (max_tokens-1)/(total_gen - ttft), min/avg/max.
    """
    results = {"mode": "decode", "runs": []}
    print(f"\n=== Decode throughput (decode url: {decode_url}) ===")
    prompt = build_prompt(256, needle=False)
    n_tokens = int(round(len(prompt.split()) * 1.35))
    decode_tps_list = []
    for i in range(runs):
        ttft, _, total_gen, n_tok = chat_stream(decode_url, model, prompt, max_tokens=256, temperature=0)
        decode_time = total_gen - ttft
        decode_tps = (n_tok - 1) / decode_time if decode_time > 0 and n_tok > 1 else 0.0
        decode_tps_list.append(decode_tps)
        results["runs"].append({
            "run": i,
            "ttft": ttft,
            "total_gen": total_gen,
            "decode_tps": decode_tps,
        })
        print(
            f"  run={i}  ttft={fmt(ttft)}s  total_gen={fmt(total_gen)}s  "
            f"decode={fmt(decode_tps, 1)} tok/s"
        )
    results["decode_tps_min"] = min(decode_tps_list)
    results["decode_tps_avg"] = statistics.mean(decode_tps_list)
    results["decode_tps_max"] = max(decode_tps_list)
    print(
        f"  decode tok/s: min={fmt(results['decode_tps_min'], 1)}  "
        f"avg={fmt(results['decode_tps_avg'], 1)}  max={fmt(results['decode_tps_max'], 1)}"
    )
    return results


def run_needle(decode_url, model, ctx):
    """Mode needle: ctx from --ctxs (default 32768), max_tokens=96, temperature=0.

    Ask for the magic number, fallback code, and release tag.
    Check output contains QX-7F3Q, BR-91LK, v4-ORNG; print PASS/FAIL per needle.
    """
    results = {"mode": "needle", "ctx": ctx, "needles": {}}
    print(f"\n=== Needle retrieval (ctx={ctx:,}, decode url: {decode_url}) ===")
    prompt = build_prompt(ctx, needle=True)
    question = (
        "What is the magic number, the fallback code, and the release tag "
        "hidden in the text? Answer in one short line."
    )
    # Append the question to the prompt.
    full_prompt = prompt + "\n\n" + question
    ttft, output, total_gen, _ = chat_stream(decode_url, model, full_prompt, max_tokens=384, temperature=0)
    print(f"  ttft={fmt(ttft)}s  total_gen={fmt(total_gen)}s")
    print(f"  output: {output!r}")
    all_pass = True
    for needle in NEEDLE_ANSWERS:
        found = needle in output
        results["needles"][needle] = found
        status = "PASS" if found else "FAIL"
        if not found:
            all_pass = False
        print(f"  {needle}: {status}")
    results["all_pass"] = all_pass
    print(f"  overall: {'PASS' if all_pass else 'FAIL'}")
    return results


def run_match(prefill_url, decode_url, model, ctx):
    """Mode match: same build_prompt(ctx) sent to TWO endpoints, temperature=0,
    max_tokens=64; compare the two output strings; report character-level match
    percentage and first divergence index.
    """
    results = {"mode": "match", "ctx": ctx}
    print(f"\n=== Greedy output match (ctx={ctx:,}) ===")
    print(f"  prefill url: {prefill_url}")
    print(f"  decode url:  {decode_url}")
    prompt = build_prompt(ctx, needle=False)
    _, out_prefill, _, _ = chat_stream(prefill_url, model, prompt, max_tokens=64, temperature=0)
    _, out_decode, _, _ = chat_stream(decode_url, model, prompt, max_tokens=64, temperature=0)
    print(f"  prefill output: {out_prefill!r}")
    print(f"  decode  output: {out_decode!r}")

    # Character-level match percentage and first divergence index.
    n = min(len(out_prefill), len(out_decode))
    match_count = 0
    first_div = None
    for i in range(n):
        if out_prefill[i] == out_decode[i]:
            match_count += 1
        else:
            first_div = i
            break
    # If no divergence in the common prefix, check length difference.
    if first_div is None and len(out_prefill) != len(out_decode):
        first_div = n
    total_chars = max(len(out_prefill), len(out_decode), 1)
    match_pct = (match_count / total_chars) * 100.0
    results["prefill_len"] = len(out_prefill)
    results["decode_len"] = len(out_decode)
    results["match_pct"] = match_pct
    results["first_divergence"] = first_div
    results["identical"] = (out_prefill == out_decode)
    print(f"  prefill len={len(out_prefill)}  decode len={len(out_decode)}")
    print(f"  match: {fmt(match_pct, 2)}%  first divergence index: {first_div}")
    print(f"  identical: {out_prefill == out_decode}")
    return results


def print_usage():
    """Print usage + mode descriptions."""
    print(
        """
bench_poc.py — disaggregated LLM serving PoC benchmark harness

Usage:
  python3 bench_poc.py --model <model> --mode <mode> [options]

Modes:
  ttft    Measure TTFT / prefill throughput at each context size.
          For each ctx in --ctxs, run --runs calls with max_tokens=8.
          Reports TTFT min/avg/max and implied prefill tok/s.

  decode  Measure decode throughput. Prompt ~256 tokens, max_tokens=256.
          Reports decode tok/s = (max_tokens-1)/(total_gen - ttft).

  needle  Long-context needle retrieval. Uses ctx from --ctxs (default 32768).
          Injects 3 needles; asks the model to recall them. PASS/FAIL per needle.

  match   Greedy-output matching. Same prompt sent to prefill and decode URLs
          (temperature=0, max_tokens=64). Reports character-level match %
          and first divergence index.

  all     Run ttft, decode, needle, match in sequence.

Options:
  --prefill-url   Prefill pool endpoint (default http://localhost:8100)
  --decode-url    Decode pool endpoint (default http://localhost:8000)
  --model         Model name (required for actual runs)
  --mode          One of: ttft, decode, needle, match, all, help
  --ctxs          Comma-separated context sizes (default "8192,32768,131072")
  --runs          Number of runs per measurement (default 3)

Examples:
  python3 bench_poc.py --model my-model --mode ttft
  python3 bench_poc.py --model my-model --mode needle --ctxs 32768
  python3 bench_poc.py --model my-model --mode all
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Disaggregated LLM serving PoC benchmark harness",
        add_help=False,
    )
    parser.add_argument("--prefill-url", default="http://localhost:8100")
    parser.add_argument("--decode-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--ctxs", default="8192,32768,131072")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--help", action="store_true")

    args = parser.parse_args()

    # Parse ctxs.
    try:
        ctxs = [int(x.strip()) for x in args.ctxs.split(",") if x.strip()]
    except ValueError:
        print(f"invalid --ctxs: {args.ctxs}", file=sys.stderr)
        sys.exit(1)
    if not ctxs:
        print("no context sizes specified in --ctxs", file=sys.stderr)
        sys.exit(1)

    # Help mode: --mode help, no --mode, --help, or no --model.
    if args.help or args.mode is None or args.mode == "help" or args.model is None:
        print_usage()
        sys.exit(0)

    mode = args.mode.lower()
    if mode not in ("ttft", "decode", "needle", "match", "all"):
        print(f"unknown mode: {mode}", file=sys.stderr)
        print_usage()
        sys.exit(1)

    all_results = {}

    if mode == "ttft":
        all_results["ttft"] = run_ttft(args.decode_url, args.model, ctxs, args.runs)
    elif mode == "decode":
        all_results["decode"] = run_decode(args.decode_url, args.model, args.runs)
    elif mode == "needle":
        # ctx from --ctxs (default 32768) — use the first ctx value.
        needle_ctx = ctxs[0] if ctxs else 32768
        all_results["needle"] = run_needle(args.decode_url, args.model, needle_ctx)
    elif mode == "match":
        # Use the first ctx value for match.
        match_ctx = ctxs[0] if ctxs else 8192
        all_results["match"] = run_match(args.prefill_url, args.decode_url, args.model, match_ctx)
    elif mode == "all":
        # Run ttft, decode, needle, match in sequence.
        # ttft and decode against decode URL as primary.
        all_results["ttft"] = run_ttft(args.decode_url, args.model, ctxs, args.runs)
        all_results["decode"] = run_decode(args.decode_url, args.model, args.runs)
        # needle against decode URL.
        needle_ctx = ctxs[0] if ctxs else 32768
        all_results["needle"] = run_needle(args.decode_url, args.model, needle_ctx)
        # match: prefill URL for prefill, decode URL for decode.
        match_ctx = ctxs[0] if ctxs else 8192
        all_results["match"] = run_match(args.prefill_url, args.decode_url, args.model, match_ctx)

    # Dump JSON results object at the end.
    print("\n=== JSON results ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
