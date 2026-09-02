"""M1 — the headline number, measured three ways under one methodology.

WHY THIS SCRIPT EXISTS
    The README's first table is the claim the whole repo rests on: streaming
    plus a clause-split first chunk takes the paragraph from several seconds to
    under one. A speedup table is only a speedup table if its rows were
    measured the same way -- a baseline gathered under a different thread
    config, or against a different N, is credited with wins the other rows
    already have, and the ratio stops meaning anything.

    So the three rows are not collected separately and assembled later. They
    are measured here, in one run, on one machine state, under one methodology,
    and the table in the README is this script's output.

METHODOLOGY, stated so it can be argued with
    Median of N, not best-of-N. Best-of measures the machine on its luckiest
    day, which is not the day the listener gets. The spread is printed next to
    every median, because a median without a spread hides exactly the tail
    that made lead_words=5 win over 3.

THE THREE ROWS
    baseline    one create() on the whole paragraph, nothing emitted until it
                finishes. Run in-process with the server's exact session
                options, so the only difference from the rows below is the
                chunking policy -- not the thread count.
    sentence    over the WebSocket, cut on sentence boundaries only.
    clause      over the WebSocket, first segment additionally cut at a clause
                boundary (the shipped default).

    The baseline excludes connect + transport. Measured at ~30 ms against a
    ~7000 ms number, that is 0.4% -- stated rather than silently ignored.

Run against a live server:  .venv/bin/python headline.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

import numpy as np
import onnxruntime as rt
import websockets
from kokoro_onnx import Kokoro

import server as srv

URL = "ws://127.0.0.1:8000/tts"
PARAGRAPH = (
    "Headstart streams the first audio chunk before the rest of the clip is generated. "
    "It is a way to reduce latency and make the model feel more responsive. "
    "The first chunk is generated in parallel with the rest of the clip, so it can be "
    "played back immediately while the rest of the audio is still being generated."
)
REPEATS = 3


def baseline_runs() -> tuple[list[float], float]:
    """One create() on the whole paragraph. TTFB == total: nothing ships early."""
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = srv.INTRA_OP_THREADS
    options.inter_op_num_threads = srv.INTER_OP_THREADS
    session = rt.InferenceSession(srv.MODEL, options, providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, srv.VOICES)

    kokoro.create("Warm.", voice="af_sarah", speed=1.0, lang="en-us")  # see server lifespan

    times, audio_s = [], 0.0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        samples, _ = kokoro.create(PARAGRAPH, voice="af_sarah", speed=1.0, lang="en-us")
        times.append((time.perf_counter() - t0) * 1000)
        audio_s = len(samples) / srv.SAMPLE_RATE
    return times, audio_s


async def ws_once(lead: int | None) -> tuple[float, float, float]:
    """Returns (ttfb_ms, total_ms, audio_s) for one streamed request."""
    t0 = time.perf_counter()
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"text": PARAGRAPH, "lead_words": lead}))
        json.loads(await ws.recv())  # start header

        ttfb, meta = None, None
        while True:
            frame = await ws.recv()
            if isinstance(frame, str):
                meta = json.loads(frame)
                if meta["type"] == "end":
                    break
                continue
            if ttfb is None:
                ttfb = (time.perf_counter() - t0) * 1000
    return ttfb, (time.perf_counter() - t0) * 1000, meta["audio_s"]


async def ws_runs(lead: int | None) -> tuple[list[float], list[float], float]:
    ttfbs, totals, audio_s = [], [], 0.0
    for _ in range(REPEATS):
        ttfb, total, audio_s = await ws_once(lead)
        ttfbs.append(ttfb)
        totals.append(total)
    return ttfbs, totals, audio_s


def row(label: str, ttfbs: list[float], totals: list[float],
        audio_s: float, note: str) -> dict:
    return {
        "label": label,
        "ttfb": statistics.median(ttfbs),
        "lo": min(ttfbs), "hi": max(ttfbs),
        "total": statistics.median(totals),
        "audio_s": audio_s,
        "note": note,
    }


async def main() -> None:
    print(f"\n  Paragraph, {REPEATS} runs each, median reported with full spread.")
    print("  Baseline runs in-process; the two streamed rows go over the WebSocket.\n")

    bt, b_audio = baseline_runs()
    rows = [row("baseline — no streaming", bt, bt, b_audio,
                "one create(), nothing emitted until done")]

    for lead, label, note in (
        (0, "sentence chunking", "cut on sentence boundaries only"),
        (None, "+ clause-split first chunk", f"lead_words={srv.DEFAULT_LEAD_WORDS} (shipped default)"),
    ):
        ttfbs, totals, audio_s = await ws_runs(lead)
        rows.append(row(label, ttfbs, totals, audio_s, note))

    print(f"  {'':<28} {'TTFB med':>10} {'spread':>15} {'total':>10} {'audio':>8}")
    print(f"  {'-' * 76}")
    for r in rows:
        print(f"  {r['label']:<28} {r['ttfb']:>7.0f} ms {r['lo']:>6.0f}-{r['hi']:<7.0f} "
              f"{r['total']:>7.0f} ms {r['audio_s']:>7.2f}s")

    base, best = rows[0]["ttfb"], rows[-1]["ttfb"]
    print(f"\n  Speedup, baseline -> shipped default: {base / best:.1f}x "
          f"({base:.0f} ms -> {best:.0f} ms)")
    print(f"  Cost paid for it: total time {rows[-1]['total']:.0f} ms vs "
          f"{rows[0]['total']:.0f} ms unchunked "
          f"({rows[-1]['total'] / rows[0]['total']:.2f}x)")

    # The rows must describe the same audio, or the speedup is measuring a
    # shorter clip rather than a faster one. Kokoro's output length varies a
    # little with where the cuts fall, so this is a tolerance, not equality.
    spread = max(r["audio_s"] for r in rows) - min(r["audio_s"] for r in rows)
    verdict = "same clip" if spread < 0.5 else "MISMATCH — rows are not comparable"
    print(f"  Audio length across rows: {spread:.2f}s spread — {verdict}")


asyncio.run(main())
