"""M1, block 3 — how small should the first chunk be?

--lead-words 6 took the paragraph from ~1600 ms to ~1000 ms. 6 was a guess. The
floor is ~300 ms, so 1000 leaves room, and the obvious move is to cut smaller.

But smaller is not free, and it fails in two different directions:

  TOTAL TIME RISES.  Every chunk pays the ~300 ms fixed cost (block 2). Cutting
      the first sentence into more pieces pays it more times. TTFB improves
      while the clip takes longer overall.

  THE BUFFER CAN RUN DRY.  This is the one that actually breaks the product.
      `lead` is seconds of audio handed over minus seconds already played. A
      tiny first chunk buys ~0.5 s of audio for ~500 ms of work — playback
      starts almost immediately and then has to be fed faster than the model
      can generate. If lead ever goes negative the listener hears a gap, and a
      gap is worse than having waited longer in the first place.

So the right first chunk is the smallest one whose minimum lead stays safely
positive. That is a measurement, not a preference. This sweeps it.

Run against a live server:  .venv/bin/python leadsweep.py
"""

from __future__ import annotations

import asyncio
import json
import time

import websockets

URL = "ws://127.0.0.1:8000/tts"
PARAGRAPH = (
    "Headstart streams the first audio chunk before the rest of the clip is generated. "
    "It is a way to reduce latency and make the model feel more responsive. "
    "The first chunk is generated in parallel with the rest of the clip, so it can be "
    "played back immediately while the rest of the audio is still being generated."
)
LEADS = [None, 12, 10, 8, 6, 5, 4, 3]
REPEATS = 3


async def once(lead: int | None) -> dict:
    t0 = time.perf_counter()
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"text": PARAGRAPH, "lead_words": lead}))
        json.loads(await ws.recv())  # start header

        ttfb = None
        first_audio = 0.0
        min_lead = float("inf")
        chunks = 0
        meta = None
        while True:
            frame = await ws.recv()
            if isinstance(frame, str):
                meta = json.loads(frame)
                if meta["type"] == "end":
                    break
                continue
            if ttfb is None:
                ttfb = (time.perf_counter() - t0) * 1000
                first_audio = meta["audio_s"]
            min_lead = min(min_lead, meta["lead_s"])
            chunks += 1
    return {
        "lead": lead, "ttfb": ttfb, "first_audio": first_audio,
        "chunks": chunks, "total": meta["total_ms"], "min_lead": min_lead,
    }


async def main() -> None:
    # REPEATS because the first pass of this sweep produced two rows 2-3x off
    # the trend, with total time blowing up alongside TTFB -- the whole run was
    # slow, not those configs. That is background load on a laptop, and a
    # default picked from n=1 would have been picked from noise. Median, not
    # best-of: best-of measures the machine on its luckiest day, which is not
    # the day the listener gets.
    print(f"\n  {'lead_words':>10} {'chunks':>7} {'TTFB med':>10} {'spread':>14} "
          f"{'1st audio':>10} {'total':>9} {'min lead':>9}   safe?")
    print(f"  {'-' * 84}")

    base = None
    rows = []
    for lead in LEADS:
        runs = [await once(lead) for _ in range(REPEATS)]
        ttfbs = sorted(r["ttfb"] for r in runs)
        totals = sorted(r["total"] for r in runs)
        r = {
            "lead": lead,
            "ttfb": ttfbs[len(ttfbs) // 2],
            "lo": ttfbs[0], "hi": ttfbs[-1],
            "total": totals[len(totals) // 2],
            "first_audio": runs[0]["first_audio"],
            "chunks": runs[0]["chunks"],
            "min_lead": min(x["min_lead"] for x in runs),
        }
        rows.append(r)
        if base is None:
            base = r["ttfb"]
        # A chunk of audio_s seconds bought with ttfb ms of work: the listener
        # starts playing immediately, so the margin is what protects the gap.
        safe = "ok" if r["min_lead"] > 0.5 else ("TIGHT" if r["min_lead"] > 0 else "GAP")
        label = str(lead) if lead else "off"
        print(f"  {label:>10} {r['chunks']:>7} {r['ttfb']:>7.0f} ms "
              f"{r['lo']:>6.0f}-{r['hi']:<6.0f} "
              f"{r['first_audio']:>9.2f}s {r['total']:>6.0f} ms "
              f"{r['min_lead']:>+8.2f}s   {safe}")

    best = min((r for r in rows if r["min_lead"] > 0.5), key=lambda r: r["ttfb"])
    worst_total = max(rows, key=lambda r: r["total"])
    print(f"\n  Best safe TTFB: lead_words={best['lead']} at {best['ttfb']:.0f} ms "
          f"({base / best['ttfb']:.2f}x vs unsplit), min lead {best['min_lead']:+.2f}s")
    print(f"  Cost of splitting: total time worst case {worst_total['total']:.0f} ms "
          f"vs {rows[0]['total']:.0f} ms unsplit "
          f"({worst_total['total'] / rows[0]['total']:.2f}x)")
    print("\n  TTFB is what the listener feels; total time is what the server pays.")
    print("  The default should be the smallest chunk that keeps min lead safe.")


asyncio.run(main())
