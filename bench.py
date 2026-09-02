"""M2, step 1 — what happens to the number when more than one person asks.

Every latency figure in this repo so far was measured with one client and an
otherwise idle machine. That is the number a demo produces and it is not the
number a service has. This harness produces the second one.

WHY THIS COMES BEFORE THE BATCHER
    M2's deliverable is a batcher. A batcher is only worth building if the
    queue is the problem, and "the queue is the problem" is a claim, not a
    fact, until the queue is measured against the model time it is competing
    with. The server already reports queue_ms separately from gen_ms for
    exactly this reason (server.py design note 2). This reads both and says
    which one owns the tail.

THE PREDICTION, MADE BEFORE THE RUN
    One model call at a time (MODEL_SLOT). Chunks from different clients
    interleave in that slot, so a newly arrived client's first chunk waits
    behind at most one chunk from each of the other N-1 clients:

        TTFB(N) ≈ TTFB(1) + (N-1) × mean_chunk_gen_ms

    The harness prints predicted next to measured. If they agree, the queue is
    plain FIFO waiting and a batcher has a known ceiling to beat. If measured
    runs above prediction, something else is being paid -- thread contention
    inside ORT, or the event loop -- and the batcher would be aimed at the
    wrong thing.

THE CAPACITY NUMBER
    RTF is generation ÷ audio. At RTF r, one core-set can sustain 1/r
    simultaneous realtime streams before demand outruns supply; past that the
    queue grows without bound and no amount of buffering saves it. Measured
    RTF here is ~0.37, so the arithmetic says ~2.7 streams. The sweep is run
    across that line on purpose, because a predicted saturation point that is
    never crossed is not evidence.

WHAT COUNTS AS BROKEN
    Not a big p95. `lead` going negative -- the client has played everything
    it was given and the next chunk has not arrived, so the listener hears a
    hole in the middle of a sentence. That is the only failure a listener can
    detect, so it is reported as its own column rather than averaged into a
    latency percentile.

Run:  .venv/bin/python bench.py            (server must be running)
      .venv/bin/python bench.py --levels 1,2,4,8 --requests 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import websockets

PARAGRAPH = (
    "Headstart streams the first audio chunk before the rest of the clip is generated. "
    "It is a way to reduce latency and make the model feel more responsive. "
    "The first chunk is generated in parallel with the rest of the clip, so it can be "
    "played back immediately while the rest of the audio is still being generated."
)


class Result:
    """One request, from the client's side of the wire."""

    def __init__(self) -> None:
        self.ttfb_ms = 0.0
        self.total_ms = 0.0
        self.audio_s = 0.0
        self.queue_ms = 0.0      # summed over chunks: time waiting for the slot
        self.gen_ms = 0.0        # summed over chunks: time inside the model
        self.min_lead_s = 0.0    # worst buffer margin seen; < 0 means a gap
        self.chunk_gen: list[float] = []


async def one_request(ws, text: str) -> Result:
    r = Result()
    t0 = time.perf_counter()
    await ws.send(json.dumps({"text": text}))

    meta = None
    first = True
    leads = []
    while True:
        frame = await ws.recv()
        if isinstance(frame, str):
            meta = json.loads(frame)
            if meta["type"] == "end":
                break
            if meta["type"] == "chunk":
                r.queue_ms += meta["queue_ms"]
                r.gen_ms += meta["gen_ms"]
                r.chunk_gen.append(meta["gen_ms"])
                leads.append(meta["lead_s"])
            continue
        # binary frame: audio actually in the client's hands
        if first:
            r.ttfb_ms = (time.perf_counter() - t0) * 1000
            first = False

    r.total_ms = (time.perf_counter() - t0) * 1000
    r.audio_s = meta["audio_s"]
    r.min_lead_s = min(leads) if leads else 0.0
    return r


async def loop(ws, text: str, n: int, out: list[Result]) -> None:
    """One connection issuing requests back to back (closed loop).

    Closed loop, not open loop: N here means N conversations in flight, which
    is how a TTS service is actually loaded. An open-loop generator would let
    the backlog grow forever once past saturation and end up measuring the
    backlog rather than the server.
    """
    for _ in range(n):
        out.append(await one_request(ws, text))


async def level(url: str, text: str, n_clients: int, n_requests: int,
                warmup: int) -> tuple[list[Result], float]:
    """Run one concurrency level. Returns (results, measured wall seconds).

    Warmup runs on all connections first and is discarded, so the measured
    window starts with every client already in flight. Without that, the first
    client to connect gets a stretch of an empty queue and its TTFB is really
    a single-client number wearing the level's label.
    """
    results: list[Result] = []
    conns = await asyncio.gather(*[
        websockets.connect(url, max_size=None) for _ in range(n_clients)
    ])
    try:
        if warmup:
            await asyncio.gather(*[loop(ws, text, warmup, []) for ws in conns])
        t0 = time.perf_counter()
        await asyncio.gather(*[loop(ws, text, n_requests, results) for ws in conns])
        elapsed = time.perf_counter() - t0
    finally:
        await asyncio.gather(*[ws.close() for ws in conns])
    return results, elapsed


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def control_check(base: list[Result], control: list[Result]) -> bool:
    """Re-measure the first level last and require the two to agree.

    Levels run in sequence, so the lowest one supplies the baseline that every
    prediction and every ratio is expressed against. If the machine was busier
    during that first minute than during the rest, an inflated baseline makes
    the later levels look better than they are and the whole table is a
    statement about background load. Running the same level again at the end is
    the cheapest way to detect that, and it is checked here rather than eyeballed
    because a benchmark that cannot invalidate itself is decoration.
    """
    b = statistics.median([r.total_ms for r in base])
    c = statistics.median([r.total_ms for r in control])
    drift = max(b, c) / min(b, c)
    ok = drift <= 1.15
    print(f"\n  Control — first level re-run last")
    print(f"    at the start  {b:>7.0f} ms      at the end  {c:>7.0f} ms      "
          f"drift {drift:.2f}x  {'ok' if ok else 'VOID — machine state changed'}")
    if not ok:
        print("    The baseline and the control disagree, so the ratios below are")
        print("    measuring the machine, not the server. Re-run on a quiet box.")
    return ok


def report(levels: dict[int, list[Result]], wall: dict[int, float],
           control: list[Result] | None = None) -> None:
    base = levels[min(levels)]
    if control:
        if control_check(base, control):
            # Both windows are valid samples of the same state, so pool them.
            base = base + control
    ttfb1 = statistics.median([r.ttfb_ms for r in base])
    # Mean, not median. The wait is a *sum* of the other clients' chunks, and
    # the expectation of a sum is the sum of means. Chunk generation is heavily
    # skewed here -- the lead chunk is a fifth of a sentence and the rest are
    # whole ones -- so the median under-counts the wait by about 20%.
    chunk_ms = statistics.mean([g for r in base for g in r.chunk_gen])
    rtf1 = statistics.median([r.total_ms / 1000 / r.audio_s for r in base])

    n_samples = len(levels[min(levels)])
    print(f"\n  Latency under concurrency — {n_samples} requests per client, "
          f"paragraph, lead_words=server default\n")
    print(f"  {'clients':>7} {'reqs':>5} {'TTFB p50':>9} {'p90':>8} {'p95':>8} "
          f"{'max':>8} {'predicted':>10} {'gap':>7}")
    print("  " + "-" * 70)
    for n in sorted(levels):
        t = [r.ttfb_ms for r in levels[n]]
        predicted = ttfb1 + (n - 1) * chunk_ms
        p50 = pct(t, 0.50)
        print(f"  {n:>7} {len(t):>5} {p50:>6.0f} ms {pct(t, 0.90):>5.0f} ms "
              f"{pct(t, 0.95):>5.0f} ms {max(t):>5.0f} ms {predicted:>7.0f} ms "
              f"{p50 / predicted:>6.2f}x")
    print(f"\n  predicted = TTFB(1) {ttfb1:.0f} ms + (N-1) x mean chunk "
          f"{chunk_ms:.0f} ms — one chunk from each client ahead in the slot.")

    print(f"\n  Where the time goes\n")
    print(f"  {'clients':>7} {'queue p50':>10} {'gen p50':>9} {'queue share':>12} "
          f"{'min lead':>9} {'gaps':>7} {'throughput':>11}")
    print("  " + "-" * 72)
    for n in sorted(levels):
        rs = levels[n]
        q = pct([r.queue_ms for r in rs], 0.50)
        g = pct([r.gen_ms for r in rs], 0.50)
        gaps = sum(1 for r in rs if r.min_lead_s < 0)
        audio = sum(r.audio_s for r in rs)
        print(f"  {n:>7} {q:>7.0f} ms {g:>6.0f} ms {q / (q + g) * 100:>11.0f}% "
              f"{min(r.min_lead_s for r in rs):>+7.2f}s {gaps:>3}/{len(rs):<3} "
              f"{audio / wall[n]:>7.2f} a-s/s")
    print("\n  queue share is the batcher's addressable surface: model time is")
    print("  fixed by the graph, waiting is not. gaps counts requests whose")
    print("  buffer ran dry — the only failure the listener can actually hear.")

    cap = 1 / rtf1
    peak = max(sum(r.audio_s for r in levels[n]) / wall[n] for n in levels)
    print(f"\n  Capacity\n")
    print(f"    single-client RTF          {rtf1:.3f}")
    print(f"    predicted realtime streams {cap:.1f}   (1 / RTF)")
    print(f"    throughput ceiling         {peak:.2f} audio-seconds per second")
    # Independent route to the same line: arithmetic says the queue starts
    # growing past 1/RTF streams; listening says it starts at the first level
    # where a buffer runs dry. They should land in the same place.
    clean = [n for n in sorted(levels) if not any(r.min_lead_s < 0 for r in levels[n])]
    broke = [n for n in sorted(levels) if any(r.min_lead_s < 0 for r in levels[n])]
    if clean and broke:
        print(f"    observed                   gap-free to {max(clean)}, "
              f"gaps from {min(broke)}  ->  saturates between them")
    # One model call at a time means the server cannot emit audio faster than
    # one stream generates it. Measured throughput above 1/RTF is not a good
    # result, it is a contradiction -- either the slot is not serialising or the
    # RTF it is being compared against was measured on a busier machine.
    if peak > cap * 1.05:
        print(f"\n    CONTRADICTION: {peak:.2f} a-s/s exceeds the {cap:.2f} ceiling that")
        print( "    one-call-at-a-time allows. Serialisation or the baseline is wrong;")
        print( "    do not quote either number until this closes.")
    print("\n  Throughput flattens at the ceiling no matter how many clients are")
    print("  added; past it every extra client buys queue, not audio. That is the")
    print("  line a batcher has to move, and moving it means more audio per")
    print("  forward pass — not a smarter scheduler.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/tts")
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=6,
                        help="measured requests per client per level")
    parser.add_argument("--warmup", type=int, default=1,
                        help="unmeasured requests per client before the window")
    args = parser.parse_args()

    ns = [int(x) for x in args.levels.split(",")]
    levels: dict[int, list[Result]] = {}
    wall: dict[int, float] = {}

    for n in ns:
        print(f"  running {n} client{'s' if n > 1 else ''} "
              f"x {args.requests} requests...", flush=True)
        levels[n], wall[n] = await level(
            args.url, PARAGRAPH, n, args.requests, args.warmup)

    # The control: the lowest level again, last. See control_check().
    print(f"  running control — {ns[0]} client x {args.requests} requests...",
          flush=True)
    control, _ = await level(args.url, PARAGRAPH, ns[0], args.requests, args.warmup)

    report(levels, wall, control)

    total = sum(len(v) for v in levels.values())
    print(f"\n  {total} requests total. p95 at {min(len(v) for v in levels.values())} "
          f"samples is the second-worst value, not an estimate — read it as a tail")
    print("  indicator. A quotable p99 needs a few hundred requests per level.")


asyncio.run(main())
