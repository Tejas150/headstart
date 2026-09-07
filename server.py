"""M1, block 3 — the server. Audio leaves the process before it is all made.

Blocks 1 and 2 were measurement. This is the first thing in the repo a
reviewer can actually run, and the first time the project's name is literally
true: the listener gets a head start on audio that does not exist yet.

WHY THE TRANSPORT CAME LAST
    On localhost a WebSocket frame costs single-digit milliseconds. Building
    it first would have meant re-running every benchmark after the real
    optimisation landed. The number was driven to its floor first; this wraps
    a number that is already as good as it is going to get on this machine.

THE THREE DESIGN DECISIONS THAT MATTER HERE

  1. kokoro.create() is synchronous and pins a core for ~600 ms.
     Calling it directly inside an async handler blocks the event loop, and
     every other connected client stops being served -- including the ones
     just waiting to receive bytes already generated. It runs in a worker
     thread via asyncio.to_thread(). This is the single most common way a
     Python inference server is written wrong.

  2. One model call at a time, deliberately (MODEL_SLOT).
     The session is thread-safe, so concurrent Run() calls are legal. They are
     also pointless: intra_op_num_threads=8 already hands one operator all 8
     physical cores, so two concurrent requests do not go faster, they
     interleave and both get slower while the tail gets worse. Serialising
     makes the wait explicit instead of hiding it inside the runtime.

     The cost of that choice is a queue, so the queue is measured: every chunk
     reports queue_ms (waiting for the slot) separately from gen_ms (actual
     model time). Under one client queue_ms is ~0. Under load it is the whole
     story, and it is what M2's batcher exists to fix. A latency number that
     does not separate these two is not a latency number.

  3. Metadata as JSON, audio as raw binary, in two frames.
     Not base64 inside the JSON -- that is +33% bytes on the one payload
     where bytes are latency. int16 rather than float32 halves it again;
     Kokoro emits float32 in [-1, 1] and the conversion is exact enough for
     speech.

  4. A door in front of the slot, and it says no (DOOR, MAX_INFLIGHT).
     bench.py measured the ceiling: this machine sustains ~3.3 realtime
     streams, and past that the fourth listener does not get slower audio, it
     gets a hole in the middle of a sentence. Without a door the server
     accepts the fourth request anyway and breaks it after it has already
     started speaking.

     A wait is legible -- the listener reads it as loading. A stutter is not
     attributable: they cannot tell server load from a broken product, so
     they conclude the product is broken. So: never accept a stream you
     cannot finish cleanly. Past MAX_INFLIGHT a request waits at the door for
     one slot-turnover and is then refused with a retry hint, before a single
     sample has been sent.

     This buys no throughput and is not meant to. Throughput is fixed by the
     graph (finding 10: the ceiling is already reached by one client). The
     door only decides who gets served, and its claim is the tail and the gap
     count, not the median -- which it makes worse on purpose.

     Which makes the order the whole design, and first-come-first-served turns
     out to be the wrong one: a refused caller backs off, and backing off means
     losing your place to the callers who were just served. So refusals age a
     caller up the queue. See Door.

CHUNKING
    Block 1 established sentence boundaries as the cut points. But a lone
    sentence has no interior boundary, so its first chunk is the whole clip
    and TTFB collapses back to non-streaming. lead_words cuts the first
    segment at a comma or conjunction to get sound out sooner, at some cost
    to prosody. The default of 5 was swept, not guessed (leadsweep.py).

    Chunking also has a cost that is invisible in the latency table: the model
    renders a pause between sentences only when it can see the boundary, so
    generating each sentence separately silently deletes it. Measured at 277 ms
    per boundary. The pause is re-inserted as silence, which restores the
    prosody at zero model cost and improves the buffer margin rather than
    spending it. See SENTENCE_GAP_S.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import time

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from kokoro_onnx import Kokoro

# Overridable so the weights can live outside the working directory -- a
# container mounts them at a fixed path, and a K8s volume will not be at ./models.
MODEL = os.environ.get("HEADSTART_MODEL", "models/kokoro-v1.0.onnx")
VOICES = os.environ.get("HEADSTART_VOICES", "models/voices-v1.0.bin")
SAMPLE_RATE = 24000

# Measured in block 2 (isolate.py): on the full sentence intra_op=8 is the
# entire 1.27x and inter_op does nothing; on a short chunk the two interact.
# 8 is the physical core count of the 4800H -- 16 is slower than 8 because two
# SMT threads on one core share a load/store path. Reported as one setting.
#
# 8 is the measured default and stays the default, so an unconfigured run
# reproduces the README. It is overridable because the right number is the
# host's *physical* cores, and under a container CPU limit that is neither 8
# nor what os.cpu_count() reports -- cpu_count sees the host, not the quota,
# so autodetecting here would confidently pick the wrong number.
INTRA_OP_THREADS = int(os.environ.get("HEADSTART_INTRA_OP", "8"))
INTER_OP_THREADS = int(os.environ.get("HEADSTART_INTER_OP", "1"))

# Swept in leadsweep.py (median of 3, not best-of). TTFB falls monotonically as
# the first chunk shrinks -- and tracks block 2's `300 + 374 x audio_s` fit, so
# it is predictable rather than lucky. Below 5 words the median keeps improving
# but the spread explodes: 4 and 3 each threw a 2 s+ outlier in three runs,
# where 8/6/5 threw none. 876 ms with a 115 ms spread beats 625 ms with a 2 s
# tail, because the tail is what a serving SLO is written against.
# Set lead_words: 0 in a request to disable and cut on sentences only.
DEFAULT_LEAD_WORDS = 5

# Measured, not chosen. Rendering the paragraph in one call gives 17.152 s of
# audio; rendering its three sentences separately and summing gives 16.597 s.
# The 555 ms gap is not trimmed edge silence -- leading/trailing silence is
# ~30-90 ms per chunk either way. It is the inter-sentence pause the model
# renders when it can see the sentence boundary, and which chunking destroys,
# because each chunk is generated in isolation and does not know a sentence
# just ended. 555 ms over 2 interior boundaries = 277 ms each.
#
# So the pause is put back on the wire as silence. It costs zero model time,
# and because it is audio handed over for free it *raises* lead rather than
# spending it -- the buffer margin improves while the prosody is restored.
# A clause split inside a sentence gets no gap: there is no pause there to
# restore, and inserting one would be audibly wrong.
SENTENCE_GAP_S = 0.277

# Only one model call runs at a time; see design note 2 above.
MODEL_SLOT = asyncio.Semaphore(1)

# How many requests may be generating at once. Measured, not chosen.
#
# Capacity is 1/RTF = 1/0.304 = 3.3 realtime streams, and bench.py agrees from
# the other direction: throughput flattens at 3.27 audio-seconds per second,
# and buffers first run dry between 3 clients and 4. So 3 is the last level
# that was measured clean -- worst buffer margin +1.85 s, 0 of 18 requests
# stalled -- and 4 is the first that was not: 6 of 24 stalled.
#
# Not 4, for two reasons. The stalls at 4 happen *among the four accepted*, so
# admitting 4 and rejecting the fifth does not fix them; it just relabels a
# broken stream as an accepted one. And 3.3 is not a constant -- it moves with
# text length, voice, speed and what else the box is doing -- so the safe side
# of a line that drifts is below it. 3 leaves ~9% of capacity idle. 4
# overcommits by 21% and a quarter of requests stutter.
MAX_INFLIGHT = 3

# How long a request waits at the door before it is refused. Derived, not
# picked: at MAX_INFLIGHT in flight a request holds its slot for the whole of
# its lifetime, measured at ~15.7 s (10474 ms queued + 5200 ms generating), so
# across 3 slots one comes free every 15.7 / 3 = 5.2 s.
#
# That is the whole argument. If a slot has not come free within the time it
# takes one to come free, you are not next in line -- you are behind someone
# who is, and the wait only grows from here. Waiting longer converts a refusal
# the caller can act on into a timeout it cannot.
DOOR_WAIT_S = 5.2

state: dict = {}


# ------------------------------------------------------------------- door
class Busy(Exception):
    """Refused at the door. Nothing was generated and nothing was sent."""


class Waiter(object):
    __slots__ = ("priority", "seq", "fut")

    def __init__(self, priority: int, seq: int, fut: asyncio.Future) -> None:
        self.priority = priority    # consecutive refusals this caller has taken
        self.seq = seq              # arrival order, breaks ties
        self.fut = fut


class Door(object):
    """Admission control: at most `capacity` requests generating at once.

    Deliberately not asyncio.Semaphore. A semaphore is already a scheduling
    policy -- first-come-first-served, wait forever, no visibility -- it just
    does not look like one, so the policy never gets chosen on purpose. Writing
    the queue out is what makes the policy a decision: refuse rather than queue
    forever, report what is in flight, and change the order without touching
    the handler.

    THE ORDER IS NOT FIFO, AND FIFO IS WHY
        The first version handed each free slot to whoever had waited longest,
        which is the obvious fair answer and is wrong here. A refused caller
        backs off before retrying, and while it is backing off it is not in the
        queue at all -- so it returns behind the callers that were just served,
        who re-queue the instant they finish. Being refused therefore makes the
        next refusal more likely, and the effect compounds.

        Measured, at 8 clients against 3 slots: served counts per client came
        out [4, 4, 4, 3, 2, 1, 0, 0]. Two clients were refused every single
        time. The totals hide this completely -- 18 of 48 served is the same
        number whether the door rotates fairly or picks favourites -- which is
        why bench.py reports the per-client spread and not just the count.

        So the door remembers. Each refusal raises that caller's priority, and
        a free slot goes to the highest priority waiting, arrival order only
        breaking ties. Successful admission resets it to zero. This is ageing:
        the cost of being turned away is paid back the next time you ask, which
        is the property FIFO loses the moment callers back off.

    A fresh arrival still only takes a slot when nobody is waiting at all --
    barging would be faster on average and would do the starving all over again.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_flight = 0
        self.admitted = 0
        self.refused = 0
        self._waiters: list[Waiter] = []
        self._seq = 0

    @property
    def enabled(self) -> bool:
        # capacity 0 turns the door off entirely. The before/after arms of the
        # benchmark then differ by one flag rather than by a code version, so
        # "was it the same build?" stops being a question about the result.
        return self.capacity > 0

    async def enter(self, timeout: float, priority: int = 0) -> float:
        """Take a slot, or raise Busy. Returns seconds spent waiting.

        `priority` is how many times in a row this caller has already been
        refused. Higher goes first.
        """
        if not self.enabled:
            return 0.0
        if self.in_flight < self.capacity and not self._waiters:
            self.in_flight += 1
            self.admitted += 1
            return 0.0

        self._seq += 1
        w = Waiter(priority, self._seq, asyncio.get_running_loop().create_future())
        self._waiters.append(w)
        t0 = time.perf_counter()
        try:
            # If the slot is handed over as the timer fires, wait_for returns
            # the result rather than raising -- so a slot is never granted and
            # then dropped, which would shrink capacity by one permanently.
            await asyncio.wait_for(w.fut, timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ValueError):
                self._waiters.remove(w)
            self.refused += 1
            raise Busy()
        self.admitted += 1
        return time.perf_counter() - t0

    def leave(self) -> None:
        if not self.enabled:
            return
        self.in_flight -= 1
        # Most refused, then longest waiting. The list is at most a few dozen
        # entries, so a scan is cheaper to read than a heap and costs nothing.
        while self._waiters:
            w = min(self._waiters, key=lambda x: (-x.priority, x.seq))
            self._waiters.remove(w)
            if not w.fut.done():        # skip anyone who already timed out
                self.in_flight += 1
                w.fut.set_result(True)
                return


DOOR = Door(MAX_INFLIGHT)


# --------------------------------------------------------------- chunking
def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_lead(segment: str, max_words: int) -> list[str]:
    """Cut an over-long first segment at the last clause boundary that fits.

    Only ever splits the piece the listener is waiting on. Everything after it
    keeps whole-sentence prosody, because nobody is waiting on those.
    """
    words = segment.split()
    if len(words) <= max_words:
        return [segment]
    # prefer a comma/semicolon/colon, else a conjunction, else a hard cut
    for pattern in (r",|;|:", r"\b(and|but|so|because|while|which|that)\b"):
        best = None
        for m in re.finditer(pattern, segment):
            if len(segment[: m.end()].split()) <= max_words:
                best = m.end()
        if best:
            head, tail = segment[:best].strip(), segment[best:].strip()
            if head and tail:
                return [head, tail]
    return [" ".join(words[:max_words]), " ".join(words[max_words:])]


def chunk_text(text: str, lead_words: int | None = None) -> list[tuple[str, float]]:
    """Split into chunks, each paired with the silence to append after it.

    The gap distinguishes the two kinds of cut. A sentence boundary had a pause
    in the un-chunked render (see SENTENCE_GAP_S) and gets it back; a clause
    split inside a sentence did not, and gets nothing. The final chunk gets no
    trailing gap -- the clip is over, and padding the end just delays the close.
    """
    parts = sentences(text)
    if not parts:
        return []
    # Every element is followed by a real sentence boundary except the last.
    chunks = [(p, SENTENCE_GAP_S) for p in parts[:-1]] + [(parts[-1], 0.0)]
    if lead_words:
        head, *rest = split_lead(chunks[0][0], lead_words)
        if rest:
            # The lead split is interior to sentence 0, so the head gets no gap
            # and the tail inherits whatever followed the original sentence.
            chunks = [(head, 0.0), (rest[0], chunks[0][1])] + chunks[1:]
    return chunks


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SAMPLE_RATE)


# ------------------------------------------------------------------ model
def load() -> Kokoro:
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = INTER_OP_THREADS
    session = rt.InferenceSession(MODEL, options, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, VOICES)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    state["kokoro"] = load()
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm the graph. The first call allocates arenas and materialises weights;
    # it is roughly 2x a steady-state call. Paying that at startup instead of
    # letting the first real listener pay it is the whole reason serving stacks
    # keep warm pools.
    t0 = time.perf_counter()
    await asyncio.to_thread(
        state["kokoro"].create, "Warm.", voice="af_sarah", speed=1.0, lang="en-us"
    )
    warm_ms = (time.perf_counter() - t0) * 1000

    print(f"model loaded in {cold_ms:.0f} ms, warmed in {warm_ms:.0f} ms "
          f"(intra_op={INTRA_OP_THREADS}, inter_op={INTER_OP_THREADS})")
    yield
    state.clear()


app = FastAPI(title="headstart", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": "kokoro" in state,
        "sample_rate": SAMPLE_RATE,
        "intra_op_threads": INTRA_OP_THREADS,
        "inter_op_threads": INTER_OP_THREADS,
        "max_inflight": DOOR.capacity,
        "door_wait_s": DOOR_WAIT_S,
        "in_flight": DOOR.in_flight,
        "admitted": DOOR.admitted,
        "refused": DOOR.refused,
    }


async def synth(text: str, voice: str, speed: float) -> tuple[np.ndarray, float, float]:
    """Generate one chunk. Returns (samples, queue_ms, gen_ms)."""
    queued = time.perf_counter()
    async with MODEL_SLOT:
        started = time.perf_counter()
        samples, _ = await asyncio.to_thread(
            state["kokoro"].create, text, voice=voice, speed=speed, lang="en-us"
        )
        done = time.perf_counter()
    return samples, (started - queued) * 1000, (done - started) * 1000


@app.websocket("/tts")
async def tts(ws: WebSocket) -> None:
    await ws.accept()
    # Consecutive refusals on this connection. Lives here rather than in the
    # door because the connection is already the natural identity for a caller,
    # and a per-connection counter cannot leak: it dies with the socket.
    refusals = 0
    try:
        while True:
            request = json.loads(await ws.receive_text())
            t0 = time.perf_counter()

            text = request["text"]
            voice = request.get("voice", "af_sarah")
            speed = float(request.get("speed", 1.0))
            # absent OR null -> server default; 0 -> explicitly off
            lead_words = request.get("lead_words")
            if lead_words is None:
                lead_words = DEFAULT_LEAD_WORDS

            chunks = chunk_text(text, lead_words)

            # The door comes before the `start` frame, so a refusal happens
            # before the client has been told anything is coming. Refusing
            # after `start` would be the failure this is here to prevent, one
            # frame earlier.
            try:
                door_s = await DOOR.enter(DOOR_WAIT_S, priority=refusals)
            except Busy:
                refusals += 1
                await ws.send_text(json.dumps({
                    "type": "busy",
                    "in_flight": DOOR.in_flight,
                    "capacity": DOOR.capacity,
                    # A refusal without a hint just moves the decision to the
                    # caller's guesswork; with one, a client backs off by about
                    # the time it actually takes a slot to free.
                    "retry_after_s": round(DOOR_WAIT_S, 1),
                    "waited_ms": round((time.perf_counter() - t0) * 1000, 1),
                    # Sent so the caller can see it is gaining ground rather
                    # than being ignored, and so the ageing is falsifiable from
                    # the client side instead of being a claim in a comment.
                    "refusals": refusals,
                    "detail": (f"at capacity ({DOOR.capacity} streams); no slot "
                               f"freed in {DOOR_WAIT_S:.1f}s"),
                }))
                continue
            refusals = 0

            try:
                await ws.send_text(json.dumps({
                    "type": "start",
                    "sample_rate": SAMPLE_RATE,
                    "format": "s16le",
                    "chunks": len(chunks),
                    # Waiting at the door, kept apart from queue_ms (waiting for
                    # the slot) and gen_ms (inside the model) for the same
                    # reason those two are kept apart: three different costs.
                    "door_ms": round(door_s * 1000, 1),
                    "in_flight": DOOR.in_flight,
                }))

                ttfb_ms = None
                audio_s = 0.0
                gen_total = 0.0
                for i, (chunk, gap_s) in enumerate(chunks):
                    samples, queue_ms, gen_ms = await synth(chunk, voice, speed)
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    # Restored inter-sentence pause. Free audio: no model time.
                    pcm += silence(gap_s)

                    elapsed = (time.perf_counter() - t0) * 1000
                    if ttfb_ms is None:
                        ttfb_ms = elapsed
                    audio_s += len(samples) / SAMPLE_RATE + gap_s
                    gen_total += gen_ms

                    # `lead` is the point of the whole project: seconds of audio
                    # handed over, minus seconds the listener has already spent
                    # playing. Positive means they never hear a gap.
                    await ws.send_text(json.dumps({
                        "type": "chunk",
                        "index": i,
                        "text": chunk,
                        "bytes": len(pcm),
                        "audio_s": round(len(samples) / SAMPLE_RATE + gap_s, 3),
                        "gap_s": gap_s,
                        "queue_ms": round(queue_ms, 1),
                        "gen_ms": round(gen_ms, 1),
                        "elapsed_ms": round(elapsed, 1),
                        "lead_s": round(audio_s - (elapsed - ttfb_ms) / 1000, 3),
                    }))
                    await ws.send_bytes(pcm)

                total_ms = (time.perf_counter() - t0) * 1000
                await ws.send_text(json.dumps({
                    "type": "end",
                    "ttfb_ms": round(ttfb_ms or 0.0, 1),
                    "total_ms": round(total_ms, 1),
                    "door_ms": round(door_s * 1000, 1),
                    "audio_s": round(audio_s, 3),
                    # RTF = generation ÷ audio. Below 1.0 is faster than realtime.
                    "rtf": round(total_ms / 1000 / audio_s, 3) if audio_s else None,
                    # What the transport and the framework cost on top of the model.
                    "overhead_ms": round(total_ms - gen_total, 1),
                }))
            finally:
                # Held for the whole request, not per chunk. A slot released
                # between chunks would let a new stream in to compete with one
                # already mid-sentence, which is the situation the door exists
                # to prevent. `finally` so a disconnect mid-generation returns
                # the slot instead of leaking capacity one client at a time.
                DOOR.leave()
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    # Loopback by default: this binds a model that answers to anyone who can
    # reach it, so exposing it is an explicit act. In a container the interface
    # is the container's own, so the entrypoint passes --host 0.0.0.0 there.
    parser.add_argument("--host", default=os.environ.get("HEADSTART_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("HEADSTART_PORT", "8000")))
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT,
                        help="streams served at once; 0 turns the door off")
    parser.add_argument("--door-wait", type=float, default=DOOR_WAIT_S,
                        help="seconds to wait at the door before being refused")
    args = parser.parse_args()

    DOOR.capacity = args.max_inflight
    DOOR_WAIT_S = args.door_wait
    print(f"door: max_inflight={DOOR.capacity or 'off'}, wait={DOOR_WAIT_S:.1f}s")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
