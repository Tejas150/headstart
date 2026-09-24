"""M1, block 3: the server. Audio leaves the process before it is all made.

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

  2. A measured number of model calls at once, not an assumed one (MODEL_SLOT).
     The session is thread-safe, so concurrent Run() calls are legal. The first
     version ran one at a time on the argument that intra_op_num_threads=8
     already hands one operator all 8 physical cores, leaving a second caller
     nothing to use. Finding 18 measures that: the operators that dominate this
     graph saturate neither compute nor bandwidth, so one call leaves a third of
     the box idle and a second one fills it. Four clients with the door open go
     from 3.44x delivered to 5.12x, and from 6 of 12 streams running dry to
     none. MODEL_WIDTH is the setting; 1 restores the serialised behaviour.

     Calls past the last free slot still queue, so the queue is measured: every
     chunk reports queue_ms (waiting for a slot) separately from gen_ms (actual
     model time). Under one client queue_ms is ~0. Under load it is the whole
     story. A latency number that does not separate these two is not a latency
     number.

  3. Metadata as JSON, audio as raw binary, in two frames.
     Not base64 inside the JSON -- that is +33% bytes on the one payload
     where bytes are latency. int16 rather than float32 halves it again;
     Kokoro emits float32 in [-1, 1] and the conversion is exact enough for
     speech.

  4. Admission control in front of the slot, and it says no (ADMISSION,
     MAX_INFLIGHT).
     The ceiling is measured, not chosen: the box sustains about two realtime
     streams once the per-chunk fixed cost is counted, and past that the next
     listener does not get slower audio, they get a gap in the middle of a
     sentence -- and so does everyone already speaking. With no admission
     limit the server accepts that request anyway and breaks it after it has
     already started speaking.

     A wait is legible -- the listener reads it as loading. A stutter is not
     attributable: they cannot tell server load from a broken product, so
     they conclude the product is broken. So: never accept a stream you
     cannot finish cleanly. Past MAX_INFLIGHT a request waits for admission
     for one slot-turnover and is then refused with a retry hint, before a
     single sample has been sent.

     This buys no capacity and is not meant to. Ordering decides who gets
     served, never how much the box finishes -- what moves that is how much of
     the machine runs at once (MODEL_WIDTH, finding 18), which is a different
     lever. Admission control's claim is the tail and the underrun count, not
     the median, which it makes worse on purpose.

     Which makes the order the whole design, and first-come-first-served turns
     out to be the wrong one: a refused caller backs off, and backing off means
     losing your place to the callers who were just served. So refusals age a
     caller up the queue. See AdmissionControl.

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
import collections
import contextlib
import json
import os
import re
import time

from pathlib import Path

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from kokoro_onnx import Kokoro

# Resolved from this file, not the working directory. experiments/_root.py
# chdirs, the container runs from /app, and a reader may run from anywhere --
# so cwd is the one thing that cannot be relied on to find a file that ships
# alongside the source.
HERE = Path(__file__).resolve().parent

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

# How many model calls may run at once. 1 was the original, on the belief that
# intra_op=8 already hands every operator all 8 cores, so a second caller finds
# nothing idle and only makes both tails worse. Finding 18 measures that and
# finds it false: the operators that dominate this graph are bound by neither
# compute nor bandwidth, so one call leaves the box partly idle and a second
# fills it. Under bench.py with admission control off, four clients lose 6 of 12
# streams to underruns at width 1 and none at width 4.
#
# Derived from the thread count rather than typed, because the two answer the
# same question -- how much machine is there -- and the operator already has to
# set INTRA_OP_THREADS correctly for a container quota. Half is the measured
# point at 8 physical cores; at other sizes the ratio is an extrapolation and
# not something this repo has measured. 1 restores the serialised behaviour and
# is the honest setting on a 1-2 core box, where there is nothing to fill.
MODEL_WIDTH = int(os.environ.get("HEADSTART_MODEL_WIDTH",
                                 max(1, INTRA_OP_THREADS // 2)))

# How much working time the delivered-audio measurement keeps, and how much it
# needs before it will answer at all. 60 s is a few dozen chunks here, long
# enough that one unusual voice does not move it and short enough that it
# follows a change in width or thread count within a minute. Below 5 s the
# sample is a handful of chunks and the answer is noise, so it declines to give
# one and capacity falls back to the per-call number.
BUSY_WINDOW_S = 60.0
BUSY_MIN_S = 5.0

# Swept in leadsweep.py (median of 3, not best-of). TTFB falls monotonically as
# the first chunk shrinks -- and tracks block 2's `300 + 374 x audio_s` fit, so
# it is predictable rather than lucky. Below 5 words the median keeps improving
# but the spread explodes: 4 and 3 each threw a 2 s+ outlier in three runs,
# where 8/6/5 threw none. 876 ms with a 115 ms spread beats 625 ms with a 2 s
# tail, because the tail is what a serving SLO is written against.
# Set lead_words: 0 in a request to disable and cut on sentences only.
#
# Now derived rather than typed: derived_lead_words() inverts the cost model
# against TTFB_TARGET_S and lands on the same 5. This stays as the value used
# when the model has not been fitted yet.
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
class ModelSlot(object):
    """A bounded number of model calls at once, and the order in which is a decision.

    This was an asyncio.Semaphore(1), which hands the slot to whoever asked
    first. That is the same mistake FIFO makes at admission, one layer down:
    arrival order is not need order. Two streams sharing this slot are not
    symmetric -- one can have five seconds of audio buffered and the other
    none -- and giving the slot to whichever happened to call first is what
    leaves the other one silent. Measured: with two callers, one finished with
    no underruns and the other lost 2.8 s, and which one it was changed run to
    run. Total silence barely moved; it just picked a different victim.

    So the queue is ordered by buffer in hand, least first. A stream about to
    run dry goes ahead of one that is comfortable. That is earliest-deadline-
    first under another name, and it is the right rule here because the buffer
    is exactly the deadline: it is how long this stream can wait before the
    listener hears silence.

    Chunk 0 of a new stream has no buffer at all, so it sorts to the front for
    free. That is the chunk-0 priority tracked separately as task #19 -- it is
    not a special case, it is what the general rule does with a zero.

    WIDTH, AND WHY IT IS NOT 1
        It was 1, on the argument that intra_op=8 already hands every operator
        all 8 physical cores, so a second concurrent call finds no idle cores to
        use and only makes both tails worse. The first half of that is true and
        the conclusion does not follow. Finding 5 placed every operator against
        the machine's two ceilings and found `Sin`, `ConvTranspose` and `STFT`
        -- 255 ms of the profile -- bound by neither compute nor bandwidth. Work
        that saturates nothing leaves the box idle, and a second call is exactly
        what fills it.

        The number that settles it is a matched pair through this server, same
        binary and one flag apart: four clients with admission control off
        deliver 3.44x realtime at width 1 and 5.12x at width 4.

        It is not free. Per-chunk generation time rises from 4850 ms to 13131 ms
        at the median, so width trades chunk latency for delivered audio. The
        trade is worth taking because the alternative under load is not a fast
        call, it is a queued one: at width 1 those four clients wait 14760 ms
        for the slot and 6 of 12 streams run their buffer dry, and at width 4
        they wait 0 ms and none do. The buffer is what decides, and it drains at
        the same rate whether the chunk is waiting its turn or being generated
        slowly.

        Width is concurrency inside one session, not a second copy of the model.
        Sharding the cores across separate processes reaches roughly the same
        ceiling -- experiments/sharding.py measures both arms -- but pays 326 MB
        of weights per extra shard to get there. On a box where memory is the
        binding constraint, which is most rented CPU, the shared session is the
        better buy.

    The ordering rule earns its place the moment width exceeds 1. `preempted`
    counts every time need order beat arrival order, and serialised it stayed at
    0 at this server's real capacity: two admitted streams are never both
    waiting when only one call can run. Widen the slot and waiters become the
    normal condition rather than a misconfiguration, which is the situation the
    rule was written for and could not previously demonstrate.
    """

    def __init__(self, width: int = 1) -> None:
        self.width = max(1, width)
        self._running = 0
        self._waiters: list[list] = []      # [buffer_s, seq, future]
        self._seq = 0
        self.preempted = 0                  # times need order beat arrival order
        # Delivered audio per second of wall clock while the model was working
        # at all. Distinct from the cost model's realtime_speed, which is per
        # call: widening the slot makes every individual call slower and the
        # box as a whole faster, so a per-call number moves the wrong way and
        # capacity cannot be read off it. Only stretches of time with at least
        # one call running are counted, so an idle server does not read as slow.
        self._busy_since: float | None = None
        self.busy_s = 0.0
        self.busy_audio_s = 0.0

    @contextlib.asynccontextmanager
    async def acquire(self, buffer_s: float = 0.0):
        if self._running < self.width and not self._waiters:
            self._enter()
        else:
            self._seq += 1
            fut = asyncio.get_running_loop().create_future()
            self._waiters.append([max(0.0, buffer_s), self._seq, fut])
            await fut                       # the releaser enters on our behalf
        try:
            yield
        finally:
            self._release()

    def _enter(self) -> None:
        if self._running == 0:
            self._busy_since = time.perf_counter()
        self._running += 1

    def record_audio(self, audio_s: float) -> None:
        """One chunk's worth of delivered audio, against the busy clock."""
        self.busy_audio_s += audio_s
        # Keep roughly the last BUSY_WINDOW_S of working time. Scaling both
        # halves by the same factor leaves the ratio alone and lets the number
        # follow the box, which matters because width and voice both move it.
        if self.busy_s > BUSY_WINDOW_S:
            scale = BUSY_WINDOW_S / self.busy_s
            self.busy_s *= scale
            self.busy_audio_s *= scale

    def aggregate_speed(self) -> float | None:
        """Audio delivered per second of working time, or None until measured."""
        if self.busy_s < BUSY_MIN_S or self.busy_audio_s <= 0:
            return None
        return self.busy_audio_s / self.busy_s

    def _release(self) -> None:
        self._running -= 1
        if self._running == 0 and self._busy_since is not None:
            self.busy_s += time.perf_counter() - self._busy_since
            self._busy_since = None
        # Least buffer first, arrival order breaking ties. A scan, not a heap:
        # the list is one entry per admitted stream, so it is a handful. One
        # waiter woken per release, because one call's worth of room came free.
        while self._waiters:
            w = min(self._waiters, key=lambda x: (x[0], x[1]))
            if w is not min(self._waiters, key=lambda x: x[1]):
                self.preempted += 1
            self._waiters.remove(w)
            if not w[2].done():
                self._enter()
                w[2].set_result(True)
                return


MODEL_SLOT = ModelSlot(MODEL_WIDTH)


# ------------------------------------------------------------- cost model
# Generating a chunk costs a fixed amount plus an amount per second of audio:
#
#     gen_s = FIXED + SLOPE * audio_s
#
# The two halves are different things. FIXED is per-call setup that a shorter
# chunk cannot avoid, which is why chunking has a floor and why chunk 0 is never
# free. SLOPE is the marginal cost of one more second of speech -- it is the
# real RTF, and everything below is derived from these two numbers rather than
# picked.
#
# These are seeds, not the answer. CostModel refits them from live traffic and
# converges to 341 ms + 0.351 over 64 samples, so what the server actually runs
# on is measured on the box it is running on. The seeds are set to that
# converged point so the first few requests are not planned against a shape the
# machine does not have.
FIT_FIXED_S = 0.341
FIT_SLOPE = 0.351

# Audio seconds per word. Taken as proportional rather than fitted with an
# intercept: a line through the measured points has a 0.7 s intercept that is
# edge silence, and applying it to a five-word chunk overpredicts by 35%. Also
# refitted live, and converges to 0.306.
S_PER_WORD = 0.306

# What the listener should wait for the first sound. Chunk 0 is sized to fit
# inside this, and it is the only parameter here that is a choice rather than a
# measurement -- it is a product decision about how long a pause feels like a
# fault. Everything else follows from it and the fit above.
TTFB_TARGET_S = 1.0

# Only spend this fraction of the buffer on generating the next chunk. At 1.0
# each chunk is timed to land exactly as the previous one runs out, so any
# jitter is an underrun; 0.7 keeps a third of the buffer as margin and still lets
# the safe chunk size grow geometrically, so the cap stops binding after about
# three chunks.
BUFFER_SAFETY = 0.7

# Memory of the rolling queue-wait estimate, roughly the last ten chunks. Faster
# than the admission hold average because contention is what this tracks: a
# burst that has arrived matters to the chunk being planned right now, and an
# estimate that lags the burst plans against a box that no longer exists.
QUEUE_ALPHA = 0.2

# One switch that turns the queue term off, for the same reason `--max-inflight
# 0` exists: the claim here is a comparison, and a comparison that needs two
# builds is a comparison nobody reproduces. Off means the planner sees only
# generation time, which is what it did before -- and what makes the buffer run
# dry with several callers.
QUEUE_AWARE = os.environ.get("HEADSTART_QUEUE_AWARE", "1") != "0"

# How many requests may be generating at once. A seed, not the answer.
#
# The answer is derived_capacity(): audio-seconds produced per second of model
# time, read straight off the live fit window and rounded down. On this box it
# settles at 2. This constant only covers the cold start, before any traffic
# has been seen, and admission control walks away from it one slot at a time as
# the measurement arrives.
#
# The level above the line is not a slightly worse version of the level below
# it. Admitting one stream too many does not degrade that stream, it breaks
# every stream on the box: pinned one over, three callers lose 2.7-21.9 s of
# audio to buffer underruns, and every run puts underruns in all three.
# Admission control only retunes once the fit has seen a spread of chunk sizes,
# so whatever is typed here is what the first arrivals get -- which is why it is
# the low end of what has been measured, and the measurement is allowed to raise
# it rather than the other way round.
MAX_INFLIGHT = 2

# How long a request waits at admission control before it is refused.
#
# The rule: wait for the time it takes one slot to come free, and no longer. At
# capacity a request holds its slot for its whole lifetime, so across `capacity`
# slots one frees every hold_time / capacity seconds. If a slot has not come
# free in that window you are not next in line -- you are behind someone who is,
# and the wait only grows from here. Waiting longer converts a refusal the
# caller can act on into a timeout it cannot.
#
# The number below is only the seed. Hold time is not a constant -- it moves
# with text length, voice, speed and what else the box is doing -- so a fixed
# wait is calibrated to exactly one workload and wrong for every other. 5.2 came
# from a 15.7 s hold (10474 ms queued + 5200 ms generating) over 3 slots, which
# is right for that text and far too long for a one-sentence request. So
# admission control measures its own hold time instead and recomputes the wait;
# this value is what it uses until it has seen enough requests to know better.
ADMISSION_WAIT_SEED_S = 5.2

# Refuse no faster than this. Guards the first few samples and any burst of very
# short requests from making admission control hair-trigger.
ADMISSION_WAIT_MIN_S = 1.0

# And no slower. Past this the refusal has stopped being information the caller
# can act on, whatever the arithmetic says.
ADMISSION_WAIT_MAX_S = 15.0

# Weight on the newest hold-time sample. 0.2 gives a memory of roughly the last
# ten requests: long enough that one outlier does not move the wait, short
# enough that a change in workload reaches it within seconds.
ADMISSION_WAIT_ALPHA = 0.2

class CostModel(object):
    """gen_s = fixed + slope * audio_s, refitted from live synthesis.

    Seeded from the offline fit so the first request is planned sensibly, then
    kept current by least squares over recent chunks. Refitting matters because
    the coefficients are properties of this box under this load -- a slower CPU,
    a different voice, or another tenant moves both of them, and a planner
    working from stale coefficients quietly plans underruns.
    """

    def __init__(self, fixed: float, slope: float, s_per_word: float) -> None:
        self.fixed = fixed
        self.slope = slope
        self.s_per_word = s_per_word
        self.queue_s = 0.0
        self._pts: collections.deque = collections.deque(maxlen=64)

    def record(self, audio_s: float, gen_s: float, words: int) -> None:
        if audio_s <= 0 or gen_s <= 0:
            return
        self._pts.append((audio_s, gen_s))
        if words > 0:
            self.s_per_word += 0.1 * (audio_s / words - self.s_per_word)
        self._refit()

    def record_queue(self, queue_s: float) -> None:
        """Track waiting-for-a-slot separately from generating.

        These are two different facts and merging them breaks both. Queue time
        must stay out of the fit, or a busy box reads as a slow box and the
        coefficients inflate themselves. But it must be in the plan, because the
        listener's buffer drains on wall-clock time and does not care which part
        of the wait was queueing. So it is measured here and added back at
        planning time, never at fitting time.
        """
        if not QUEUE_AWARE:
            return
        self.queue_s += QUEUE_ALPHA * (max(0.0, queue_s) - self.queue_s)

    def _refit(self) -> None:
        # Least squares needs spread on the x axis to separate the fixed cost
        # from the slope. Ten chunks that are all the same length determine a
        # point, not a line -- fitting through them would hand the whole cost to
        # whichever term the noise happened to favour. So refit only once the
        # sample actually covers a range.
        if len(self._pts) < 8:
            return
        xs = [p[0] for p in self._pts]
        ys = [p[1] for p in self._pts]
        if max(xs) - min(xs) < 1.0:
            return
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        if var <= 0:
            return
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
        fixed = my - slope * mx
        # A negative slope or fixed cost is arithmetically possible from noisy
        # samples and physically meaningless; keep the last good fit instead.
        if slope > 0.01 and fixed >= 0:
            self.slope, self.fixed = slope, max(0.0, fixed)

    @property
    def samples(self) -> int:
        return len(self._pts)

    def gen_s(self, audio_s: float) -> float:
        """Generation cost alone. This is the fit, and it is queue-free."""
        return self.fixed + self.slope * audio_s

    def mean_gen_s(self) -> float:
        """Average time one chunk holds the model slot."""
        if not self._pts:
            return self.gen_s(1.0)
        return sum(p[1] for p in self._pts) / len(self._pts)

    def queue_estimate(self, in_flight: int = 0) -> float:
        """Expected wait for the model slot, for a stream about to be planned.

        Two sources, and the larger wins. The rolling average is what queueing
        has actually cost lately; the count is what it is about to cost. Only
        the count sees a burst arrive, because the average is by construction a
        few chunks behind and a burst is over before it catches up -- which is
        why the first version still underran the second caller's buffer even
        though the queue term was already in the planner.

        The count is simple because the model slot is: one call at a time, so a
        stream with `in_flight` others ahead of it waits behind that many chunks.
        """
        return max(self.queue_s, max(0, in_flight) * self.mean_gen_s())

    def plan_s(self, audio_s: float, queue_s: float | None = None) -> float:
        """What the listener actually waits: the queue, then the generation."""
        q = self.queue_s if queue_s is None else queue_s
        return q + self.gen_s(audio_s)

    def max_audio_s(self, budget_s: float, queue_s: float | None = None) -> float:
        """Longest chunk that arrives within `budget_s`. Zero if none fits.

        The queue comes out of the budget before the chunk is sized, which is
        what makes the planner contention-aware: as the box fills, the same
        buffer buys a shorter chunk.
        """
        q = self.queue_s if queue_s is None else queue_s
        return max(0.0, (budget_s - q - self.fixed) / self.slope)

    def min_audio_s(self, queue_s: float | None = None) -> float:
        """Shortest chunk that leaves the listener better off than it found them.

        A chunk earns `a` seconds of buffer and spends `queue + fixed + slope*a`
        getting there, so the buffer grows only while

            a * (1 - slope) > fixed + queue

        Below that line every chunk is a net withdrawal and the stream drains no
        matter how carefully the next one is sized. This is the reason the naive
        fix is backwards: under contention the per-chunk *ceiling* falls, but the
        *floor* rises, because each extra chunk pays the queue again. Cutting
        smaller to catch up is exactly wrong.

        When the floor rises above the ceiling there is no sustainable chunk size
        at all -- which is a statement about admission, not about chunking, and
        is what `sustainable()` reports to the admission layer.
        """
        if self.slope >= 1.0:
            return float("inf")     # slower than realtime; nothing is sustainable
        q = self.queue_s if queue_s is None else queue_s
        return (self.fixed + q) / (1.0 - self.slope)

    def realtime_speed(self) -> float:
        """Seconds of audio produced per second of model time, measured.

        Published as `realtime_speed` on /health. 2.45 means the server makes
        audio 2.45x faster than it plays. The `rtf` on the `end` frame is the
        same idea inverted -- wall time over audio, for one request -- but not
        the same number: `rtf` carries the queue and admission wait, this counts
        model time only. Idle they nearly agree; under load `rtf` climbs and
        this does not.

        Not 1/slope. Slope is the marginal cost of one more second of speech and
        therefore the capacity of an infinitely long chunk; this server serves
        short ones and pays `fixed` on every single call, so the capacity it
        actually has is lower than the asymptote by however much chunking it is
        doing. On this box 1/slope says 3 streams and the buffers break at 3 --
        the missing stream is the fixed cost, charged once per chunk.

        Measured straight off the samples already in the fit: total audio made
        over total time spent making it. Falls back to the asymptote until the
        window has something in it.
        """
        if not self._pts:
            return 1.0 / self.slope if self.slope > 0 else 1.0
        audio = sum(p[0] for p in self._pts)
        gen = sum(p[1] for p in self._pts)
        return audio / gen if gen > 0 else 1.0

    def sustainable(self, budget_s: float) -> bool:
        """Is there any chunk size that both fits the budget and grows the buffer?

        Floor above ceiling means no. That is not a chunking failure and cannot
        be chunked around: the box is taking in more work than it can finish,
        and the only lever left is admission control.
        """
        return self.min_audio_s() < self.max_audio_s(budget_s)


COST = CostModel(FIT_FIXED_S, FIT_SLOPE, S_PER_WORD)

state: dict = {}


# ------------------------------------------------------ admission control
class Busy(Exception):
    """Refused before admission. Nothing was generated and nothing was sent."""


class Waiter(object):
    __slots__ = ("priority", "seq", "fut")

    def __init__(self, priority: int, seq: int, fut: asyncio.Future) -> None:
        self.priority = priority    # consecutive refusals this caller has taken
        self.seq = seq              # arrival order, breaks ties
        self.fut = fut


class AdmissionControl(object):
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
        number whether the server rotates fairly or picks favourites -- which is
        why bench.py reports the per-client spread and not just the count.

        So admission control remembers. Each refusal raises that caller's
        priority, and a free slot goes to the highest priority waiting, arrival
        order only breaking ties. Admission resets it to zero. This is ageing:
        the cost of being turned away is paid back the next time you ask, which
        is the property FIFO loses the moment callers back off.

    A fresh arrival still only takes a slot when nobody is waiting at all --
    barging would be faster on average and would do the starving all over again.
    """

    def __init__(self, capacity: int, seed_hold_s: float | None = None) -> None:
        self.capacity = capacity
        # An operator who names a number owns it. Auto-tracking is for the
        # default, where the alternative is a constant typed on a different box.
        self.pinned = False
        self.in_flight = 0
        self.admitted = 0
        self.refused = 0
        self._waiters: list[Waiter] = []
        self._seq = 0
        # Seeded as a hold time, not as a wait, so that changing capacity at
        # runtime moves the wait immediately instead of waiting for the average
        # to catch up.
        seed = seed_hold_s if seed_hold_s is not None else ADMISSION_WAIT_SEED_S * max(1, capacity)
        self.hold_s = seed
        self.hold_samples = 0

    def record_hold(self, held_s: float) -> None:
        """One completed request's slot occupancy, folded into the average."""
        self.hold_s += ADMISSION_WAIT_ALPHA * (held_s - self.hold_s)
        self.hold_samples += 1
        self.retune()

    def retune(self) -> None:
        """Move capacity toward what the box is measured to sustain.

        One stream over the measured line is the whole of the multi-stream
        silence. Pinned at 3, three callers lose 2.7-21.9 s to underruns and all
        three streams break on every run; at 2 they lose 0-2.3 s and the third is
        refused before a sample is sent. Nothing about the third stream's
        chunking is fixable, because the box is not making audio fast enough to
        keep three buffers full, and a stream admitted into that is a stream
        accepted and not finished -- which is the one thing this server claims
        not to do.

        One slot at a time, and only once the fit has seen a spread of chunk
        sizes. Capacity is a measurement here, and a measurement that jumps on
        every sample is a measurement nobody can act on. Shrinking never evicts
        anyone: it refuses the next arrival, which is what refusal is for.
        """
        if self.pinned or not self.enabled or COST.samples < 16:
            return
        want = derived_capacity()
        if want > self.capacity:
            self.capacity += 1
        elif want < self.capacity:
            self.capacity -= 1

    @property
    def wait_s(self) -> float:
        """How long the next arrival waits before being refused.

        One slot-turnover: the observed hold time divided across the slots. The
        clamp is not a fudge -- it is the range outside which a refusal stops
        being useful, at one end because it fires before the server has learned
        anything, at the other because the caller has already given up.
        """
        if not self.enabled:
            return 0.0
        turnover = self.hold_s / self.capacity
        return min(ADMISSION_WAIT_MAX_S, max(ADMISSION_WAIT_MIN_S, turnover))

    @property
    def enabled(self) -> bool:
        # capacity 0 turns admission control off entirely. The before/after arms
        # of the benchmark then differ by one flag rather than by a code
        # version, so
        # "was it the same build?" stops being a question about the result.
        return self.capacity > 0

    async def enter(self, timeout: float | None = None, priority: int = 0) -> float:
        """Take a slot, or raise Busy. Returns seconds spent waiting.

        `priority` is how many times in a row this caller has already been
        refused. Higher goes first. `timeout` defaults to the measured
        slot-turnover time; pass a number only to override it.
        """
        if not self.enabled:
            return 0.0
        if timeout is None:
            timeout = self.wait_s
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

    def leave(self, held_s: float | None = None) -> None:
        if not self.enabled:
            return
        if held_s is not None:
            self.record_hold(held_s)
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


ADMISSION = AdmissionControl(MAX_INFLIGHT)



# --------------------------------------------------------------- chunking
def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


BOUNDARIES = ((r",|;|:", True),
              (r"\b(and|but|so|because|while|which|that)\b", False))


def split_lead(segment: str, max_words: int) -> list[str]:
    """Cut an over-long segment at a clause boundary. Never mid-phrase.

    Two boundary kinds, and the cut goes on opposite sides of them. A comma or
    semicolon closes the clause before it, so the cut goes after the mark. A
    conjunction opens the clause after it -- "and", "because", "which" all
    belong to what follows -- so the cut goes before the word. Cutting after a
    conjunction strands it at the end of the head, where the voice has nothing
    left to attach it to.

    When no boundary fits inside the budget, the cut overshoots to the first
    one past it rather than cutting on the word count. The budget is a latency
    target and the boundary is an audible fact, so the budget is what gives.
    Overshooting costs TTFB in proportion to how far away the next breath is;
    cutting to the number produces "Streaming speech is a scheduling", a
    fragment with no prosodic close, and that is heard on every request. On the
    demo text the two rules are 1.03 s and 1.74 s to first audio, and this is
    why the headline TTFB is the larger number.

    A segment with no boundary anywhere is returned whole: it cannot be divided
    without mangling, and a late chunk beats a broken one.
    """
    words = segment.split()
    if len(words) <= max_words:
        return [segment]
    # Inside the budget, prefer a comma to a conjunction: both fit, so take the
    # stronger prosodic break. Past the budget the preference inverts to plain
    # distance -- every word of overshoot is TTFB spent, and a comma twenty
    # words out costs more than a conjunction ten words out is worth. Ranking
    # by kind here is what made the demo open on a 21-word chunk, 2.5 s to
    # first audio, when a "that" at word 10 was sitting in front of it.
    for pattern, after in BOUNDARIES:
        best = None
        for m in re.finditer(pattern, segment):
            pos = m.end() if after else m.start()
            if pos and len(segment[:pos].split()) <= max_words:
                best = pos              # keep looking: want the last that fits
        if best:
            head, tail = segment[:best].strip(), segment[best:].strip()
            if head and tail:
                return [head, tail]

    nearest = None
    for pattern, after in BOUNDARIES:
        for m in re.finditer(pattern, segment):
            pos = m.end() if after else m.start()
            if pos and len(segment[:pos].split()) > max_words:
                if nearest is None or pos < nearest:
                    nearest = pos
                break                   # matches are in order; first is nearest
    if nearest:
        head, tail = segment[:nearest].strip(), segment[nearest:].strip()
        if head and tail:
            return [head, tail]
    return [segment]


def derived_lead_words(fit: "CostModel | None" = None) -> int:
    """How many words fit in the TTFB budget.

    Invert the cost model: the audio that can be generated in TTFB_TARGET_S is
    (target - fixed) / slope, and words are that over S_PER_WORD.

    The floor is not arithmetic. Below about 4 words the median TTFB keeps
    improving but the spread explodes -- 4 and 3 each threw a 2 s outlier in
    three runs where 8/6/5 threw none -- and a serving SLO is written against
    the tail, not the median. So the fit picks the number and the measured
    variance sets the floor under it.
    """
    f = fit or COST
    return max(4, min(30, round(f.max_audio_s(TTFB_TARGET_S) / f.s_per_word)))


def derived_capacity(fit: "CostModel | None" = None) -> int:
    """How many streams can be served at once.

    One stream needs one second of speech per second of wall clock, so the
    machine carries as many streams as it makes audio-seconds per second.
    Rounded down, because the level above the line is not a slightly worse
    version of the level below it -- it is the level where buffers run dry, and
    a stream that stalls has failed rather than degraded.

    Read off delivered audio per second of working time when the slot has
    measured that, and off the per-call fit until it has. The two agree at width
    1 and must not be interchanged above it: widening the slot makes every
    individual call slower and the box as a whole faster, so the per-call number
    falls exactly when capacity rises. Capacity is a claim about what the
    machine can deliver, so it is measured at the machine.

    Measured delivered speed, not 1 / SLOPE. The asymptote reads 3.2 on this box
    and 3 is exactly where the buffers break: pinned at 3, three callers lose
    2.7-21.9 s of audio to underruns and every run damages all three streams; at
    2, two callers lose 0-2.3 s and the third is refused before a sample is
    sent. What the asymptote leaves out is the per-call fixed cost, which this
    server pays on every chunk because it chunks for TTFB -- so 1 / slope is the
    capacity of an infinitely long chunk, and this server serves short ones on
    purpose.
    """
    f = fit or COST
    delivered = MODEL_SLOT.aggregate_speed() if fit is None else None
    return max(1, int(delivered if delivered is not None else f.realtime_speed()))


def chunk_text(text: str, lead_words: int | None = None,
               in_flight: int = 0) -> list[tuple[str, float]]:
    """Split into chunks, each paired with the silence to append after it.

    Every chunk is bounded by the audio already in the listener's hands, not
    just the first one. The old version sized chunk 0 for TTFB and then sent
    whole sentences after it, which is fine until a sentence is long: a 10.4 s
    sentence takes 3.8 s to generate, and against a 1.6 s buffer that is 2.2 s
    of silence in the middle of the speech. Bounding only the opener optimises the
    moment the listener notices least and ignores the one they notice most.

    So the planner carries the buffer forward. A chunk is admitted only if the
    model says it generates in less than BUFFER_SAFETY of what is in hand;
    otherwise it is cut at a clause boundary until it is. The bound relaxes as
    it goes -- each chunk played is buffer earned -- so it stops binding after
    about three chunks and the rest of the text keeps whole-sentence prosody.

    The gap distinguishes the two kinds of cut. A sentence boundary had a pause
    in the un-chunked render (see SENTENCE_GAP_S) and gets it back; a clause
    split inside a sentence did not, and gets nothing. The final chunk gets no
    trailing gap -- the clip is over, and padding the end just delays the close.
    """
    parts = sentences(text)
    if not parts:
        return []
    if lead_words is None:
        lead_words = derived_lead_words()

    # Fixed for the whole plan, read once at the moment the request arrives.
    # Re-reading it per chunk would let the plan wobble with other people's
    # arrivals mid-sentence, and a plan that changes shape under you is worse
    # than one that is slightly stale.
    queue_s = COST.queue_estimate(in_flight)

    # Merge before splitting. A very short sentence is the same problem as an
    # over-long one seen from the other side: "Hello there." is 0.6 s of audio
    # that took 0.7 s to make, so it hands the listener a buffer smaller than
    # the fixed cost of the next call, and the next chunk is late however small
    # it is cut. Splitting cannot fix that -- splitting pays the fixed cost more
    # often -- so the opener is grown instead, by gluing whole sentences on
    # until it is worth generating. Merged sentences are rendered in one call,
    # so the pause between them is the model's own and needs no SENTENCE_GAP_S.
    floor = lead_words or 1
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1].split()) < floor:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    parts = merged

    out: list[tuple[str, float]] = []
    buffer_s = 0.0
    for si, part in enumerate(parts):
        tail_gap = SENTENCE_GAP_S if si < len(parts) - 1 else 0.0
        pending = [part]
        while pending:
            seg = pending.pop(0)
            if not out and lead_words:
                budget_words = lead_words          # chunk 0: sized for TTFB
            elif lead_words:
                # Everything after: sized for the buffer, which the listener
                # earned by waiting through the chunks before it.
                #
                # Two bounds, not one. The ceiling is the buffer -- do not
                # promise more audio than there is time to make. The floor is
                # the net-withdrawal line: a chunk shorter than min_audio_s
                # costs more wall-clock than the audio it returns, so it drains
                # the buffer however carefully it was sized. Under contention
                # the floor rises while the ceiling falls, and when they cross
                # there is no chunk size that works -- at which point the floor
                # wins, because if the stream is going to fall behind it should
                # at least stop paying the fixed cost twice as often.
                # BUFFER_SAFETY appears on both bounds and means the same thing
                # both times: leave a margin. On the ceiling it spends only 70%
                # of the buffer in hand. On the floor it divides, because
                # min_audio_s is break-even -- a chunk sized exactly there grows
                # the buffer by nothing, and the first jitter is an underrun. Sizing
                # 1/0.7 above break-even is what makes it grow.
                safe_s = max(COST.max_audio_s(buffer_s * BUFFER_SAFETY, queue_s),
                             COST.min_audio_s(queue_s) / BUFFER_SAFETY)
                budget_words = max(lead_words, int(safe_s / COST.s_per_word))
            else:
                budget_words = 0                   # splitting disabled entirely

            if budget_words and len(seg.split()) > budget_words:
                head, *rest = split_lead(seg, budget_words)
                if rest:
                    pending.insert(0, rest[0])
                    seg = head

            # An interior cut keeps the sentence's gap for whichever piece ends
            # it; the pieces before that end mid-sentence and get nothing.
            gap = tail_gap if not pending else 0.0
            out.append((seg, gap))
            audio_s = len(seg.split()) * COST.s_per_word + gap
            # Drain on plan_s, not gen_s. This is the whole of the multi-stream
            # silence: the fit is deliberately queue-free, so on a busy box the
            # planner believed each chunk arrived a queue-wait earlier than it
            # did, and carried a buffer forward that the listener never had.
            buffer_s = max(0.0, buffer_s - COST.plan_s(audio_s, queue_s)) + audio_s
    return out


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
        "max_inflight": ADMISSION.capacity,
        "capacity_pinned": ADMISSION.pinned,
        # Live, not configured: the wait is recomputed from measured hold time
        # on every arrival, so this moves with the workload. hold_s and
        # hold_samples are here so the number can be checked rather than
        # believed -- wait_s should always be hold_s / max_inflight, clamped.
        "admission_wait_s": round(ADMISSION.wait_s, 2),
        "hold_s": round(ADMISSION.hold_s, 2),
        "hold_samples": ADMISSION.hold_samples,
        # The cost model everything above is derived from, and the numbers it
        # produces. Published so the derivation can be checked rather than taken
        # on trust: capacity is measured delivered speed (see derived_capacity,
        # and note it is NOT 1/slope), and lead words are
        # (ttfb_target - fixed) / slope / s_per_word.
        "fit_fixed_s": round(COST.fixed, 3),
        "fit_slope": round(COST.slope, 3),
        "s_per_word": round(COST.s_per_word, 3),
        "fit_samples": COST.samples,
        "queue_s": round(COST.queue_s, 3),
        "min_chunk_s": round(COST.min_audio_s(), 2),
        "sustainable": COST.sustainable(TTFB_TARGET_S * 4),
        "ttfb_target_s": TTFB_TARGET_S,
        "lead_words": derived_lead_words(),
        # Two different speeds, and mixing them up is the easiest mistake here.
        # realtime_speed is per call: how much audio one model call makes per
        # second it spends making it. delivered_speed is per machine: audio
        # produced per second the model was working at all, however many calls
        # were running. They agree at model_width 1 and separate above it, in
        # opposite directions -- each call gets slower, the box gets faster.
        # Capacity comes off the second one. Null until 5 s of working time.
        "realtime_speed": round(COST.realtime_speed(), 2),
        "model_width": MODEL_SLOT.width,
        "delivered_speed": (round(MODEL_SLOT.aggregate_speed(), 2)
                            if MODEL_SLOT.aggregate_speed() is not None else None),
        "busy_s": round(MODEL_SLOT.busy_s, 1),
        "derived_capacity": derived_capacity(),
        "queue_aware": QUEUE_AWARE,
        "slot_preempted": MODEL_SLOT.preempted,
        "in_flight": ADMISSION.in_flight,
        "admitted": ADMISSION.admitted,
        "refused": ADMISSION.refused,
    }


@app.get("/demo")
async def demo() -> FileResponse:
    """The console. Same origin as /tts, so the page needs no CORS and no
    configuration -- `docker run` and open this URL is the whole setup."""
    return FileResponse(HERE / "demo" / "index.html")


@app.post("/admission")
async def set_admission(capacity: int) -> dict:
    """Change admission capacity at runtime. 0 turns admission control off, -1
    hands it back to the measurement.

    This is a control-plane endpoint on a data-plane server, which is a real
    smell, and it is here for one reason: admission control's whole claim is a
    comparison, and a comparison the reader has to restart the server to see is
    a comparison they will not run. It is the same argument that put
    `--max-inflight 0` in the benchmark -- one build, one flag, so "was it the
    same server?" is not a question about the result.

    Naming a number pins it, because an operator who names a number owns it.
    -1 is the way back: it clears the pin and lets capacity track measured
    delivered speed again, so a reader who has just pinned capacity open to
    watch it fail can return to the default without a restart.

    The cost is named rather than hidden: this mutates serving behaviour and is
    unauthenticated. It is acceptable because the server binds loopback by
    default, and it is the first thing to put behind auth or a build flag if
    this were ever exposed. Capacity is otherwise a deploy-time decision.
    """
    if capacity < 0:
        ADMISSION.pinned = False
        ADMISSION.capacity = derived_capacity()
        return {"capacity": ADMISSION.capacity, "in_flight": ADMISSION.in_flight,
                "pinned": False}
    ADMISSION.capacity = capacity
    ADMISSION.pinned = True
    return {"capacity": ADMISSION.capacity, "in_flight": ADMISSION.in_flight, "pinned": True}


async def synth(text: str, voice: str, speed: float,
                buffer_s: float = 0.0) -> tuple[np.ndarray, float, float]:
    """Generate one chunk. Returns (samples, queue_ms, gen_ms).

    `buffer_s` is how much audio this stream has already handed the listener,
    and it is the stream's place in the queue: least first. See ModelSlot.
    """
    queued = time.perf_counter()
    async with MODEL_SLOT.acquire(buffer_s):
        started = time.perf_counter()
        samples, _ = await asyncio.to_thread(
            state["kokoro"].create, text, voice=voice, speed=speed, lang="en-us"
        )
        done = time.perf_counter()
    # Feed the planner. Only the time inside the slot counts: queueing is not a
    # cost of generating this chunk, and folding it in would make the model
    # think the box got slower every time it got busier.
    COST.record(len(samples) / SAMPLE_RATE, done - started, len(text.split()))
    MODEL_SLOT.record_audio(len(samples) / SAMPLE_RATE)
    # The wait for a slot is recorded separately and added back at planning
    # time. It is not a property of the model, but the listener's buffer drains
    # through it all the same.
    COST.record_queue(started - queued)
    return samples, (started - queued) * 1000, (done - started) * 1000


@app.websocket("/tts")
async def tts(ws: WebSocket) -> None:
    await ws.accept()
    # Consecutive refusals on this connection. Lives here rather than in
    # AdmissionControl because the connection is already the natural identity
    # for a caller, and a per-connection counter cannot leak: it dies with the
    # socket.
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
                lead_words = derived_lead_words()

            # Planned before admission control, so `in_flight` is who is already
            # generating -- exactly the contention this stream is about to meet.
            chunks = chunk_text(text, lead_words, in_flight=ADMISSION.in_flight)

            # Admission control comes before the `start` frame, so a refusal happens
            # before the client has been told anything is coming. Refusing
            # after `start` would be the failure this is here to prevent, one
            # frame earlier.
            wait_s = ADMISSION.wait_s
            try:
                admission_wait_s = await ADMISSION.enter(wait_s, priority=refusals)
            except Busy:
                refusals += 1
                await ws.send_text(json.dumps({
                    "type": "busy",
                    "in_flight": ADMISSION.in_flight,
                    "capacity": ADMISSION.capacity,
                    # A refusal without a hint just moves the decision to the
                    # caller's guesswork; with one, a client backs off by about
                    # the time it actually takes a slot to free. This is the
                    # current measured slot turnover, not a constant, so the
                    # hint tracks the load the caller is actually retrying into.
                    "retry_after_s": round(wait_s, 1),
                    "waited_ms": round((time.perf_counter() - t0) * 1000, 1),
                    # Sent so the caller can see it is gaining ground rather
                    # than being ignored, and so the ageing is falsifiable from
                    # the client side instead of being a claim in a comment.
                    "refusals": refusals,
                    "detail": (f"at capacity ({ADMISSION.capacity} streams); no slot "
                               f"freed in {wait_s:.1f}s"),
                }))
                continue
            refusals = 0
            # Occupancy starts when the slot is taken, not when the request
            # arrived: time spent waiting for admission is not time a slot was
            # held, and folding it in would inflate the average that sets the
            # wait, which would then inflate itself.
            t_admit = time.perf_counter()

            try:
                await ws.send_text(json.dumps({
                    "type": "start",
                    "sample_rate": SAMPLE_RATE,
                    "format": "s16le",
                    "chunks": len(chunks),
                    # What chunk 0 was actually sized to, echoed back. The caller
                    # may send lead_words or leave it out; either way the number
                    # the planner used is the one worth reporting, and a client
                    # that reads it off /health instead is reading a value that
                    # may have been refit since this request landed.
                    "lead_words": lead_words,
                    # Waiting for admission, kept apart from queue_ms (waiting
                    # for the slot) and gen_ms (inside the model) for the same
                    # reason those two are kept apart: three different costs.
                    "admission_wait_ms": round(admission_wait_s * 1000, 1),
                    "in_flight": ADMISSION.in_flight,
                }))

                ttfb_ms = None
                audio_s = 0.0
                gen_total = 0.0
                for i, (chunk, gap_s) in enumerate(chunks):
                    # Buffer in hand at the moment this chunk asks for the slot:
                    # audio handed over, minus audio already played. Chunk 0 is
                    # zero, which is why a new stream goes first.
                    lead_s = 0.0 if ttfb_ms is None else (
                        audio_s - (time.perf_counter() * 1000 - t0 * 1000 - ttfb_ms) / 1000)
                    samples, queue_ms, gen_ms = await synth(chunk, voice, speed, lead_s)
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
                    "admission_wait_ms": round(admission_wait_s * 1000, 1),
                    "audio_s": round(audio_s, 3),
                    # RTF = generation ÷ audio. Below 1.0 is faster than realtime.
                    "rtf": round(total_ms / 1000 / audio_s, 3) if audio_s else None,
                    # What the transport and the framework cost on top of the model.
                    "overhead_ms": round(total_ms - gen_total, 1),
                }))
            finally:
                # Held for the whole request, not per chunk. A slot released
                # between chunks would let a new stream in to compete with one
                # already mid-sentence, which is the situation admission
                # control exists to prevent. `finally` so a disconnect returns
                # the slot instead of leaking capacity one client at a time.
                ADMISSION.leave(time.perf_counter() - t_admit)
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
    parser.add_argument("--max-inflight", type=int, default=None,
                        help=("streams served at once; 0 turns admission "
                              "control off. Omit and capacity tracks measured "
                              "delivered speed"))
    parser.add_argument("--admission-wait", type=float,
                        default=ADMISSION_WAIT_SEED_S,
                        help=("seed slot-turnover in seconds; the server "
                              "replaces this with its own measurement as "
                              "requests complete"))
    parser.add_argument("--model-width", type=int, default=MODEL_WIDTH,
                        help=("model calls allowed to run at once; 1 "
                              "serialises them. Above 1 each call is slower "
                              "and the box delivers more audio"))
    args = parser.parse_args()

    MODEL_SLOT.width = max(1, args.model_width)
    if args.max_inflight is not None:
        ADMISSION.capacity = args.max_inflight
        ADMISSION.pinned = True
    # Given as a turnover, stored as a hold time, because that is the quantity
    # admission control actually measures.
    ADMISSION.hold_s = args.admission_wait * max(1, ADMISSION.capacity)
    print(f"admission: max_inflight={ADMISSION.capacity or 'off'}"
          f"{'' if ADMISSION.pinned else ' (seed; tracks delivered audio speed)'}, "
          f"wait={ADMISSION.wait_s:.1f}s (seed; recomputed from live hold time), "
          f"model_width={MODEL_SLOT.width}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
