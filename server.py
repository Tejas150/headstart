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
import re
import time

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
SAMPLE_RATE = 24000

# Measured in block 2 (isolate.py): on the full sentence intra_op=8 is the
# entire 1.27x and inter_op does nothing; on a short chunk the two interact.
# 8 is the physical core count of the 4800H -- 16 is slower than 8 because two
# SMT threads on one core share a load/store path. Reported as one setting.
INTRA_OP_THREADS = 8
INTER_OP_THREADS = 1

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

state: dict = {}


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
            await ws.send_text(json.dumps({
                "type": "start",
                "sample_rate": SAMPLE_RATE,
                "format": "s16le",
                "chunks": len(chunks),
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
                "audio_s": round(audio_s, 3),
                # RTF = generation ÷ audio. Below 1.0 is faster than realtime.
                "rtf": round(total_ms / 1000 / audio_s, 3) if audio_s else None,
                # What the transport and the framework cost on top of the model.
                "overhead_ms": round(total_ms - gen_total, 1),
            }))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
