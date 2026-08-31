"""M1, block 1 — how long until the listener hears something?

Three ways of producing the exact same audio, measured against one number:
time to first sound.

  A  create()         one call, nothing emitted until the clip is done
  B  create_stream()  the library's own streaming generator
  C  sentence chunks  we split the text and synthesize piece by piece

Run against two inputs: the single sentence M0 benchmarked (so the numbers
tie back to the README), and a three-sentence paragraph.

Two things this is meant to expose.

kokoro-onnx splits on MAX_PHONEME_LENGTH (510) only. That is a guard
against overrunning the model's input window, not a latency feature. Any
text under the limit yields exactly one chunk, so create_stream() behaves
identically to create(). Streaming is not free after all; the chunking has
to be ours.

And chunking can only cut where the text allows. A lone sentence has no
interior boundary, so mode C cannot beat mode A on it. Time to first sound
is set by where you can cut, not by how fast the model runs.

The `lead` column is the point of the project: seconds of audio in hand
minus seconds already played. While it stays positive, the listener never
hears a gap.
"""

import asyncio
import re
import time

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

M0_SENTENCE = (
    "Headstart streams the first audio chunk before the rest of the clip is generated."
)
PARAGRAPH = (
    "Headstart streams the first audio chunk before the rest of the clip is generated. "
    "It is a way to reduce latency and make the model feel more responsive. "
    "The first chunk is generated in parallel with the rest of the clip, so it can be "
    "played back immediately while the rest of the audio is still being generated."
)
VOICE = "af_sarah"
SR = 24000


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


async def compare(kokoro: Kokoro, label: str, text: str, wav_out: str) -> None:
    phonemes = " ".join(kokoro.tokenizer.phonemize(text, "en-us").split())
    batches = len(kokoro._split_phonemes(phonemes))
    parts = sentences(text)

    print(f"\n{'=' * 68}\n{label}")
    print(f"{len(phonemes)} phonemes (limit 510) -> {batches} library batch(es), "
          f"{len(parts)} sentence(s)\n")

    results = {}

    start = time.perf_counter()
    audio, _ = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
    total = time.perf_counter() - start
    results["A  create()"] = (total, total, len(audio) / SR, 1)

    start = time.perf_counter()
    ttfb, collected = None, []
    async for samples, _ in kokoro.create_stream(text, voice=VOICE, speed=1.0, lang="en-us"):
        if ttfb is None:
            ttfb = time.perf_counter() - start
        collected.append(samples)
    total = time.perf_counter() - start
    results["B  create_stream()"] = (ttfb, total, sum(len(c) for c in collected) / SR,
                                     len(collected))

    print(f"  {'chunk':>5}  {'arrived':>10}  {'chunk':>7}  {'ready':>7}  {'lead':>7}")
    start = time.perf_counter()
    ttfb, audio_total, collected = None, 0.0, []
    for i, part in enumerate(parts, 1):
        samples, _ = kokoro.create(part, voice=VOICE, speed=1.0, lang="en-us")
        arrived = time.perf_counter() - start
        if ttfb is None:
            ttfb = arrived
        collected.append(samples)
        audio_total += len(samples) / SR
        lead = audio_total - (arrived - ttfb)
        print(f"  {i:>5}  {arrived * 1000:>7.0f} ms  {len(samples) / SR:>6.2f}s  "
              f"{audio_total:>6.2f}s  {lead:>6.2f}s")
    total = time.perf_counter() - start
    results["C  sentence chunks"] = (ttfb, total, audio_total, len(parts))
    sf.write(wav_out, np.concatenate(collected), SR)

    print(f"\n  {'mode':<20} {'first sound':>12} {'total':>10} {'audio':>8} {'chunks':>7}")
    for name, (t, tot, secs, n) in results.items():
        print(f"  {name:<20} {t * 1000:>9.0f} ms {tot * 1000:>7.0f} ms "
              f"{secs:>7.2f}s {n:>7}")

    base = results["A  create()"][0]
    best = results["C  sentence chunks"][0]
    print(f"\n  first sound {base / best:.1f}x sooner  ->  {wav_out}")


async def main() -> None:
    t0 = time.perf_counter()
    kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
    print(f"cold start {(time.perf_counter() - t0) * 1000:.0f} ms")

    await compare(kokoro, "M0 sentence (matches the README baseline)",
                  M0_SENTENCE, "out_stream_m0.wav")
    await compare(kokoro, "Three-sentence paragraph",
                  PARAGRAPH, "out_stream_paragraph.wav")


asyncio.run(main())
