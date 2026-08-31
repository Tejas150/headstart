"""M1, block 2b — the fixed cost is inside the session, so tune the session.

floor.py showed time-to-first-sound bottoms out around a 346 ms fixed cost
per call, and that phonemization accounts for 0.3 ms of it. The rest is
inside ONNX Runtime. Before rewriting anything, try the knob that costs
nothing to turn: how many threads the runtime uses for one inference.

The default is one thread per logical core — 16 here. That is tuned for
throughput on a big graph. A small graph can lose to it, because every
parallel section costs a fork and a join, and if the work inside is smaller
than the synchronisation around it, more threads make it slower.

Two chunk sizes are measured, because the answer can differ: a short chunk
(what time-to-first-sound depends on) and the full M0 sentence (what total
throughput depends on).
"""

import time

import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
SR = 24000
REPEATS = 5

SHORT = "Headstart streams."
FULL = "Headstart streams the first audio chunk before the rest of the clip is generated."

THREAD_COUNTS = [1, 2, 4, 6, 8, 12, 16]


def build(threads: int | None) -> Kokoro:
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    session = rt.InferenceSession(MODEL, options,
                                  providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, VOICES)


def best_ms(kokoro: Kokoro, text: str) -> tuple[float, float]:
    kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")  # warm
    best, audio_s = float("inf"), 0.0
    for _ in range(REPEATS):
        start = time.perf_counter()
        samples, _ = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
        best = min(best, (time.perf_counter() - start) * 1000)
        audio_s = len(samples) / SR
    return best, audio_s


print(f"{'threads':>8}  {'short chunk':>13}  {'full sentence':>15}  {'RTF (full)':>11}")

results = []
for threads in [None, *THREAD_COUNTS]:
    kokoro = build(threads)
    short_ms, short_s = best_ms(kokoro, SHORT)
    full_ms, full_s = best_ms(kokoro, FULL)
    label = "default" if threads is None else str(threads)
    results.append((label, short_ms, full_ms))
    print(f"{label:>8}  {short_ms:>10.0f} ms  {full_ms:>12.0f} ms  "
          f"{full_ms / 1000 / full_s:>11.2f}")
    del kokoro

baseline = results[0]
best_short = min(results[1:], key=lambda r: r[1])
best_full = min(results[1:], key=lambda r: r[2])

print(f"\n  short chunk : best at {best_short[0]} threads, "
      f"{best_short[1]:.0f} ms vs {baseline[1]:.0f} ms default "
      f"({baseline[1] / best_short[1]:.2f}x)")
print(f"  full sentence: best at {best_full[0]} threads, "
      f"{best_full[2]:.0f} ms vs {baseline[2]:.0f} ms default "
      f"({baseline[2] / best_full[2]:.2f}x)")
print(f"\n  ({SHORT!r} -> {'short chunk'}, audio length drives the rest)")
