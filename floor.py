"""M1, block 2 — how low can time-to-first-sound actually go?

Block 1 established that chunking is what buys latency: cut the text sooner
and the listener hears sound sooner. The obvious next question is how far
that goes. If a smaller first chunk is always faster, is the floor zero?

It is not. Every synthesis call pays two costs:

  fixed     phonemization, tokenization, style lookup, ONNX session
            dispatch — paid once per call regardless of length
  variable  the forward pass itself, proportional to audio produced

Chunking smaller shrinks the variable part and leaves the fixed part alone,
so time-to-first-sound approaches the fixed cost and stops. That floor is
the number this script measures, by synthesizing progressively longer
prefixes of one sentence and fitting a line through the results:

    synthesis_ms = fixed + slope * audio_seconds

The intercept is the floor. The slope is the real-time factor in disguise
(slope/1000 = RTF), which is a useful cross-check against the README's 0.41.

Also measured separately: phonemization, which happens on the text before
the model is touched at all and is therefore pure fixed cost.
"""

import statistics
import time

import numpy as np
from kokoro_onnx import Kokoro

VOICE = "af_sarah"
SR = 24000
REPEATS = 3

# Progressive prefixes of the M0 sentence, so every measurement is drawn from
# the same text the README baseline used.
WORDS = ("Headstart streams the first audio chunk before the rest of the "
         "clip is generated.").split()
PREFIXES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]


def timed(fn, repeats=REPEATS):
    """Best-of-N wall time in ms, plus the return value. Best-of, not mean:
    we want the floor, and the slow runs are scheduler noise, not the model."""
    best, value = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        best = min(best, (time.perf_counter() - start) * 1000)
    return best, value


def main() -> None:
    t0 = time.perf_counter()
    kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
    print(f"cold start {(time.perf_counter() - t0) * 1000:.0f} ms  "
          f"(paid once per process, excluded from everything below)\n")

    # The first inference of a process is slower — ONNX Runtime allocates its
    # arenas and the OS faults the weights in. Burn one so it does not
    # contaminate the shortest prefix.
    kokoro.create("Warm up.", voice=VOICE, speed=1.0, lang="en-us")

    print(f"  {'words':>5}  {'phon':>5}  {'phonemize':>10}  {'synth':>9}  "
          f"{'audio':>7}  {'RTF':>5}")

    rows = []
    for n in PREFIXES:
        text = " ".join(WORDS[:n])
        if not text.endswith((".", "!", "?")):
            text += "."

        phon_ms, phonemes = timed(
            lambda t=text: kokoro.tokenizer.phonemize(t, "en-us"))

        synth_ms, (samples, _) = timed(
            lambda t=text: kokoro.create(t, voice=VOICE, speed=1.0,
                                         lang="en-us"))

        audio_s = len(samples) / SR
        rows.append((n, len(phonemes), phon_ms, synth_ms, audio_s))
        print(f"  {n:>5}  {len(phonemes):>5}  {phon_ms:>7.1f} ms  "
              f"{synth_ms:>6.0f} ms  {audio_s:>6.2f}s  {synth_ms / 1000 / audio_s:>5.2f}")

    audio = np.array([r[4] for r in rows])
    synth = np.array([r[3] for r in rows])
    slope, intercept = np.polyfit(audio, synth, 1)

    predicted = slope * audio + intercept
    residual = float(np.max(np.abs(synth - predicted)))

    phon_median = statistics.median(r[2] for r in rows)

    print(f"\n  fit: synth_ms = {intercept:.0f} + {slope:.0f} * audio_seconds"
          f"   (max residual {residual:.0f} ms)")
    print(f"  implied RTF from slope: {slope / 1000:.2f}"
          f"   (README measured 0.41)")
    print(f"\n  fixed cost per call ...... {intercept:>6.0f} ms   <- the floor")
    print(f"  of which phonemization ... {phon_median:>6.1f} ms")

    if intercept > 0:
        budget = (400 - intercept) / slope
        print(f"\n  to hit 400 ms time-to-first-sound, the first chunk may be"
              f" at most {budget:.2f}s of audio")
        print(f"  ({budget * SR:.0f} samples; roughly "
              f"{budget / 0.35:.0f}-{budget / 0.25:.0f} words at normal pace)")


main()
