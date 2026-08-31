"""M1, block 2a — which knob actually bought the 1.34x?

threads.py compared a default session against sessions that set
intra_op_num_threads AND inter_op_num_threads=1 together. That is two
variables moved at once, so the speedup could not be attributed to either.
A number you cannot attribute is a number an interviewer can take apart.

Four configurations, one variable at a time:

  default        neither knob set
  intra only     intra_op_num_threads = 8   (8 = physical cores)
  inter only     inter_op_num_threads = 1
  both           what threads.py actually measured

intra_op splits ONE operation across cores — a single matrix multiply
divided into pieces. inter_op runs INDEPENDENT operations concurrently.
They are different kinds of parallelism and there is no reason to assume
the win came from the one we assumed.
"""

import time

import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
SR = 24000
REPEATS = 5
PHYSICAL_CORES = 8

SHORT = "Headstart streams."
FULL = "Headstart streams the first audio chunk before the rest of the clip is generated."

CONFIGS = [
    ("default", None, None),
    ("intra=8 only", PHYSICAL_CORES, None),
    ("inter=1 only", None, 1),
    ("both", PHYSICAL_CORES, 1),
]


def build(intra: int | None, inter: int | None) -> Kokoro:
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra is not None:
        options.intra_op_num_threads = intra
    if inter is not None:
        options.inter_op_num_threads = inter
    session = rt.InferenceSession(MODEL, options,
                                  providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, VOICES)


def best_ms(kokoro: Kokoro, text: str) -> float:
    kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")  # warm
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
        best = min(best, (time.perf_counter() - start) * 1000)
    return best


print(f"physical cores {PHYSICAL_CORES}, best of {REPEATS}\n")
print(f"  {'config':<14} {'short chunk':>12} {'full sentence':>14}  vs default")

results = {}
for label, intra, inter in CONFIGS:
    kokoro = build(intra, inter)
    short_ms = best_ms(kokoro, SHORT)
    full_ms = best_ms(kokoro, FULL)
    results[label] = (short_ms, full_ms)

    if label == "default":
        note = "—"
    else:
        base_s, base_f = results["default"]
        note = f"{base_s / short_ms:.2f}x short, {base_f / full_ms:.2f}x full"
    print(f"  {label:<14} {short_ms:>9.0f} ms {full_ms:>11.0f} ms  {note}")
    del kokoro

base_s, base_f = results["default"]
intra_s = results["intra=8 only"][0]
inter_s = results["inter=1 only"][0]
both_s = results["both"][0]

print("\n  attribution on the short chunk (the TTFB-relevant one):")
print(f"    intra_op alone accounts for  {base_s - intra_s:>6.0f} ms of the "
      f"{base_s - both_s:.0f} ms total gain")
print(f"    inter_op alone accounts for  {base_s - inter_s:>6.0f} ms")

if abs((base_s - intra_s) + (base_s - inter_s) - (base_s - both_s)) > 30:
    print("    -> the two do NOT simply add; they interact. Report 'both' as "
          "one setting, not two independent wins.")
else:
    print("    -> the effects are roughly additive.")
