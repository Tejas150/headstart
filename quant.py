"""M1, block 2c — does int8 actually buy anything, and does it still sound right?

profile_ops.py predicted ~1.27x from the matmul-family share of the floor.
That prediction assumed the whole family is quantizable. It is not:
onnxruntime's dynamic-quantization registry covers

    Attention, Conv, EmbedLayerNormalization, Gather, LSTM, MatMul, Transpose

and NOT ConvTranspose or Gemm. ConvTranspose alone is 34 ms of the 299 ms
floor, so the reachable share is smaller than the family share. This script
recomputes the prediction from the actual registry, then measures.

Dynamic quantization is the right variant here: weights are quantized ahead of
time, activation ranges are computed per-run. No calibration dataset needed,
which matters because we do not have one.

THE PART THAT MATTERS MORE THAN THE SPEEDUP
    A vocoder can be wrecked by quantization while still producing audio of
    the right length. A 1.3x speedup that buzzes is not a win, and a
    benchmark that only prints milliseconds cannot tell you. So this compares
    the int8 waveform against the fp32 one and writes both to disk to listen.

    Caveat on the comparison: Kokoro predicts phoneme durations inside the
    graph. If quantization nudges a duration, the audio shifts in time and
    sample-wise correlation collapses even when it sounds fine. So length
    delta is reported separately, and correlation is only meaningful if the
    lengths match.
"""

import os
import time

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.registry import IntegerOpsRegistry

FP32 = "models/kokoro-v1.0.onnx"
INT8 = "models/kokoro-v1.0-int8.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
SR = 24000
REPEATS = 5
PHYSICAL_CORES = 8

SHORT = "Headstart streams."
FULL = "Headstart streams the first audio chunk before the rest of the clip is generated."

# ms of the 299 ms measured floor, per operator (profile_ops.py, pass 2)
FLOOR_MS = {
    "Sin": 95.5, "Conv": 85.7, "STFT": 35.0, "ConvTranspose": 33.8,
    "Add": 11.8, "Transpose": 5.4, "MatMul": 4.5, "Pow": 3.7,
    "LSTM": 2.9, "Gemm": 2.6, "Resize": 2.4, "Gather": 2.0,
}
FLOOR_TOTAL = 299.0

# Transpose/Gather quantization moves narrower data; it does not make math
# cheaper. Counting them as "accelerated" would flatter the prediction.
REAL_COMPUTE = {"Conv", "MatMul", "LSTM", "Attention", "Gemm", "ConvTranspose"}


def build(path: str) -> Kokoro:
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = PHYSICAL_CORES
    options.inter_op_num_threads = 1
    session = rt.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, VOICES)


def bench(kokoro: Kokoro, text: str) -> tuple[float, np.ndarray]:
    samples, _ = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")  # warm
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        samples, _ = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
        best = min(best, (time.perf_counter() - start) * 1000)
    return best, samples


# ---------------------------------------------------------------- prediction
quantizable = set(IntegerOpsRegistry.keys()) & REAL_COMPUTE
reachable = sum(ms for op, ms in FLOOR_MS.items() if op in quantizable)
unreachable = sum(ms for op, ms in FLOOR_MS.items()
                  if op in REAL_COMPUTE and op not in quantizable)

print("PREDICTION, before measuring\n")
print(f"  compute ops int8 can touch ..... {sorted(quantizable)}")
print(f"  their share of the floor ....... {reachable:.0f} of {FLOOR_TOTAL:.0f} ms"
      f"  ({reachable / FLOOR_TOTAL * 100:.0f}%)")
print(f"  compute ops it cannot .......... {unreachable:.0f} ms"
      f"  (ConvTranspose, Gemm — not in the registry)")
print(f"  -> ceiling if that math were free: "
      f"{1 / (1 - reachable / FLOOR_TOTAL):.2f}x")
print(f"  -> realistic (~2x on that math):  "
      f"{1 / (1 - reachable / FLOOR_TOTAL / 2):.2f}x")
print("\n  Note: the 4800H is Zen 2 — no VNNI int8 acceleration. Expect the")
print("  memory-bandwidth win, not the instruction win. This may undershoot.\n")

# ---------------------------------------------------------------- quantize
# First attempt quantized everything the registry allows. The resulting model
# would not load:
#
#   NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
#
# Every Conv in this graph has rank-3 weights — they are 1-D convolutions
# (88 Conv, 6 ConvTranspose, all rank 3). ORT's ConvInteger CPU kernel is
# 2-D only. So the op holding 86 of the 93 reachable ms cannot be quantized
# on this runtime at all. What is left is MatMul and LSTM: 7 ms of a 299 ms
# floor. We measure it anyway rather than assert the outcome.
OPS = ["MatMul"]

if not os.path.exists(INT8):
    print(f"quantizing (ops: {OPS}) ...")
    started = time.perf_counter()
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8,
                     op_types_to_quantize=OPS)
    print(f"  done in {time.perf_counter() - started:.0f}s")

fp32_mb = os.path.getsize(FP32) / 1024**2
int8_mb = os.path.getsize(INT8) / 1024**2
print(f"\n  model on disk: {fp32_mb:.0f} MB -> {int8_mb:.0f} MB"
      f"  ({fp32_mb / int8_mb:.2f}x smaller)")
print(f"  revised expectation: MatMul+LSTM are ~7 ms of the {FLOOR_TOTAL:.0f} ms"
      f" floor -> at best {1 / (1 - 7 / FLOOR_TOTAL):.2f}x\n")

# ---------------------------------------------------------------- measure
print("\nMEASURED\n")
print(f"  {'':<10}{'short':>12}{'full':>12}{'audio (full)':>15}")

results = {}
audio = {}
for label, path in (("fp32", FP32), ("int8", INT8)):
    kokoro = build(path)
    short_ms, short_wav = bench(kokoro, SHORT)
    full_ms, full_wav = bench(kokoro, FULL)
    results[label] = (short_ms, full_ms)
    audio[label] = (short_wav, full_wav)
    print(f"  {label:<10}{short_ms:>9.0f} ms{full_ms:>9.0f} ms"
          f"{len(full_wav) / SR:>13.2f}s")
    del kokoro

f_short, f_full = results["fp32"]
i_short, i_full = results["int8"]
print(f"\n  speedup: {f_short / i_short:.2f}x short, {f_full / i_full:.2f}x full")

# refit the floor from the two clip lengths, for each precision
print(f"\n  {'':<10}{'FIXED':>10}{'per audio-s':>14}")
for label in ("fp32", "int8"):
    s_ms, f_ms = results[label]
    s_a = len(audio[label][0]) / SR
    f_a = len(audio[label][1]) / SR
    slope = (f_ms - s_ms) / (f_a - s_a)
    fixed = s_ms - slope * s_a
    print(f"  {label:<10}{fixed:>7.0f} ms{slope:>11.0f} ms")

# ---------------------------------------------------------------- quality
print("\n\nQUALITY — does it still sound like speech?\n")
for name, idx in (("short", 0), ("full", 1)):
    a = audio["fp32"][idx].astype(np.float64)
    b = audio["int8"][idx].astype(np.float64)
    print(f"  {name}: fp32 {len(a)} samples, int8 {len(b)} samples", end="")
    if len(a) != len(b):
        print(f"  -> LENGTH DIFFERS by {abs(len(a) - len(b)) / SR * 1000:.0f} ms;"
              " durations shifted, correlation not meaningful")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    else:
        print("  -> same length")
    corr = np.corrcoef(a, b)[0, 1]
    rms_err = np.sqrt(np.mean((a - b) ** 2))
    rms_sig = np.sqrt(np.mean(a ** 2))
    snr = 20 * np.log10(rms_sig / rms_err) if rms_err > 0 else float("inf")
    print(f"    correlation {corr:+.4f}   error-to-signal {snr:>5.1f} dB", end="")
    if corr > 0.99:
        print("   -> effectively identical")
    elif corr > 0.9:
        print("   -> audible difference possible, LISTEN")
    else:
        print("   -> DEGRADED, listen before trusting the speedup")

try:
    import soundfile as sf
    for label in ("fp32", "int8"):
        sf.write(f"out_{label}.wav", audio[label][1], SR)
    print("\n  wrote out_fp32.wav and out_int8.wav — listen to both.")
except ImportError:
    print("\n  (soundfile not installed; skipped writing wavs)")

print("\n  Numbers cannot settle this one. Your ears decide whether int8 ships.")
