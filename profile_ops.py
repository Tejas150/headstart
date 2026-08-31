"""M1, block 2b — what is the 346 ms fixed cost actually made of?

floor.py found ~346 ms per call that does not scale with how much audio is
requested, and showed phonemization is 0.3 ms of it. The rest is inside the
ONNX session. This asks the runtime directly, via its kernel trace
(`SessionOptions.enable_profiling`).

Two passes, because one is not enough to answer the question.

PASS 1 — where does the time go, and is it even math?
    Sum every kernel and compare against the run as a whole. The gap is
    framework overhead: dispatch, allocation, synchronisation. If overhead
    dominates, faster math cannot help and the fix is a different shape.

PASS 2 — which of those operators are the FIXED part?
    An operator whose cost is proportional to audio length is not what
    floor.py measured; it is the slope. Profiling a short and a long clip
    and fitting each operator across the two separates them:

        op_ms = op_fixed + op_slope * audio_seconds

    Operators with a large intercept and a flat slope ARE the floor. That is
    the list worth optimising, and it is not the same as the list of
    operators that cost the most overall.

Note: profiling inflates absolute timings. Proportions and the fixed/variable
split are what this is for, not headline numbers.
"""

import collections
import json
import os

import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
SR = 24000
ITERATIONS = 3

SHORT = "Headstart streams."
FULL = "Headstart streams the first audio chunk before the rest of the clip is generated."

MATMUL_FAMILY = {"MatMul", "Gemm", "Conv", "ConvTranspose", "Einsum"}


def profile(text: str) -> tuple[dict[str, float], float, float, float, int]:
    """Return per-op ms/run, wall ms/run, kernel ms/run, audio seconds, launches."""
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.enable_profiling = True

    session = rt.InferenceSession(MODEL, options,
                                  providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, VOICES)

    kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")  # warm
    for _ in range(ITERATIONS):
        samples, _ = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
    audio_s = len(samples) / SR

    trace_path = session.end_profiling()
    with open(trace_path) as handle:
        events = json.load(handle)
    os.remove(trace_path)

    runs = sorted((e for e in events
                   if e.get("cat") == "Session" and e["name"] == "model_run"),
                  key=lambda e: e["ts"])[-ITERATIONS:]
    start = runs[0]["ts"]

    kernels = [e for e in events
               if e.get("cat") == "Node"
               and e["name"].endswith("kernel_time")
               and e["ts"] >= start]

    by_op: dict[str, float] = collections.defaultdict(float)
    for kernel in kernels:
        by_op[kernel["args"].get("op_name", "?")] += kernel["dur"] / 1000 / ITERATIONS

    wall_ms = sum(r["dur"] for r in runs) / 1000 / ITERATIONS
    kernel_ms = sum(by_op.values())
    return dict(by_op), wall_ms, kernel_ms, audio_s, len(kernels) // ITERATIONS


short_ops, short_wall, short_kernel, short_audio, short_launches = profile(SHORT)
full_ops, full_wall, full_kernel, full_audio, full_launches = profile(FULL)

print("PASS 1 — is the time math, or is it overhead?\n")
print(f"  {'':<18}{'short':>12}{'full':>12}")
print(f"  {'audio':<18}{short_audio:>11.2f}s{full_audio:>11.2f}s")
print(f"  {'wall / run':<18}{short_wall:>10.1f} ms{full_wall:>10.1f} ms")
print(f"  {'kernel / run':<18}{short_kernel:>10.1f} ms{full_kernel:>10.1f} ms")
print(f"  {'framework overhead':<18}{short_wall - short_kernel:>10.1f} ms"
      f"{full_wall - full_kernel:>10.1f} ms")
print(f"  {'  as % of wall':<18}"
      f"{(short_wall - short_kernel) / short_wall * 100:>11.0f}%"
      f"{(full_wall - full_kernel) / full_wall * 100:>11.0f}%")
print(f"  {'kernel launches':<18}{short_launches:>12}{full_launches:>12}")

print("\n\nPASS 2 — fixed vs scaling, per operator"
      f"  (fit across {short_audio:.2f}s and {full_audio:.2f}s)\n")
print(f"  {'operator':<16}{'short':>9}{'full':>9}{'FIXED':>10}{'per audio-s':>13}"
      f"{'verdict':>12}")

span = full_audio - short_audio
rows = []
for op in set(short_ops) | set(full_ops):
    t_short = short_ops.get(op, 0.0)
    t_full = full_ops.get(op, 0.0)
    slope = (t_full - t_short) / span
    fixed = max(0.0, t_short - slope * short_audio)
    rows.append((op, t_short, t_full, fixed, slope))

rows.sort(key=lambda r: r[3], reverse=True)
fixed_total = sum(r[3] for r in rows)
slope_total = sum(r[4] for r in rows)

for op, t_short, t_full, fixed, slope in rows[:12]:
    if fixed < 1 and slope < 1:
        continue
    verdict = "FLOOR" if fixed > slope else "scales"
    print(f"  {op:<16}{t_short:>7.1f}ms{t_full:>7.1f}ms{fixed:>8.1f}ms"
          f"{slope:>11.1f}ms{verdict:>12}")

print(f"\n  fixed cost, all operators ..... {fixed_total:>7.1f} ms"
      f"    (floor.py measured 346 ms unprofiled)")
print(f"  cost per audio-second ......... {slope_total:>7.1f} ms"
      f"    (floor.py measured 374 ms)")

floor_share = {op: f for op, _, _, f, _ in rows}
top_floor = sorted(floor_share.items(), key=lambda kv: kv[1], reverse=True)[:3]
named = ", ".join(f"{op} {f / fixed_total * 100:.0f}%" for op, f in top_floor)
print(f"\n  the floor is mostly: {named}")

mm_fixed = sum(f for op, _, _, f, _ in rows if op in MATMUL_FAMILY)
print(f"\n  quantizable (matmul-family) share of the FLOOR: "
      f"{mm_fixed / fixed_total * 100:.0f}%")
print(f"  -> int8 ceiling on the floor if that math were free: "
      f"{1 / (1 - mm_fixed / fixed_total):.2f}x")
print(f"  -> realistic (~2x on that math only): "
      f"{1 / (1 - mm_fixed / fixed_total / 2):.2f}x")
