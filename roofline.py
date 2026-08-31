"""M1, block 2d — is the floor the hardware, or is it the runtime?

Everything so far says WHERE the time goes. Nothing so far says WHY it takes
that long, and without that there is no scaling formula. "How fast a machine
do I need for 200 ms" has three different answers:

    memory-bandwidth-bound -> scales with GB/s
    compute-bound          -> scales with cores x clock x vector width
    runtime-bound          -> scales with almost nothing; fix the kernel

Roofline settles it. Measure what the machine can actually do, measure what
the inference actually achieves, and see which ceiling it is sitting under.
If it is under neither, the ceiling is software.

FOUR MEASUREMENTS

  1. Machine peaks. Threaded STREAM-triad for bandwidth, large SGEMM for
     FLOPS. Measured on this box, not read off a spec sheet — spec sheets
     quote theoretical numbers no real code reaches.

  2. Per-operator arithmetic intensity, from the profiler's tensor shapes.
     FLOPs / bytes decides which roof an operator can possibly hit. The
     ridge point (peak_flops / peak_bw) is the break-even intensity.

  3. Sin, head to head against numpy. Sin is 32% of the floor. Elementwise
     sine is trivially vectorizable, so if ORT is much slower than numpy on
     the identical buffer, that time is a kernel problem and no amount of
     hardware fixes it.

  4. Thread scaling, 1 to 8. Compute-bound work scales close to linearly.
     Memory-bound work flattens early because the cores queue for one bus.
     The shape of the curve is a second, independent vote.

A prediction that does not survive all four is not a prediction worth
booking a cloud instance on.
"""

import collections
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
TEXT = "Headstart streams."
CORES = 8
REPEATS = 5

ELEMENTWISE = {"Sin", "Cos", "Add", "Mul", "Sub", "Div", "Pow", "Exp", "Log",
               "Sqrt", "Tanh", "Sigmoid", "Relu", "LeakyRelu", "Erf", "Neg"}


def _prod(shape):
    n = 1
    for d in shape:
        n *= d if isinstance(d, int) else 1
    return n


def _shapes(entry):
    out = []
    for item in entry or []:
        for _, shape in item.items():
            out.append(shape)
    return out


# ============================================================ 1. machine peaks
def peak_bandwidth(threads: int = CORES) -> float:
    """Threaded copy, a[:] = b[:]. Returns GB/s.

    A STREAM triad in numpy needs two ufunc calls (multiply, then add),
    which is five passes over memory, not the three the classic formula
    counts. Counting three understates bandwidth by 1.67x — and since the
    whole scaling argument divides by this number, that error would
    propagate straight into the hardware recommendation. A copy is two
    unambiguous passes: one read, one write.
    """
    n = 32 * 1024 * 1024  # 128 MB per array, far past the 8 MB L3
    a = np.empty(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    chunk = n // threads

    def copy(i):
        s = slice(i * chunk, (i + 1) * chunk)
        np.copyto(a[s], b[s])

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(copy, range(threads)))  # warm
        best = float("inf")
        for _ in range(5):
            t = time.perf_counter()
            list(pool.map(copy, range(threads)))
            best = min(best, time.perf_counter() - t)
    return (2 * n * 4) / best / 1e9  # 1 read + 1 write


def peak_gflops() -> float:
    """Peak fp32 throughput.

    numpy's bundled BLAS is not necessarily a good one, and if it is slower
    than onnxruntime's own GEMM then using it as 'peak' makes the model look
    like it exceeds the machine's capability — which is nonsense and a sign
    the baseline is wrong, not the measurement. So take the best of numpy's
    SGEMM and onnxruntime's own MatMul kernel on the same problem.
    """
    n = 2048
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)

    a @ b  # warm
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t)
    numpy_gf = (2 * n ** 3) / best / 1e9

    # same GEMM, through onnxruntime
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph(
        [node], "gemm",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [n, n]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [n, n])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [n, n])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10  # onnx 1.22 emits IR 13; this runtime caps at 11
    options = rt.SessionOptions()
    options.intra_op_num_threads = CORES
    options.inter_op_num_threads = 1
    sess = rt.InferenceSession(model.SerializeToString(), options,
                               providers=["CPUExecutionProvider"])
    sess.run(None, {"A": a, "B": b})  # warm
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        sess.run(None, {"A": a, "B": b})
        best = min(best, time.perf_counter() - t)
    ort_gf = (2 * n ** 3) / best / 1e9

    print(f"   (numpy BLAS {numpy_gf:.0f} GFLOP/s, onnxruntime GEMM "
          f"{ort_gf:.0f} GFLOP/s — taking the higher as peak)")
    return max(numpy_gf, ort_gf)


print("1. WHAT THIS MACHINE CAN ACTUALLY DO\n")
bw = peak_bandwidth()
gf = peak_gflops()
ridge = gf / bw
print(f"   memory bandwidth (triad, {CORES} threads) .. {bw:>7.1f} GB/s")
print(f"   compute (SGEMM 2048, BLAS) ............... {gf:>7.1f} GFLOP/s")
print(f"   ridge point = compute / bandwidth ........ {ridge:>7.1f} FLOP/byte")
print(f"   -> an operator below {ridge:.0f} FLOP/byte can only ever be "
      "bandwidth-bound;")
print("      above it, only compute-bound.")


# ================================================ 2. per-operator from a trace
def trace_once():
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = CORES
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    session = rt.InferenceSession(MODEL, options,
                                  providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, VOICES)
    kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")  # warm
    kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
    path = session.end_profiling()
    with open(path) as handle:
        events = json.load(handle)
    os.remove(path)
    runs = sorted((e for e in events if e.get("cat") == "Session"
                   and e["name"] == "model_run"), key=lambda e: e["ts"])
    start = runs[-1]["ts"]
    return [e for e in events if e.get("cat") == "Node"
            and e["name"].endswith("kernel_time") and e["ts"] >= start]


def flops_and_bytes(args):
    op = args.get("op_name", "?")
    ins = _shapes(args.get("input_type_shape"))
    outs = _shapes(args.get("output_type_shape"))
    nbytes = 4 * (sum(_prod(s) for s in ins) + sum(_prod(s) for s in outs))
    out_n = sum(_prod(s) for s in outs)

    if op == "Conv" and len(ins) >= 2 and len(ins[1]) == 3:
        oc, icg, k = ins[1]
        flops = 2 * oc * icg * k * (out_n // oc if oc else 0)
    elif op == "ConvTranspose" and len(ins) >= 2 and len(ins[1]) == 3:
        ic, ocg, k = ins[1]
        in_len = _prod(ins[0]) // ic if ic else 0
        flops = 2 * ic * ocg * k * in_len
    elif op in ("MatMul", "Gemm") and len(ins) >= 2:
        kdim = ins[0][-1] if ins[0] else 1
        flops = 2 * out_n * (kdim if isinstance(kdim, int) else 1)
    elif op == "LSTM" and len(ins) >= 3:
        flops = 2 * (_prod(ins[1]) + _prod(ins[2]))
    elif op in ELEMENTWISE:
        flops = out_n
    else:
        flops = 0
    return op, flops, nbytes


kernels = trace_once()
agg = collections.defaultdict(lambda: [0.0, 0, 0, 0])  # ms, flops, bytes, calls
for k in kernels:
    op, fl, by = flops_and_bytes(k["args"])
    row = agg[op]
    row[0] += k["dur"] / 1000
    row[1] += fl
    row[2] += by
    row[3] += 1

total_ms = sum(r[0] for r in agg.values())
print("\n\n2. WHAT THE INFERENCE ACHIEVES, PER OPERATOR\n")
print(f"   {'op':<15}{'ms':>8}{'GFLOP/s':>10}{'GB/s':>9}{'intensity':>11}"
      f"{'bound by':>14}")

for op, (ms, fl, by, calls) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:10]:
    if ms < 3:
        continue
    achieved_gf = fl / (ms / 1000) / 1e9 if ms else 0
    achieved_bw = by / (ms / 1000) / 1e9 if ms else 0
    intensity = fl / by if by else 0
    roof = min(gf, bw * intensity) if intensity else bw
    util = max(achieved_gf / gf, achieved_bw / bw) * 100
    if util < 25:
        verdict = "NEITHER"
    elif intensity > ridge:
        verdict = "compute"
    else:
        verdict = "bandwidth"
    print(f"   {op:<15}{ms:>8.1f}{achieved_gf:>10.1f}{achieved_bw:>9.1f}"
          f"{intensity:>11.1f}{verdict:>14}")

tot_fl = sum(r[1] for r in agg.values())
tot_by = sum(r[2] for r in agg.values())
print(f"\n   whole inference: {tot_fl / 1e9:.2f} GFLOP, {tot_by / 1e9:.2f} GB "
      f"moved, {total_ms:.0f} ms")
print(f"   achieved {tot_fl / (total_ms / 1000) / 1e9:>6.1f} GFLOP/s "
      f"of {gf:.0f} peak  ({tot_fl / (total_ms / 1000) / 1e9 / gf * 100:.0f}% of compute)")
print(f"   achieved {tot_by / (total_ms / 1000) / 1e9:>6.1f} GB/s    "
      f"of {bw:.0f} peak  ({tot_by / (total_ms / 1000) / 1e9 / bw * 100:.0f}% of bandwidth)")


# ================================================== 3. Sin, head to head
print("\n\n3. THE Sin KERNEL, AGAINST NUMPY ON THE SAME BUFFER\n")
sin_events = [k for k in kernels if k["args"].get("op_name") == "Sin"]
if sin_events:
    shapes = [tuple(_shapes(k["args"]["output_type_shape"])[0]) for k in sin_events]
    common = collections.Counter(shapes).most_common(1)[0][0]
    ort_ms = sum(k["dur"] for k in sin_events
                 if tuple(_shapes(k["args"]["output_type_shape"])[0]) == common) \
        / 1000 / max(1, shapes.count(common))
    n = _prod(list(common))
    buf = np.random.rand(n).astype(np.float32)
    out = np.empty_like(buf)

    np.sin(buf, out=out)  # warm
    t = time.perf_counter()
    for _ in range(10):
        np.sin(buf, out=out)
    np1 = (time.perf_counter() - t) / 10 * 1000

    chunk = n // CORES
    def _sin(i):
        s = slice(i * chunk, (i + 1) * chunk)
        np.sin(buf[s], out=out[s])
    with ThreadPoolExecutor(CORES) as pool:
        list(pool.map(_sin, range(CORES)))
        t = time.perf_counter()
        for _ in range(10):
            list(pool.map(_sin, range(CORES)))
        npN = (time.perf_counter() - t) / 10 * 1000

    print(f"   tensor {common} = {n:,} floats ({n * 4 / 1e6:.1f} MB)")
    print(f"   onnxruntime Sin, {CORES} threads .... {ort_ms:>7.2f} ms")
    print(f"   numpy sin, 1 thread ................ {np1:>7.2f} ms")
    print(f"   numpy sin, {CORES} threads ............. {npN:>7.2f} ms")
    print(f"   -> ORT is {ort_ms / npN:.1f}x slower than threaded numpy, "
          f"{ort_ms / np1:.1f}x slower than SINGLE-threaded numpy")
    reachable = sum(k["dur"] for k in sin_events) / 1000
    print(f"   Sin total this run: {reachable:.1f} ms; at numpy's rate it "
          f"would be {reachable / (ort_ms / npN):.1f} ms")
    print(f"   that is {reachable - reachable / (ort_ms / npN):.0f} ms of pure "
          "kernel-quality loss, on hardware you already own.")


# ==================================================== 4. thread scaling
print("\n\n4. HOW THE WHOLE INFERENCE SCALES WITH THREADS\n")
print(f"   {'threads':>8}{'ms':>10}{'speedup':>10}{'efficiency':>13}")
base = None
for threads in (1, 2, 4, 8):
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = rt.InferenceSession(MODEL, options,
                                  providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, VOICES)
    kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
    best = float("inf")
    for _ in range(REPEATS):
        t = time.perf_counter()
        kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
        best = min(best, (time.perf_counter() - t) * 1000)
    base = base or best
    print(f"   {threads:>8}{best:>10.0f}{base / best:>9.2f}x"
          f"{base / best / threads * 100:>12.0f}%")
    del kokoro

print("\n   near-linear -> compute-bound, buy cores.")
print("   flattens early -> bandwidth-bound, buy memory channels.")
print("   poor from the start AND far off both roofs -> runtime-bound; "
      "the kernel is the problem, not the machine.")
