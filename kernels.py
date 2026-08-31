"""M1, block 2e — the three operators the roofline could not model.

roofline.py labelled STFT, ConvTranspose and LSTM "NEITHER" — far from the
compute roof and far from the bandwidth roof. But that verdict came from an
analytic FLOP model that returns zero for all three, because it does not know
how to count an FFT, a transposed convolution, or a recurrent cell. "Zero
FLOPs" and "achieves nothing" are indistinguishable to that model.

Those three are ~110 ms of a ~690 ms run, and they sit inside the term of the
scaling formula that claims faster hardware will not help. If the verdict is
wrong, the formula is wrong, and the cloud instance gets chosen on a bad
number. So they get measured, not modelled.

METHOD — the one that already worked
    For Sin, the argument that settled it was not a roofline. It was running
    numpy on the identical buffer and finding that one numpy thread beat
    onnxruntime's eight. That is unarguable: same machine, same data, same
    arithmetic, one is slower.

    So: take each operator's real tensor shapes and real duration straight
    out of the profile of the real run, reimplement exactly that computation
    in numpy, and time it. No ONNX graph surgery — the trace already says
    what onnxruntime took, and rebuilding single-op models introduces its own
    artefacts.

WHAT EACH RESULT WOULD MEAN
    numpy much faster  -> kernel-quality loss. Recoverable here and now, and
                          it does NOT scale away with better hardware.
    comparable         -> the work is genuinely that expensive; it belongs in
                          the hardware-bound term after all.
"""

import collections
import itertools
import json
import math
import os
import time

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
TEXT = "Headstart streams."
CORES = 8
TARGETS = ("STFT", "ConvTranspose", "LSTM", "Sin")


def _shape(item):
    for _, s in item.items():
        return [d if isinstance(d, int) else 1 for d in s]
    return []


def trace():
    o = rt.SessionOptions()
    o.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = CORES
    o.inter_op_num_threads = 1
    o.enable_profiling = True
    s = rt.InferenceSession(MODEL, o, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(s, VOICES)
    k.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")  # warm
    k.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
    p = s.end_profiling()
    with open(p) as h:
        ev = json.load(h)
    os.remove(p)
    runs = sorted((e for e in ev if e.get("cat") == "Session"
                   and e["name"] == "model_run"), key=lambda e: e["ts"])
    t0 = runs[-1]["ts"]
    return [e for e in ev if e.get("cat") == "Node"
            and e["name"].endswith("kernel_time") and e["ts"] >= t0]


def timeit(fn, repeats=10):
    fn()  # warm
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t) * 1000)
    return best


kernels = trace()
by_op = collections.defaultdict(list)
for k in kernels:
    by_op[k["args"].get("op_name")].append(k)

total_ms = sum(k["dur"] for k in kernels) / 1000
print(f"real run: {len(kernels)} kernels, {total_ms:.0f} ms total\n")

recoverable = 0.0
rows = []

# ------------------------------------------------------------------ STFT
if by_op["STFT"]:
    ev = max(by_op["STFT"], key=lambda e: e["dur"])
    ins = [_shape(i) for i in ev["args"]["input_type_shape"]]
    out = _shape(ev["args"]["output_type_shape"][0])
    sig_len = ins[0][-1]
    frame_len = ins[2][0] if len(ins) > 2 and ins[2] else (out[2] - 1) * 2
    frames, bins = out[1], out[2]
    hop = max(1, round((sig_len - frame_len) / max(1, frames - 1)))
    ort_ms = sum(e["dur"] for e in by_op["STFT"]) / 1000

    signal = np.random.rand(sig_len).astype(np.float32)
    window = np.hanning(frame_len).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(signal, frame_len)[::hop]
    view = np.ascontiguousarray(view[:frames])

    def np_stft():
        return np.fft.rfft(view * window, axis=1)

    np_ms = timeit(np_stft)
    mflop = frames * 5 * frame_len * math.log2(frame_len) / 1e6
    print(f"STFT   signal {sig_len}, frame {frame_len}, hop {hop} -> "
          f"{frames} frames x {bins} bins")
    print(f"       that is {mflop:.1f} MFLOP of transform in total\n"
          f"       onnxruntime .... {ort_ms:>8.2f} ms   ({mflop / ort_ms:.2f} GFLOP/s)\n"
          f"       numpy rfft ..... {np_ms:>8.2f} ms   ({mflop / np_ms:.2f} GFLOP/s)\n"
          f"       -> onnxruntime is {ort_ms / np_ms:.0f}x slower\n")
    rows.append(("STFT", ort_ms, np_ms))
    recoverable += ort_ms - np_ms

# --------------------------------------------------------- ConvTranspose
if by_op["ConvTranspose"]:
    ort_ms = sum(e["dur"] for e in by_op["ConvTranspose"]) / 1000
    np_total = 0.0
    detail = []
    unmodelled = []
    for shape_key, group in itertools.groupby(
            sorted(by_op["ConvTranspose"],
                   key=lambda e: str(e["args"]["input_type_shape"])),
            key=lambda e: str(e["args"]["input_type_shape"])):
        group = list(group)
        ins = [_shape(i) for i in group[0]["args"]["input_type_shape"]]
        if len(ins) < 2 or len(ins[1]) != 3:
            # weights folded into an initializer and not reported in the
            # trace; no shape to reimplement against, so leave it out of the
            # comparison rather than guess a kernel size
            unmodelled.append((str(ins), len(group),
                               sum(e["dur"] for e in group) / 1000))
            continue
        x_shape, w_shape = ins[0], ins[1]
        c_in, l_in = x_shape[1], x_shape[2]
        c_out_g, k = w_shape[1], w_shape[2]
        o_shape = _shape(group[0]["args"]["output_type_shape"][0])
        stride = max(1, round(o_shape[2] / l_in))
        x = np.random.rand(c_in, l_in).astype(np.float32)
        w = np.random.rand(c_in, k).astype(np.float32)
        out_len = stride * l_in + k

        def np_convt(x=x, w=w, k=k, stride=stride, out_len=out_len, l_in=l_in):
            o = np.zeros((x.shape[0], out_len), dtype=np.float32)
            for j in range(k):
                o[:, j:j + stride * l_in:stride] += x * w[:, j:j + 1]
            return o

        ms = timeit(np_convt) * len(group)
        np_total += ms
        detail.append((x_shape, w_shape, len(group),
                       sum(e["dur"] for e in group) / 1000, ms))

    print("ConvTranspose  (all depthwise: weight middle dim is 1)")
    for x_shape, w_shape, n, o_ms, n_ms in detail:
        print(f"       in {x_shape} w {w_shape} x{n}: "
              f"ORT {o_ms:>7.2f} ms   numpy {n_ms:>7.2f} ms")
    for shp, n, o_ms in unmodelled:
        print(f"       x{n}: ORT {o_ms:>7.2f} ms   numpy    —    "
              f"(weights folded, shape not in trace)")
    modelled_ort = sum(d[3] for d in detail)
    print(f"       comparable ..... ORT {modelled_ort:>7.2f} ms   "
          f"numpy {np_total:>7.2f} ms")
    if np_total:
        print(f"       -> onnxruntime is {modelled_ort / np_total:.0f}x slower\n")
    rows.append(("ConvTranspose", modelled_ort, np_total or None))
    recoverable += max(0.0, modelled_ort - np_total)

# ------------------------------------------------------------------ LSTM
if by_op["LSTM"]:
    ev = by_op["LSTM"][0]
    ins = [_shape(i) for i in ev["args"]["input_type_shape"]]
    out = _shape(ev["args"]["output_type_shape"][0])
    seq, _, in_size = ins[0]
    dirs, hidden = out[1], out[3]
    ort_ms = sum(e["dur"] for e in by_op["LSTM"]) / 1000
    calls = len(by_op["LSTM"])
    # 4 gates, input and recurrent projections, per timestep per direction
    flop = 2 * seq * dirs * 4 * hidden * (in_size + hidden) * calls
    print(f"LSTM   seq {seq}, input {in_size}, hidden {hidden}, "
          f"{dirs} directions, {calls} calls")
    print(f"       {flop / 1e6:.0f} MFLOP in {ort_ms:.2f} ms  -> "
          f"{flop / (ort_ms / 1000) / 1e9:.0f} GFLOP/s")
    print("       batch 1 and a serial dependency across timesteps means this")
    print("       CANNOT saturate; tens of GFLOP/s is a reasonable result.\n")
    rows.append(("LSTM", ort_ms, None))

# ------------------------------------------------------------------- Sin
if by_op["Sin"]:
    ort_ms = sum(e["dur"] for e in by_op["Sin"]) / 1000
    n = sum(int(e["args"]["activation_size"]) for e in by_op["Sin"]) // 4
    buf = np.random.rand(n).astype(np.float32)
    outb = np.empty_like(buf)
    np_ms = timeit(lambda: np.sin(buf, out=outb))
    print(f"Sin    {n:,} elements across {len(by_op['Sin'])} calls")
    print(f"       onnxruntime .... {ort_ms:>8.2f} ms")
    print(f"       numpy 1 thread . {np_ms:>8.2f} ms")
    print(f"       -> onnxruntime is {ort_ms / np_ms:.1f}x slower\n")
    rows.append(("Sin", ort_ms, np_ms))
    recoverable += ort_ms - np_ms

# ---------------------------------------------------------------- verdict
print("=" * 62)
print(f"{'operator':<16}{'ORT ms':>10}{'numpy ms':>11}{'recoverable':>14}")
for name, o_ms, n_ms in rows:
    if n_ms is None:
        print(f"{name:<16}{o_ms:>10.1f}{'—':>11}{'not a target':>14}")
    else:
        print(f"{name:<16}{o_ms:>10.1f}{n_ms:>11.1f}{o_ms - n_ms:>13.1f}")
print("=" * 62)
print(f"\n  recoverable kernel-quality loss: {recoverable:.0f} ms of "
      f"{total_ms:.0f} ms  ({recoverable / total_ms * 100:.0f}%)")
print("  This is time a better kernel removes on the hardware you own.")
print("  It does NOT shrink when you buy a faster machine, which is exactly")
print("  why it belongs in its own term of the scaling formula.")
