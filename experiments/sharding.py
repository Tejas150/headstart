"""M1, block 3: one call at a time leaves the box half idle. How to fill it.

roofline.py (finding 5) found that the operators which dominate the floor,
Sin, ConvTranspose and STFT, sit under neither the bandwidth roof nor the
compute roof. A third of the model's time is the machine waiting. Threading
one inference harder does not recover it: threads.py found the curve flat
past 8 and negative past that, which is what "not compute-bound" looks like.

The other way to use idle time is to put a second piece of work in it. Two
shapes of that, and they are not the same shape:

  SHARDING (arm 1)
    Several processes, each with its own session and its own slice of the
    cores. Isolation is perfect; the cost is a full copy of the 326 MB model
    per process. Core assignment matters more than it looks: on this box
    logical CPUs pair as SMT siblings (0,1) (2,3) ..., so a mask of `0-3` is
    two physical cores, not four, and half the machine sits out. The masks
    below step by 2 for that reason.

  CONCURRENCY (arm 2)
    One session, several threads calling run() on it at once. ONNX Runtime
    allows this. No extra memory at all. The cost is that the calls contend
    for the same intra-op thread pool, so each individual call gets slower.

The number that decides between them is not per-call latency, it is audio
produced per second of wall time with everything running. Both arms report
that, plus the per-stream median so the latency cost is visible rather than
hidden inside an aggregate.

Why it matters for the server: whichever wins sets MODEL_WIDTH, and on a
rented CPU box memory is usually the binding constraint, not cores, which
is a thumb on the scale that a throughput number alone will not show you.

Run it on an idle machine. Both arms saturate every core by design, so
anything else running shows up as a loss attributed to the wrong cause.
"""

import _root  # noqa: F401  -- chdir to repo root; see _root.py

import json
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
SR = 24000

TEXT = ("A listener cannot tell the difference between a server that is under "
        "load and a product that is simply broken.")

WINDOW_S = 25.0          # per arm leg; long enough that one slow call cannot
                         # move the aggregate, short enough to finish a sweep


def build(threads: int, affinities: str | None = None) -> rt.InferenceSession:
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if affinities:
        options.add_session_config_entry("session.intra_op_thread_affinities",
                                         affinities)
    return rt.InferenceSession(MODEL, options,
                               providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- worker mode
#
# Arm 1 needs separate processes, and a separate process needs an entry point.
# Re-executing this file with `--worker` keeps it to one file instead of two,
# at the cost of a branch at the top that most readers can skip.

def worker(threads: int, window_s: float, mask: str) -> None:
    if mask != "none":
        os.sched_setaffinity(0, {int(c) for c in mask.split(",")})

    kokoro = Kokoro.from_session(build(threads), VOICES)
    kokoro.create("Warm.", voice=VOICE, speed=1.0, lang="en-us")

    print("READY", flush=True)
    sys.stdin.readline()                     # all workers start together

    audio_s, calls = 0.0, []
    started = time.perf_counter()
    while time.perf_counter() - started < window_s:
        a = time.perf_counter()
        samples, _ = kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
        calls.append(time.perf_counter() - a)
        audio_s += len(samples) / SR

    print("RESULT " + json.dumps({
        "audio_s": audio_s,
        "wall_s": time.perf_counter() - started,
        "calls": calls,
    }), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    worker(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    raise SystemExit(0)


# ------------------------------------------------------------- arm 1, sharding

def shard(label: str, threads: int, masks: list[str]) -> None:
    """One process per mask, all producing at once, for WINDOW_S."""
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "--worker", str(threads),
             str(WINDOW_S), mask],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        for mask in masks
    ]
    for p in procs:
        if p.stdout.readline().strip() != "READY":
            raise RuntimeError("a worker failed to load the model")
    for p in procs:
        p.stdin.write("go\n")
        p.stdin.flush()

    results = []
    for p in procs:
        results.append(json.loads(p.stdout.readline().split(" ", 1)[1]))
        p.wait()

    report(label, results, extra_model_copies=len(masks) - 1)


# --------------------------------------------------------- arm 2, concurrency

def payload(kokoro: Kokoro) -> dict:
    """The model's inputs, built once, so the arm times run() and nothing else.

    Phonemization is 0.3 ms (floor.py) so hoisting it does not change the
    answer, but it does mean every caller sends byte-identical work, which
    removes one way the arms could differ for an uninteresting reason.
    """
    phonemes = " ".join(kokoro.tokenizer.phonemize(TEXT, "en-us").split())
    tokens = kokoro.tokenizer.tokenize(phonemes)
    style = kokoro.get_voice_style(VOICE)[min(len(tokens), 510) - 1]
    return {
        kokoro._tokens_input: np.array([[0, *tokens, 0]], dtype=np.int64),
        "style": np.asarray(style, dtype=np.float32),
        "speed": np.array([1.0], dtype=np.float32),
    }


def caller(session, inputs, pin, out, index) -> None:
    if pin:
        # setaffinity(0) is this thread on Linux, not the process, which is
        # the whole reason a pinned in-process arm is possible at all.
        os.sched_setaffinity(0, set(pin))

    session.run(None, inputs)                # warm this thread's arenas

    audio_s, calls = 0.0, []
    started = time.perf_counter()
    while time.perf_counter() - started < WINDOW_S:
        a = time.perf_counter()
        outputs = session.run(None, inputs)
        calls.append(time.perf_counter() - a)
        audio_s += len(np.asarray(outputs[0]).ravel()) / SR

    out[index] = {"audio_s": audio_s,
                  "wall_s": time.perf_counter() - started,
                  "calls": calls}


def concurrent(label: str, sessions: list, pins: list,
               extra_model_copies: int) -> None:
    inputs = payload(Kokoro.from_session(sessions[0], VOICES))
    out = [None] * len(sessions)
    threads = [
        threading.Thread(target=caller,
                         args=(sessions[i], inputs, pins[i], out, i))
        for i in range(len(sessions))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report(label, out, extra_model_copies)


# ----------------------------------------------------------------- reporting

def report(label, results, extra_model_copies) -> None:
    aggregate = sum(r["audio_s"] for r in results) / max(r["wall_s"]
                                                         for r in results)
    per_stream = [statistics.median(r["calls"]) for r in results]
    memory = f"+{extra_model_copies * 326} MB" if extra_model_copies else "-"
    print(f"  {label:<44} {aggregate:>5.2f}x  {memory:>8}   "
          f"per stream {', '.join(f'{1000 * p:.0f}' for p in per_stream)} ms")


ARM = sys.argv[1] if len(sys.argv) > 1 else "all"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print(f"{TEXT[:40]}... x {WINDOW_S:.0f} s per leg, {VOICE}")
if REPEATS > 1:
    print(f"{REPEATS} repeats. Read the spread, not the best row")
print()
print(f"  {'layout':<44} {'output':>5}  {'memory':>8}   per-stream latency")
print(f"  {'-' * 44} {'-' * 5}  {'-' * 8}   {'-' * 24}")

for _ in range(REPEATS):
    if ARM in ("all", "shard"):
        # Baseline: one call at a time on a session that owns every core.
        shard("1 process x 8 threads (baseline)", 8, ["none"])

        # Masks step by 2 because siblings pair adjacently; `0-3` would be two
        # physical cores wearing four cores' clothing.
        shard("2 processes x 4 threads, pinned", 4, ["0,2,4,6", "8,10,12,14"])
        shard("4 processes x 2 threads, pinned", 2,
              ["0,2", "4,6", "8,10", "12,14"])
        shard("8 processes x 1 thread,  pinned", 1,
              [str(c) for c in range(0, 16, 2)])
        print()

    if ARM in ("all", "concurrent"):
        # Same session object shared by every caller. Its 1-caller leg is the
        # baseline for this arm, not the sharded one above: this path skips
        # phonemization and calls run() directly, so the two arms' absolute
        # numbers are not comparable even though their ratios are.
        wide = build(8)
        concurrent("1 session x 8 threads, 1 caller", [wide], [None], 0)
        concurrent("1 session x 8 threads, 2 callers", [wide] * 2,
                   [None] * 2, 0)
        concurrent("1 session x 8 threads, 4 callers", [wide] * 4,
                   [None] * 4, 0)
        del wide

        # The middle ground: two sessions in one process, each pinned to half
        # the box. Cheaper than four processes, still a second copy of weights.
        left, right = build(4, "3;5;7"), build(4, "11;13;15")
        concurrent("2 sessions x 4 threads, pinned, in-process",
                   [left, right], [[1], [9]], 1)
        del left, right
        print()

print("  output    = audio seconds produced per wall second, all streams")
print("  memory    = weights held beyond the first copy")
print("  per stream= median time for one call, per stream, in order")
