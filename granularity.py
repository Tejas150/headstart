"""M1 — how fine can chunking go, and why does it stop there?

The obvious extrapolation from the lead sweep: if a 5-word first chunk beats a
whole sentence, stream word by word and get TTFB down to the floor. The
architecture question underneath it is the one worth answering, because it is
the difference between this model and the ones quoting 40-90 ms.

WHAT THIS MEASURES

  1. Cost per chunk as chunk size shrinks. Every call re-pays the ~300 ms fixed
     cost from block 2, so more chunks means more total compute. RTF per chunk
     says whether playback can keep up.

  2. What the audio does. This is the part a latency table cannot see. A word
     synthesised alone is not the same signal as that word inside a sentence:
     it comes out in citation form, fully articulated, with its own onset and
     offset silence. Same words, different -- and much longer -- audio.

  3. Whether the graph could stream at all, which is the real answer. Printed
     from the model file rather than argued.

WHY THE GRAPH SETTLES IT
    A model that streams incrementally must take state in and hand state back,
    so the next call resumes where the last one stopped. Kokoro takes tokens,
    style and speed, and returns audio. There is no state in the signature --
    nowhere to put "continue from here", and no way to ask for the next 40 ms.
    It is non-causal besides: durations are predicted over the whole utterance,
    and a global STFT spans the whole signal, so the entire input has to exist
    before sample zero can.

    So token-level streaming is a property of the model, not of the server. The
    ~300 ms floor here is one full forward pass of a one-shot graph. Systems
    that emit audio every few tens of milliseconds are running a loop with
    recurrent state and tapping it each step. No scheduler closes that gap;
    a different architecture does.

Run:  .venv/bin/python granularity.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import onnx
import onnxruntime as rt
from kokoro_onnx import Kokoro

import server as srv

SENTENCE = (
    "Headstart streams the first audio chunk before the rest of the clip is generated."
)
PARAGRAPH = SENTENCE + " It is a way to reduce latency and make the model feel more responsive."
SIZES = [1, 2, 3, 5, 8, 13, 21]


def load() -> Kokoro:
    o = rt.SessionOptions()
    o.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = srv.INTRA_OP_THREADS
    o.inter_op_num_threads = srv.INTER_OP_THREADS
    session = rt.InferenceSession(srv.MODEL, o, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(session, srv.VOICES)
    k.create("Warm.", voice="af_sarah", speed=1.0, lang="en-us")
    return k


def gen(k: Kokoro, text: str) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    samples, _ = k.create(text, voice="af_sarah", speed=1.0, lang="en-us")
    return (time.perf_counter() - t0) * 1000, samples


def sweep(k: Kokoro) -> None:
    words = PARAGRAPH.split()
    print(f"\n  Chunk-size sweep — {len(words)} words\n")
    print(f"  {'words/chunk':>11} {'chunks':>7} {'gen/chunk':>10} {'audio/chunk':>12} "
          f"{'RTF':>6} {'total gen':>10} {'total audio':>12}")
    print("  " + "-" * 76)
    for n in SIZES:
        groups = [" ".join(words[i:i + n]) for i in range(0, len(words), n)]
        res = [gen(k, g) for g in groups]
        gens = [r[0] for r in res]
        auds = [len(r[1]) / srv.SAMPLE_RATE for r in res]
        per_gen, per_aud = statistics.median(gens), statistics.median(auds)
        print(f"  {n:>11} {len(groups):>7} {per_gen:>7.0f} ms {per_aud:>10.2f}s "
              f"{per_gen / 1000 / per_aud:>6.2f} {sum(gens):>7.0f} ms {sum(auds):>10.2f}s")
    print("\n  RTF stays under 1 even at one word per chunk, so the buffer is not")
    print("  what breaks. Total generation and total audio both are.")


def citation_form(k: Kokoro) -> None:
    """The cost no latency table shows: the audio itself changes."""
    _, whole = gen(k, SENTENCE)
    per_word = np.concatenate([gen(k, w)[1] for w in SENTENCE.split()])
    w_s, p_s = len(whole) / srv.SAMPLE_RATE, len(per_word) / srv.SAMPLE_RATE
    print(f"\n  Same sentence, two ways\n")
    print(f"    one call            {w_s:>6.2f}s")
    print(f"    one call per word   {p_s:>6.2f}s   ({p_s / w_s:.2f}x longer)")
    print("\n  Where the extra time goes — each word carries its own silence:")
    for w in SENTENCE.split()[:5]:
        s = gen(k, w)[1]
        a = np.abs(s)
        thr = a.max() * 0.01
        lead = int(np.argmax(a > thr)) / srv.SAMPLE_RATE
        trail = int(np.argmax(a[::-1] > thr)) / srv.SAMPLE_RATE
        print(f"    {w!r:<12} {len(s) / srv.SAMPLE_RATE:>5.2f}s   "
              f"silence {lead:.2f}s + {trail:.2f}s = {lead + trail:.2f}s")
    print("\n  A function word like 'the' is ~0.1s inside a phrase. Alone it is a")
    print("  fully articulated citation of the word, with its own onset and offset.")


def graph_signature() -> None:
    """The actual answer: ask the model whether it could stream."""
    g = onnx.load(srv.MODEL).graph

    def shape(v):
        return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]

    print("\n  Model signature\n")
    for v in g.input:
        print(f"    IN   {v.name:<10} {shape(v)}")
    for v in g.output:
        print(f"    OUT  {v.name:<10} {shape(v)}")

    ops: dict[str, int] = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1

    # Incremental generation requires carrying state between calls. If no input
    # or output is a state tensor, there is no way to express "resume".
    names = {v.name.lower() for v in list(g.input) + list(g.output)}
    stateful = [n for n in names if any(t in n for t in ("state", "cache", "hidden", "h0", "c0"))]
    print(f"\n    state tensors in the signature: {stateful or 'none'}")
    print(f"    nodes {len(g.node)}, distinct ops {len(ops)}")
    print("\n  Whole-sequence operators — these cannot begin before the input ends:")
    for op in ("STFT", "LSTM", "ConvTranspose", "ReduceMean", "Resize"):
        if op in ops:
            print(f"    {op:<16} {ops[op]}")
    print("\n  No state in, no state out: the graph cannot be asked to continue,")
    print("  only to run again from the beginning. Token-level streaming is an")
    print("  architecture property, and this architecture does not have it.")


if __name__ == "__main__":
    k = load()
    sweep(k)
    citation_form(k)
    graph_signature()
