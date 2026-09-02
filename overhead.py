"""M1, block 3 — does serving the model cost anything beyond running it?

client.py reports ~1690 ms to generate the M0 sentence. threads.py, tuned
identically, measured ~1540 ms for the same text on the same machine. 10% is
too big to wave through and too big to quote in a README unexplained.

Three candidates, and they are separable:

  A  direct            the tuned baseline, nothing around it
  B  in asyncio.to_thread   the model call moved onto a worker thread, but
                            no server, no event loop doing anything else
  C  through the server the whole path, measured by the client

A -> B isolates the thread hop. B -> C isolates uvicorn, the WebSocket, and
serialisation. If the loss is in A -> B it is mine and it is fixable; if it is
in B -> C it is transport and it is the price of being a server.

The suspicion worth testing: onnxruntime pins its intra-op thread pool to the
thread that created the session. Running inference from a *different* thread
than the one that built the session can cost the affinity, which would show up
here and nowhere in the benchmarks.
"""

import asyncio
import time

import onnxruntime as rt
from kokoro_onnx import Kokoro

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"
VOICE = "af_sarah"
TEXT = "Headstart streams the first audio chunk before the rest of the clip is generated."
REPEATS = 5


def build() -> Kokoro:
    o = rt.SessionOptions()
    o.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = 8
    o.inter_op_num_threads = 1
    s = rt.InferenceSession(MODEL, o, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(s, VOICES)


def bench(fn) -> tuple[float, float]:
    fn()  # warm
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return min(times), sum(times) / len(times)


async def main() -> None:
    k = build()
    call = lambda: k.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")

    a_best, a_avg = bench(call)
    print(f"  A  direct ................... {a_best:>7.0f} ms best  {a_avg:>7.0f} ms avg")

    async def threaded():
        return await asyncio.to_thread(call)

    await threaded()  # warm
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        await threaded()
        times.append((time.perf_counter() - t) * 1000)
    b_best, b_avg = min(times), sum(times) / len(times)
    print(f"  B  via asyncio.to_thread .... {b_best:>7.0f} ms best  {b_avg:>7.0f} ms avg")

    print(f"\n  thread hop costs {b_best - a_best:+.0f} ms best, {b_avg - a_avg:+.0f} ms avg"
          f"  ({b_best / a_best:.2f}x)")
    print("\n  Compare against client.py's gen_ms for the same sentence to get the")
    print("  server's share. Anything left over is uvicorn + WebSocket + framing.")


asyncio.run(main())
