# headstart

A streaming TTS inference server. The listener starts hearing audio while the rest of the clip is still being generated, which is where the name comes from.

Built to answer one question with real numbers: *how fast can you serve a neural TTS model on a laptop CPU, and where exactly does the latency live?*

Everything runs on a Ryzen 7 4800H (8C/16T, DDR4-3200), with no GPU anywhere in it. Constrained hardware is the more interesting version of the problem, and a good share of what follows is about the things that could not be made faster, and why.

---

## The number

Three-sentence paragraph, 17.15 s of audio, measured end-to-end from the client:

| | time to first audio | spread | total time |
|---|---|---|---|
| one `create()` call, nothing emitted until done | 5658 ms | 5636–5682 | 5658 ms |
| streamed, chunks sized against the listener's buffer | 1940 ms | 1894–2187 | 7163 ms |

That is 2.9x sooner to first sound for the same 17.15 s of audio, and the buffer never runs dry: audio in hand stays ahead of audio played at every chunk, so there is no gap. The clip takes 1.27x longer to finish. That is the trade, and you pay it because every chunk re-pays a fixed per-call cost (finding 2).

> Median of 3 with the full spread shown, both rows measured in one run by `experiments/headline.py`. Best-of-N would measure the machine on its luckiest day, which is not the day the listener gets. A median with no spread would hide the tail, and the tail is what sets the floor under the chunk size in finding 6.
>
> Both rows come from one script and one machine state on purpose. Rows gathered under different conditions give you two unrelated numbers in a column rather than a speedup. `experiments/headline.py` also asserts that both rows describe the same length of audio, so the comparison cannot quietly become "a shorter clip" instead of "a faster one."

How much this buys depends on how the text opens. This paragraph's first sentence is 14 words, and the chunker never splits mid-sentence, so chunk 0 is that whole sentence and 1940 ms is what a 14-word forward pass costs. Give the same server text with a 39-word opening sentence and it goes from 4088 ms to 1393 ms, the same 2.9x, because now there is something to cut. What moves the number is the length of the first thing the listener is waiting on, not the length of the clip.

Transport barely registers. Connect, WebSocket and framing together are 33 ms of a 6.4 s run, which is why the transport was built last. The ordering note under Status explains that choice.

---

## The second number

Speed is one question. The other is how many people one machine can serve before the audio starts breaking up. On this laptop it is 2, and where that 2 comes from matters more than the value. The server watches how much faster than realtime it delivers audio, meaning seconds produced per second of wall clock while it is working, which reads between 2.9x and 3.3x here. It admits that many streams, rounded down. One listener consumes one second of speech per second of real time, so a box delivering 2.9x carries two of them. Nobody typed it in.

It watches delivered audio rather than per-call speed because those are different numbers once the model runs several calls at once, and they move in opposite directions (finding 18).

The reading is taken from inside its own limit, and that caps it. Capacity comes from what the server delivered, and what it delivered is what two admitted streams could keep it busy with. Finding 18 forces four streams through the same build with the door off and gets 5.12x, so the ceiling is higher than the door will ever let this measurement see. Nothing here is wrong. 2 is a safe number and it was earned rather than guessed. But a loop that only ever samples the state it chose cannot climb out of a conservative one on its own. Raising it takes a deliberate probe: admit one more than the reading says, and keep it only if the buffers hold. That is written down as open, not built.

Past that line a server that accepts everyone lets everyone's buffer run dry, so this one stops accepting everyone. Three callers arriving together, same server, capacity moved between arms:

| streams admitted | refused | underruns | silence, per run |
|---|---|---|---|
| 3 | 0 | 5, 6, 9, 10, 11 | 2.7 s, 6.4 s, 7.3 s, 20.0 s, 21.9 s |
| 2 | 1 | 0, 0, 1, 1, 2, 2, 2 | 0 ms, 0 ms, 0.1 s, 0.8 s, 0.8 s, 1.7 s, 2.3 s |

Seven runs at the measured capacity, five at one above it, across a quiet afternoon and a busy one. The damage from the third stream does not land on the third stream. Every run in the top row broke up all three, including the mildest of them. The caller who is turned away hears about it before a single sample is sent, and gets a retry hint, instead of finding out mid-sentence.

At eight callers the same argument gets louder. Eight clients, six requests each:

| 8 clients at once | requests that stalled | p95 to first audio | turned away |
|---|---|---|---|
| accept everyone | 48 / 48 | 17828 ms | 0 |
| refuse past capacity | 0 / 16 | 7457 ms | 32 / 48 |

Every stall gone, and the people who do get served wait less than half as long. Nothing was optimised to get there. The queue simply stopped being allowed to grow. Standing in an unbounded queue turned out to be worse than being told "not now" and coming back.

The order of that queue matters as much as its length, in two places. At admission control, plain first-come-first-served starves 2 of 8 clients completely, because a refused caller backs off and loses its place to whoever was just served (finding 15). At the model, the slot should go to whichever stream has the least audio left to play (finding 16).

---

## What changed, and what each change bought

Six steps, each with the script that measured it. The rows do not chain into one number, because they were measured on different texts and different harnesses. Reading them as a single figure falling from 5658 ms to 876 ms would be wrong.

| what changed | what it bought | measured on |
|---|---|---|
| baseline: one call, nothing emitted until done | 5658 ms to first audio | `headline.py`, 17.15 s paragraph |
| `intra_op=8` instead of the 16-thread default | 1914 ms to 1541 ms | `isolate.py`, one sentence |
| chunk on sentences, send as each one finishes | 5658 ms to 1940 ms | `headline.py`, same paragraph |
| cut chunk 0 at a clause, around five words | 1624 ms to 876 ms | `leadsweep.py`, long opening sentence |
| refuse past the measured capacity | 48 of 48 stalls to 0 | `bench.py`, 8 clients |
| run four model calls at once instead of one | 3.44x to 5.12x delivered | `bench.py`, 4 clients, door off |

The full account, including the things that were tried and did not work and the things that have not been done, is in [`docs/optimisations.md`](docs/optimisations.md). The eighteen measurements behind all of it are in [`docs/findings.md`](docs/findings.md).

---

## Architecture

```
  client ──WS──▶  FastAPI /tts
                    │  plan the chunks, against the buffer and the queue
                    │  (sizes are read from the live fit, not from constants)
                    │
                    │  ┌── admission control: as many streams as the measured speed allows
                    │  │   full? wait, then refuse with a retry hint
                    │  │   order: most-refused first, then longest-waiting
                    │  └── refused ──▶ {"type":"busy"}, no audio sent
                    │
                    │  ├─ the model slot: a measured number of calls at once
                    │  │  order: least audio in hand first
                    │  └─ asyncio.to_thread, keeps the event loop alive
                    │
                    │  + 277 ms silence at sentence boundaries  ← free audio, deepens the buffer
                    │  per chunk: JSON metadata frame, then raw int16 PCM frame
  client ◀──────────┘  playback starts on chunk 0, while chunk 1 is still generating
```

Two gates, doing different jobs. Admission control decides *whether* you get served, and holds its slot for the whole request. Releasing it between chunks would let a new stream in to compete with one already mid-sentence, which is the thing it exists to prevent. The model slot decides *when* each individual chunk runs, and how many run together. Admission control bounds the queue, the slot bounds how much of the machine is in use at once. Both are ordered by need rather than arrival, for the reasons in findings 15 and 16.

Nothing in that diagram is a tuned constant. Chunk size, how many streams fit, how long admission control waits, and the shortest chunk worth generating are all derived from one three-coefficient fit that the server refits from its own completed work. `/health` publishes the coefficients next to every number derived from them, so the arithmetic can be checked. Constants in the source are seeds for a cold start, and an operator who names a number keeps it.

Metadata goes as JSON and audio as a separate binary frame, rather than base64 inside the JSON, which would add 33% to the one payload where bytes are latency. The audio is int16 rather than float32, which halves it again.

---

## Run it

```bash
docker build -t headstart .
docker run --rm -p 8000:8000 headstart
curl localhost:8000/health
```

That is the whole setup. The image carries the 311 MB of weights, so there is nothing to download and no Python to install. That matters, because a reader who cannot run it cannot check any of the numbers. The cost is a 1.14 GB image. The alternative was a small image plus a setup step, and the setup step is precisely what I wanted gone.

If you would rather not bake the weights in, use `--build-arg FETCH_MODELS=0` and mount your own at `/app/models`.

One flag matters if you constrain CPU. The server defaults to 8 ONNX threads because that is the physical core count it was measured on, and `os.cpu_count()` inside a container reports the *host* rather than your quota, so it cannot autodetect this correctly. Tell it:

```bash
docker run --cpus 4 -e HEADSTART_INTRA_OP=4 -p 8000:8000 headstart
```

<details>
<summary>Or run it directly, which is what the benchmarks were measured on</summary>

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# if python3-venv isn't installed and you don't have sudo:
#   pip3 install --user virtualenv && python3 -m virtualenv .venv

mkdir -p models
B=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
curl -L -o models/kokoro-v1.0.onnx "$B/kokoro-v1.0.onnx"
curl -L -o models/voices-v1.0.bin  "$B/voices-v1.0.bin"

.venv/bin/python server.py &
.venv/bin/python client.py --paragraph --compare
```

`--compare` runs the same text twice, with and without the clause split, and plays both. Playback pipes raw PCM into `aplay`, so there is no audio library to install and nothing to configure. Add `--no-play` to measure only.

To reproduce the load findings, the sweep with admission control on and then off:

```bash
.venv/bin/python bench.py --levels 1,2,3,4,8 --requests 6   # admission control on
.venv/bin/python server.py --max-inflight 0 &               # admission control off, then re-run bench.py
```

Capacity can also be moved on a running server, which is how the three-caller table above was measured without restarting between arms:

```bash
curl -X POST "localhost:8000/admission?capacity=3"    # pin it; 0 turns admission control off
curl -X POST "localhost:8000/admission?capacity=-1"   # hand it back to the measurement
```

Naming a number pins it, and `-1` hands it back to the measurement, so a reader who has just pinned capacity open to watch it fail can return to the default without a restart. Omit `--max-inflight` and capacity tracks the measured delivered speed from the start.

Model weights are gitignored (311 MB). Pull them with the commands above.

</details>

To talk to it yourself, the WebSocket protocol is written up in [`docs/api.md`](docs/api.md): the request fields, the `start` / `chunk` / `end` / `busy` frames, an annotated trace from a real run, and what to do when you get refused.

Benchmark on a quiet machine. `bench.py` re-runs its lowest level at the end as a control and prints the drift. Anything over 15% and it declares the run void rather than publishing. It means it, and a browser eating a core is enough to fail a run. That is worth the inconvenience, because a benchmark contaminated that way produces numbers that are wrong and entirely plausible-looking.

---

## Status

| # | Milestone | State |
|---|---|---|
| M0 | Scaffold + model speaks | done |
| M1 | Streaming: chunking policy, then latency floor, then WebSocket | done |
| M2 | Scheduler + benchmark harness (p50/p95/p99) | done: harness, admission control, queue policy, contention-aware planning |
| M3 | Docker + kind + HPA + Prometheus/Grafana | Docker done |
| M4 | Go gateway + Piper backend comparison | |

Still to come: result cache, Prometheus `/metrics`, a `TTSBackend` interface with Piper behind it, kind + HPA, Go gateway. The open measurement is a runtime bake-off, ONNX Runtime against PyTorch on chunk-0-sized inputs, where the published real-time factors differ enough (0.72 against 0.49) to move the term that dominates TTFB.

M1 was deliberately ordered chunking, then latency, then transport. The obvious order is to build the WebSocket first, but transport moves TTFB by single-digit milliseconds on localhost, measured at 33 ms of 6.4 s. Building it first would have meant re-running every benchmark after the real optimisation landed.

M2 is a scheduler rather than a batcher because of finding 7. The batcher half is impossible without graph surgery. The harness ran before the scheduler on purpose: a scheduler is worth building only if the queue owns the tail, and finding 10 shows it owns 88% of it, against a hard capacity line no queue policy can move. That bounded what the scheduler could honestly claim before a line of it was written, and findings 11 through 16 claim exactly that and no more: every stall removed, nobody starved, chunks planned against real contention, and not one extra second of audio per second.

The ceiling itself moved later, and not by scheduling. Finding 18 takes delivered audio from 3.44x to 5.12x at four clients by running several model calls at once, which is a claim about how much of the machine is in use rather than about who goes next. The scheduler bound stands: no ordering policy moves that line, and finding 7 still rules out the batcher that would move it further.

---

## The honest bar

Cartesia's Sonic-3 advertises 40 to 90 ms time-to-first-audio, and independent measurements land nearer 166 to 190 ms. That is served GPU infrastructure, and this is one laptop CPU.

The claim here is not that I matched it. It is: here is the floor on this machine, here is what it decomposes into operator by operator, here is what each lever bought, and here is the model that predicts what different hardware would do. Being several times off a frontier system with a full account of where the time goes is more useful than a fast number with no decomposition.

And per finding 9, most of the remaining gap is not a serving gap at all. A one-shot graph's floor is a full forward pass, an incremental model's is one step of a loop. Knowing which of those you are holding decides whether the next win comes from engineering or from a different model.

---

## Repo map

The root holds the product, `experiments/` holds the evidence. The split is deliberate: the files at the top are what you would run or deploy, and the fifteen below them are one-shot measurements whose output is a number in [`docs/findings.md`](docs/findings.md).

**The server**

| file | what it is |
|---|---|
| `server.py` | The server, admission control, the model slot and the cost model. Also findings 6, 8, 11, 12, 13, 15, 16, 17, 18 |
| `client.py` | Reference client: streams, plays, and reports what it experienced |
| `bench.py` | Findings 10, 14, 15, 18. Percentiles and saturation, admission control on/off, the per-client fairness check, the model-width matched pair |
| `demo/` | The console at `/demo`, showing audio received racing audio played, with a runtime capacity toggle |

**The evidence**, in `experiments/`

| file | what it establishes |
|---|---|
| `speak.py` | M0 baseline: cold start 909 ms, synthesis 1785 ms, RTF 0.41 |
| `stream.py` | Finding 1: `create_stream()` against `create()` against our chunking |
| `floor.py` | Finding 2: the fixed/variable fit |
| `threads.py` | Finding 3: the thread sweep |
| `isolate.py` | Finding 3: attributing it to `intra_op` alone |
| `profile_ops.py` | Finding 2: per-operator fixed/scaling decomposition |
| `quant.py` | Finding 4: int8, including the audio-quality check |
| `roofline.py` | Finding 5: machine ceilings and per-operator placement |
| `kernels.py` | Finding 5: numpy head-to-head proving the kernel loss |
| `overhead.py` | Finding 8: cost of the thread hop |
| `leadsweep.py` | Finding 6: the first-chunk sweep that set the floor at 4 words |
| `headline.py` | The table at the top: both rows, one run, one methodology |
| `granularity.py` | Finding 9: chunk-size sweep, citation-form cost, and the graph signature |
| `voice_ttfb.py` | Finding 17: seven voices, one stream at a time, speaking rate against cost per second |
| `sharding.py` | Finding 18: process sharding against in-process concurrency, output per second against memory |

Run them from anywhere in the checkout. `experiments/_root.py` anchors the working directory to the repo root, so the relative paths inside each script stay correct and `import server` still resolves. The server takes the other route and reads `HEADSTART_MODEL` from the environment, because unlike these it does have to run somewhere else.

Every number in this repo came from one of these, or from `/health` on a running server, on the machine described at the top. Re-running them is the point.

---

## Stack

Python 3.10 · ONNX Runtime (CPU) · kokoro-onnx · FastAPI + WebSockets · numpy
Planned: Prometheus + Grafana · kind · Go gateway
