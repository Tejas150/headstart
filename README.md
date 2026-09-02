# headstart

A **streaming TTS inference server**. The listener starts hearing audio while the rest of the clip is still being generated — the name is the architecture.

Built to answer one question with real numbers: *how fast can you serve a neural TTS model on a laptop CPU, and where exactly does the latency live?*

**No GPU.** Everything runs on a Ryzen 7 4800H (8C/16T, DDR4-3200). Optimising inference on constrained hardware is the interesting problem; throwing a GPU at it isn't. The findings below are mostly about what *couldn't* be made faster, and why — which turned out to be the more useful half.

---

## The number

Three-sentence paragraph, 16.6 s of audio, measured end-to-end from the client:

| | time to first audio | how |
|---|---|---|
| baseline — no streaming | **7023 ms** | one `create()` call, nothing emitted until done |
| sentence chunking | **1757 ms** | our own splitter; the library's doesn't help (finding 1) |
| + clause-split first chunk | **917 ms** | cut the segment the listener is waiting on (finding 6) |

**7.7× sooner to first sound**, same 16.6 s of audio, and the buffer never runs dry — `lead` (audio in hand minus audio already played) stays positive at every chunk, so there is no gap.

> These are **single-shot end-to-end numbers, not best-of-N.** Run-to-run spread is roughly ±100 ms. A latency README that quotes best-of-5 is measuring the machine on its luckiest day; the listener doesn't get that day.

Transport is not where the time goes: **connect + WebSocket + framing is 33 ms of a 6.4 s run.** That is why the transport was built *last* — see the ordering note under Status.

---

## Run it

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

`--compare` runs the same text twice, with and without the clause split, and plays both. Playback pipes raw PCM into `aplay`, so there's no audio library to install and nothing to configure. Add `--no-play` to measure only.

Model weights are gitignored (311 MB) — pull them with the commands above.

---

## What I found

### 1. The library's streaming is not a latency feature

`kokoro-onnx` exposes `create_stream()`, which reads like the answer. It isn't. It splits only on `MAX_PHONEME_LENGTH` (510) — a guard against overrunning the model's input window. **Any text under that limit yields exactly one chunk, so `create_stream()` is `create()`.** Worse, its splitter deliberately *balances* batch sizes for prosody, which is the opposite of what time-to-first-byte wants.

So the chunking had to be ours. Reading the source instead of the README is the entire finding.

### 2. Chunking has a floor, and the floor is 346 ms

Every call pays a **fixed** cost (phonemization, tokenization, style lookup, session dispatch) and a **variable** cost (the forward pass, proportional to audio). Chunking shrinks the variable part only, so TTFB approaches the fixed cost and stops. Fitting a line through progressively longer prefixes:

```
synth_ms = 346 + 374 × audio_seconds        (n=10, max residual 66 ms)
```

Phonemization is **0.3 ms** of that 346. The floor lives inside the ONNX session, so no amount of smarter text handling reaches it.

Profiling per-operator gives an independent decomposition — **299 ms fixed, 327 ms/audio-second.** Both terms are 1.15× below the fit above, which is exactly the separately-measured thread win, since the fit was run untuned. Two unrelated methods agreeing is the reason I trust either.

**What the floor is made of:**

| | share of the 299 ms floor |
|---|---|
| `Sin` | 32% |
| `Conv` | 29% |
| `STFT` | 12% |
| `ConvTranspose` | 11% |

**This is a convolutional vocoder, not a transformer.** `MatMul` is 4.5 ms of 299 — under 2%. Every instinct trained on LLM serving points at the wrong operator here. `Sin` is top of the list because Kokoro's vocoder builds speech by summing sine waves at the predicted pitch, over ~44 million values per sentence.

Framework overhead — Python, the ONNX dispatcher, everything that isn't math — is **2%** (669 ms of kernel time inside a 686 ms wall). There is no free win in the plumbing.

### 3. Thread tuning: 1.27×, free — and it's one setting, not two

8 threads beats the default and beats 16. 8 is the physical core count; two SMT threads on one core share a load/store path, so the second one contends rather than helps.

The first version of this measurement was **confounded** — the tuned runs set `intra_op` *and* `inter_op`, the default set neither, so "1.34×" couldn't be attributed. Isolating them:

| config | short chunk | full sentence |
|---|---|---|
| default | 770 ms | 1970 ms |
| `intra_op=8` only | 740 ms | **1548 ms** |
| `inter_op=1` only | 733 ms | 1945 ms |
| both | 593 ms | 1552 ms |

**On the full sentence `intra_op` is the entire story; `inter_op` does nothing.** On a short chunk you need both — 30 ms and 37 ms separately, 178 ms together. They interact rather than add. So it ships as one setting with one number, not two independent wins.

### 4. int8 quantization is structurally unavailable here

Predicted ~1.27× from the matmul-family share. Then the quantized model wouldn't load:

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
```

**ONNX Runtime's `ConvInteger` CPU kernel is 2-D only, and all 94 convolutions in this graph are 1-D** (audio slides a filter over time; images slide over height *and* width). So the operator holding 86 of the 93 quantizable milliseconds cannot be quantized on this runtime at all. Quantizing `MatMul` alone: **1.00× speedup**, and phoneme durations shifted by 21–64 ms — no speed, different audio.

Also worth knowing before trying: this CPU is Zen 2, which has no VNNI, so int8 was only ever going to be a memory-traffic win rather than an instruction win.

The deliverable here is the diagnosis, not a number. "int8 gave 1.27×" survives no follow-up questions; "the runtime's integer convolution kernel is 2-D and this model is 1-D, which I found by reading weight ranks after the model refused to load" does.

### 5. Not hardware-bound. **Mixed-bound** — so hardware scaling is piecewise

The real question behind all of this: *have I hit the machine's limit, or my own?* They have opposite consequences — one means buy a bigger box, the other means the bigger box wastes money.

Measured both ceilings (**33.8 GB/s**, **391 GFLOP/s**, ridge point **11.6 FLOP/byte**) and placed every operator:

| op | ms | GFLOP/s | GB/s | bound by |
|---|---|---|---|---|
| `Conv` | 261.5 | **438.7** | 2.3 | **compute — saturated** |
| `Sin` | 157.7 | 0.3 | 2.3 | *neither* |
| `ConvTranspose` | 48.9 | 0.0 | 0.3 | *neither* |
| `STFT` | 48.6 | 0.0 | 0.0 | *neither* |
| `Add` | 43.6 | 2.7 | 28.4 | bandwidth |
| `Mul` | 33.0 | 5.3 | 47.8 | bandwidth |

`Conv` genuinely saturates. `Add`/`Mul` sit at the memory roof. But **three operators are far from *both* roofs — that is the signature of a slow kernel, not a busy machine.**

A roofline is a model, so I proved the accusation by reimplementing each one in numpy and running it on the same machine against the same shapes:

| | ONNX Runtime | numpy | |
|---|---|---|---|
| `STFT` | 50.9 ms | 1.2 ms | **43×** |
| `ConvTranspose` | 6.8 ms | 1.4 ms | 5× |
| `Sin` | 163.3 ms | 108.3 ms | 1.5× |
| `LSTM` | 12.3 ms | — | **not a target** — 39 GFLOP/s is fine for batch-1 |

**~110 ms of 686 ms — 16% — is recoverable kernel quality**, and that's a floor on the estimate: three `ConvTranspose` nodes had folded weights I couldn't reimplement, so they're excluded rather than guessed.

Which gives the scaling model, and the point is that **it is piecewise, because no single multiplier exists**:

```
TTFB(target) ≈ 261ms × (FLOPS_here / FLOPS_there)     ← Conv, saturating compute
             + 100ms × (BW_here    / BW_there)        ← elementwise, at the memory roof
             + 110ms × 1.0                            ← kernel loss: does NOT scale
             + 215ms × (clock_here / clock_there)     ← launch overhead, serial work
```

**Anyone quoting "2× the hardware → 2× the speed" is wrong, and this says precisely where.** 16% of the time doesn't move at all. Plugging in a modern server CPU predicts **~1.6×** — a short chunk near 370 ms. That's a falsifiable prediction, and validating it on a cloud instance is the next experiment.

> Two bugs in my own measurement, both caught by noticing an impossible number rather than by a crash. The bandwidth probe counted 5 memory passes where the standard formula counts 3, understating bandwidth by 1.67× — and *every* scaling claim divides by that number. And `Conv` reported 459 GFLOP/s against a 349 GFLOP/s "peak", which can't happen, so the baseline was weak rather than the reading wrong.

### 6. Where the first cut goes *is* the latency policy

Block 2 said the floor is ~300 ms, so a 1757 ms first chunk is nowhere near it — because a long first sentence is a long forward pass. Cutting the first segment at a clause boundary takes it to **917 ms** for the same total audio.

It isn't free. The two renderings correlate at only **+0.71** and diverge 40 ms in, so prosody changes across the whole clip, not just at the seam. It's a **flag, not a default** — a tradeoff to listen to, not a fact to assume.

### 7. This graph cannot batch

Checked before building a dynamic batcher, which is the sort of thing worth checking first:

```
INPUTS   tokens  int64  [1, 'sequence_length']
         style   float  [1, 256]
OUTPUTS  audio   float  ['audio_length']
```

**The batch dimension is a literal 1, not symbolic** — only sequence length is dynamic. And the output has no batch dimension at all, so even a batched input would produce one undifferentiated waveform. Real request batching needs graph surgery, not a scheduler.

Same shape as finding 4: structurally closed, for a specific reason. See Status for how M2 changes as a result.

### 8. Serving the model costs 2%, and one design decision costs everything

`kokoro.create()` is synchronous and pins a core for ~1.5 s. Calling it directly inside an async handler blocks the event loop and stalls *every* connected client — including ones merely waiting to receive bytes that already exist. It runs in a worker thread. That hop costs **+23 ms at best, +115 ms on average**; the event loop staying responsive is worth it.

One model call runs at a time, on purpose. The session is thread-safe, so concurrent calls are legal — they're just pointless, because `intra_op=8` already gives one operator all 8 physical cores. Two concurrent requests don't go faster, they interleave and the tail gets worse.

That choice creates a queue, so the queue is **measured**: every chunk reports `queue_ms` separately from `gen_ms`. Two clients at once:

```
client A   time to first audio  1712 ms
client B   time to first audio  3421 ms      ← waited one full generation
```

A latency number that doesn't separate queue time from model time isn't a latency number. This is the measured motivation for M2's scheduler, rather than an assertion that one is needed.

---

## Architecture

**What runs today:**

```
  client ──WS──▶  FastAPI /tts
                    │  split text into chunks (sentence, or clause for the first)
                    │  ├─ asyncio.Semaphore(1)  ← one model call at a time; queue is measured
                    │  └─ asyncio.to_thread     ← keeps the event loop alive
                    │
                    │  per chunk: JSON metadata frame, then raw int16 PCM frame
  client ◀──────────┘  playback starts on chunk 0, while chunk 1 is still generating
```

Metadata as JSON, audio as a separate binary frame — **not** base64 inside the JSON, which is +33% on the one payload where bytes are latency, and int16 rather than float32, which halves it again.

**Still to come:** scheduler with admission control, result cache, Prometheus `/metrics`, `TTSBackend` interface with Piper behind it, Docker + kind + HPA, Go gateway.

---

## Status

| # | Milestone | State |
|---|---|---|
| **M0** | Scaffold + model speaks | ✅ done |
| **M1** | Streaming — chunking policy, then latency floor, then WebSocket | ✅ done |
| **M2** | Scheduler + benchmark harness (p50/p95/p99) | ← current |
| M3 | Docker + kind + HPA + Prometheus/Grafana | |
| M4 | Go gateway + Piper backend comparison | |

**M1 was deliberately ordered chunking → latency → transport.** The obvious order is to build the WebSocket first, but transport moves TTFB by single-digit milliseconds on localhost (measured: 33 ms of 6.4 s). Building it first would have meant re-running every benchmark after the real optimisation landed.

**M2 was re-scoped after finding 7.** It was "dynamic batcher + benchmarks". The batcher half is impossible without graph surgery, so it becomes a scheduler — admission control, queue policy, and the p50/p95/p99 harness that finding 8 already motivates.

---

## The honest bar

Cartesia's Sonic-3 advertises **40–90 ms** time-to-first-audio; independent measurements land nearer **166–190 ms**. That is served GPU infrastructure, and this is one laptop CPU.

The claim here is not "I matched that." It's: *here is the floor on this machine, here is what it decomposes into operator by operator, here is what each lever bought, and here is the model that predicts what different hardware would do.* Being several times off a frontier system with a full account of where the time goes is more useful than a fast number with no decomposition.

---

## Repo map — which script proves which claim

| file | what it establishes |
|---|---|
| `speak.py` | M0 baseline — cold start 909 ms, synthesis 1785 ms, RTF 0.41 |
| `stream.py` | Finding 1 — `create_stream()` vs `create()` vs our chunking |
| `floor.py` | Finding 2 — the fixed/variable fit, `346 + 374 × audio_s` |
| `threads.py` | Finding 3 — the thread sweep |
| `isolate.py` | Finding 3 — attributing it to `intra_op` alone |
| `profile_ops.py` | Finding 2 — per-operator fixed/scaling decomposition |
| `quant.py` | Finding 4 — int8, including the audio-quality check |
| `roofline.py` | Finding 5 — machine ceilings and per-operator placement |
| `kernels.py` | Finding 5 — numpy head-to-head proving the kernel loss |
| `overhead.py` | Finding 8 — cost of the thread hop |
| `server.py` / `client.py` | The server, and findings 6 and 8 |

Every number in this README came from one of these on the machine described at the top. Re-running them is the point.

---

## Stack

Python 3.10 · ONNX Runtime (CPU) · kokoro-onnx · FastAPI + WebSockets · numpy
Planned: Prometheus + Grafana · Docker + kind · Go gateway
