# headstart

A **streaming TTS inference server**. The listener starts hearing audio while the rest of the clip is still being generated — the name is the architecture.

Built to answer one question with real numbers: *how fast can you serve a neural TTS model on a laptop CPU, and where exactly does the latency live?*

**No GPU.** Everything runs on a Ryzen 7 4800H (8C/16T, DDR4-3200). Optimising inference on constrained hardware is the interesting problem; throwing a GPU at it isn't. The findings below are mostly about what *couldn't* be made faster, and why — which turned out to be the more useful half.

---

## The number

Three-sentence paragraph, 17.2 s of audio, measured end-to-end from the client:

| | time to first audio | spread | how |
|---|---|---|---|
| baseline — no streaming | **5380 ms** | 5298–5994 | one `create()` call, nothing emitted until done |
| sentence chunking | **1594 ms** | 1534–1622 | our own splitter; the library's doesn't help (finding 1) |
| + clause-split first chunk | **840 ms** | 828–854 | cut the segment the listener is waiting on (finding 6) |

**6.4× sooner to first sound**, for the same 17.2 s of audio, and the buffer never runs dry — `lead` (audio in hand minus audio already played) stays positive at every chunk, so there is no gap. Total time paid for it: 5999 ms vs 5380 ms, **1.12×**.

> **Median of 3, with the full spread shown, and all three rows measured in one run** by `experiments/headline.py`. Not best-of-N: best-of measures the machine on its luckiest day, which isn't the day the listener gets. A median without a spread would hide the tail — and the tail is exactly what chose the default in finding 6.
>
> All three rows come from one script and one machine state on purpose. A speedup table whose rows were gathered under different conditions isn't a speedup table; it's three unrelated numbers in a column. `experiments/headline.py` also asserts that every row describes the same length of audio, so the comparison can't quietly become "a shorter clip" instead of "a faster one."

Transport is not where the time goes: **connect + WebSocket + framing is 33 ms of a 6.4 s run.** That is why the transport was built *last* — see the ordering note under Status.

---

## The second number

Speed is one question. The other is how many people one machine can serve before the audio starts breaking up. **The answer is 3.3 at once**, and it comes out the same two independent ways: the model needs about 0.3 seconds of compute for every second of audio it makes, and measured throughput stops climbing at 3.27 seconds of audio per second no matter how many clients arrive.

Past that line, a server that accepts everyone gives everyone holes in their audio. So this one stops accepting everyone. Eight clients, six requests each, same build, one flag apart:

| 8 clients at once | requests that stalled | p95 to first audio | turned away |
|---|---|---|---|
| accept everyone | **48 / 48** | 17828 ms | 0 |
| refuse past capacity | **0 / 16** | **7457 ms** | 32 / 48 |

**Every stall gone, and the people who do get served wait less than half as long.** Turning some people away made it better for everyone still inside — the unbounded queue was worse for the people standing in it than being told "not now" would have been.

The order of that queue turned out to matter as much as its length. Plain first-come-first-served — the policy you get for free from a semaphore — **starved 2 of 8 clients completely**, because a refused caller backs off and loses its place to whoever was just served. Every summary metric looked healthy; only counting per client showed it. Findings 11 and 12.

---

## Run it

```bash
docker build -t headstart .
docker run --rm -p 8000:8000 headstart
curl localhost:8000/health
```

That's the whole setup. The image carries the 311 MB of weights, so there is nothing to download and no Python to install — which is the point, since a reader who can't run it can't check any of the numbers below. It's ~1.14 GB as a result, and that trade is deliberate: the alternative is a small image plus a setup step, and the setup step is the thing being removed.

If you'd rather not bake the weights in, `--build-arg FETCH_MODELS=0` and mount your own at `/app/models`.

One flag matters if you constrain CPU. The server defaults to 8 ONNX threads because that's the physical core count it was measured on, and `os.cpu_count()` inside a container reports the *host*, not your quota — so it can't autodetect this correctly. Tell it:

```bash
docker run --cpus 4 -e HEADSTART_INTRA_OP=4 -p 8000:8000 headstart
```

<details>
<summary>Or run it directly, which is what the benchmarks below were measured on</summary>

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

To reproduce findings 11 and 12 — the load sweep with the door on, then off:

```bash
.venv/bin/python bench.py --levels 1,2,3,4,8 --requests 6   # door on (default: 3 streams)
.venv/bin/python server.py --max-inflight 0 &               # door off, then re-run bench.py
```

Model weights are gitignored (311 MB) — pull them with the commands above.

</details>

**To talk to it yourself**, the WebSocket protocol is written up in [`docs/api.md`](docs/api.md) — the request fields, the `start` / `chunk` / `end` / `busy` frames, a real annotated trace, and what to do when you get refused.

**Benchmark on a quiet machine.** `bench.py` re-runs its lowest level at the end as a control and prints the drift; anything over 15% and it declares the run void rather than publishing. It means it — a browser eating a core is enough to fail it, which is a feature rather than an inconvenience, because those numbers would have been wrong and silently plausible.

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

Setting `intra_op` and `inter_op` together and comparing against a run that sets neither gives a number — "1.34×" — that **cannot be attributed to either knob**. `experiments/isolate.py` varies them one at a time:

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

> Both machine ceilings are sanity-checked against the readings that depend on them, because neither fails loudly. The bandwidth probe has to count 3 memory passes, not the 5 a naive read of the loop suggests — a 1.67× error there propagates into *every* scaling claim. And a measured `Conv` rate above the "peak" is arithmetically impossible, so it indicts the peak: the original baseline was simply too weak to be a ceiling.

### 6. Where the first cut goes *is* the latency policy

Block 2 said the floor is ~300 ms, so a 1594 ms first chunk is nowhere near it — because a long first sentence is a long forward pass. Cutting the first segment at a clause boundary takes it to **840 ms** for the same total audio.

How small should that first chunk be? Smaller fails in two directions: total time rises (every chunk pays the ~300 ms fixed cost again), and the buffer can run dry — a tiny first chunk starts playback almost immediately and then has to be fed faster than the model generates. `lead` going negative is the one failure that actually breaks a streaming product. So it was swept (`experiments/leadsweep.py`, median of 3):

| `lead_words` | TTFB median | spread | min lead |
|---|---|---|---|
| off | 1624 ms | 1551–2064 | +4.31 s |
| 8 | 1116 ms | 1066–1155 | +2.73 s |
| 6 | 1013 ms | 981–1015 | +2.20 s |
| **5** | **876 ms** | **786–901** | **+1.90 s** |
| 4 | 710 ms | 683–**2185** | +1.54 s |
| 3 | 625 ms | 614–**2586** | +0.52 s |

**The default is 5, chosen on the tail rather than the median.** Below 5 the median keeps falling, but the spread explodes — 4 and 3 each threw a 2 s+ outlier in three runs, where 8/6/5 threw none. 876 ms with a 115 ms spread beats 625 ms with a 2 s tail, because a serving SLO is written against p99, not p50. TTFB across the sweep also tracks block 2's `300 + 374 × audio_s` fit, so the curve is predictable rather than lucky.

It isn't free. The two renderings correlate at only **+0.71** and diverge 40 ms in, so prosody changes across the whole clip, not just at the seam. Sending `lead_words: 0` turns it off.

**And chunking has a cost that no latency table shows.** The model renders a pause between sentences only when it can see the boundary — generate each sentence separately and the pause silently disappears. Measured: the paragraph in one call is 17.152 s, its three sentences summed are 16.597 s. **555 ms over 2 boundaries, 277 ms each.** Not trimmed edge silence; leading/trailing silence is ~30–90 ms per chunk either way.

So the pause is re-inserted as silence at sentence boundaries — and *not* at a clause split inside a sentence, where there was never a pause to restore. It costs zero model time, and since it's audio handed over for free it **raises** `lead` instead of spending it: the prosody is restored and the buffer margin improves at the same time.

This is invisible to every latency metric in the table, which is why `experiments/headline.py` asserts on audio length as well as time. A chunking policy that ships sooner by quietly emitting less audio would otherwise look like a win.

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

A latency number that doesn't separate queue time from model time isn't a latency number. Finding 10 takes this to percentiles.

### 9. Chunking bottoms out at the architecture, not at the scheduler

If a 5-word first chunk beats a whole sentence, why not go all the way and stream one word at a time? **What I tested:** the same paragraph cut into chunks of 1, 5 and 21 words (`experiments/granularity.py`).

| words per chunk | time per chunk | audio per chunk | total audio produced |
|---|---|---|---|
| 1 | 388 ms | 0.67 s | **18.26 s** |
| 5 | 706 ms | 1.58 s | 9.43 s |
| 21 | 1486 ms | 4.16 s | 8.32 s |

**What I found: generation keeps up fine — the audio is what breaks.** Even one word at a time, generation runs at 0.58× realtime, so the buffer never drains. But a word synthesised on its own is spoken in isolation, with its own pause before and after. The same sentence is 4.31 s in one call and **9.49 s word-by-word**. `"the"` alone renders as 0.73 s, half of it silence, against roughly 0.1 s inside a phrase. Twenty-eight of those is a word list, not a sentence — and 3.6× the compute, since every chunk re-pays the fixed cost from finding 2.

**Why it can't be fixed by cutting smaller still:** the model takes text in and hands finished audio back, and nothing else.

```
IN   tokens [1, sequence_length]   style [1, 256]   speed [1]
OUT  audio  [audio_length]
     state tensors in the signature: none
```

To generate audio incrementally, a model has to hand back its internal state so the next call can resume where the last one stopped. There's nowhere in this signature to put that — no way to ask for "the next 40 ms". It also plans the whole utterance up front (durations are predicted across the full text, and one `STFT` spans the whole signal), so the complete input has to exist before the first sample can.

So the ~300 ms floor is **one full pass of a model that only knows how to run start-to-finish.** Systems that emit audio every few tens of milliseconds are running a loop and tapping it each step — a state-space model carries a small running state precisely so that loop exists. That's a property of the model, not of the server, and no scheduler, batcher or transport changes it.

### 10. How many people can it serve at once? Three.

Every number above was measured with one client on an idle laptop. **What I tested:** 1, 2, 3, 4 and 8 clients all asking for the same paragraph at the same time, each firing its next request as soon as the last finished (`bench.py`, 6 requests per client).

| clients | TTFB p50 | p95 | time spent queueing | spare audio in buffer | requests that stalled |
|---|---|---|---|---|---|
| 1 | 707 ms | 735 ms | 0% | +1.90 s | 0 / 6 |
| 2 | 1504 ms | 3196 ms | 50% | +1.90 s | 0 / 12 |
| 3 | 3774 ms | 5481 ms | 67% | +1.85 s | 0 / 18 |
| 4 | 4555 ms | 7923 ms | 75% | **−1.03 s** | 6 / 24 |
| 8 | 10720 ms | 17404 ms | 88% | **−19.51 s** | 48 / 48 |

**What I found:**

**1. It's all queueing, not slower generation.** Model time per request never moved — 5163, 5303, 5200, 5228, 5222 ms — while waiting grew from 0% to 88% of a request's life. The server isn't degrading under load; requests are just standing in line.

**2. The percentile lies. The buffer doesn't.** `lead` is spare audio in hand: how far ahead of the listener we are. While it's positive, playback is smooth. It goes negative at 4 clients, and at 8 *every* request stalls mid-sentence. At 4 clients p95 reads 7.9 s — bad, but it doesn't look fatal, and by then a quarter of listeners are hearing a hole. **For streaming audio the SLO belongs on buffer margin, not on TTFB percentiles.**

**3. Capacity is 3.3 streams, confirmed two ways.** Generation takes 0.304 s per second of audio, so one machine can feed 1 / 0.304 ≈ 3.3 listeners in real time. Measured throughput does flatten at 3.27 seconds of audio per second, and the first stall appears between 3 clients and 4. Arithmetic and behaviour agree.

**4. That ceiling is already hit by one client.** Extra clients buy queue, not audio. So no scheduler or queue policy raises it — only generating more audio per forward pass would, and per finding 7 this graph's batch dimension is a literal 1. That bounds what M2 can honestly claim before it's built.

Levels run one after another, so the first one sets the baseline everything else is compared against. `bench.py` re-runs that level again at the end and prints the drift; more than 15% apart and it reports the run as void, because a benchmark measured across a changing machine is describing the machine.

### 11. Saying no is the feature. Every stall goes away.

Finding 10 says capacity is about 3 streams and the 4th listener doesn't get slower audio, they get a hole in the middle of a sentence. So the server now refuses work it can't finish: past 3 streams in flight a request waits up to 5.2 s for a slot and is then turned away, **before any audio has been sent**.

**What I tested:** the same sweep as finding 10, run twice back to back — once with the door switched off (`--max-inflight 0`) and once on. Same binary both times, one flag apart, so the arms can't differ by a code change.

| clients | | TTFB p50 | p95 | spare audio in buffer | requests that stalled | turned away |
|---|---|---|---|---|---|---|
| 4 | off | 4573 ms | 8014 ms | **−1.38 s** | **6 / 24** | 0 |
| 4 | on | 6151 ms | 6361 ms | +1.90 s | **0 / 18** | 6 / 24 |
| 8 | off | 10707 ms | 17828 ms | **−21.54 s** | **48 / 48** | 0 |
| 8 | on | 6102 ms | 7457 ms | +1.90 s | **0 / 16** | 32 / 48 |

**What I found:**

**1. Nobody hears a stall any more.** Stalls went from 6 of 24 and 48 of 48 to zero at both levels. Spare audio in the buffer holds at +1.90 s under 8 clients — the same margin a single client gets. **Whoever gets served now gets the single-client experience, and everyone else is told up front instead of finding out mid-sentence.**

**2. The 4th client's median got worse on purpose.** At 4 clients, p50 rose 4573 → 6151 ms, because an admitted client now waits at the door before its first byte. That is the trade being bought: a longer wait is legible to a listener as loading, a hole in a sentence is not — they can't tell server load from a broken product, so they conclude the product is broken.

**3. Past capacity it got better on every measure at once.** At 8 clients p50 fell 10707 → 6102 ms and p95 fell 17828 → 7457 ms. Nothing was optimised; the queue simply stopped being allowed to grow. **The unbounded queue was worse for the people standing in it than being turned away would have been.**

**4. It costs no throughput.** Peak 3.23 vs 3.27 seconds of audio per second, and 3.20 vs 3.27 at 8 clients — unchanged within run-to-run noise. The door reallocates who waits; it does not create or destroy capacity, and per finding 10 nothing at this layer could.

**5. The share served tracks capacity, and the shortfall is the price of backing off.** At 4 clients, 18 of 24 got through — exactly 3/4, the capacity share. At 8 clients it's 16 of 48, or 1/3, a little under the 3/8 the arithmetic predicts. The reason shows up in the door wait: admitted clients wait 2341 ms at 4 clients but only 616 ms at 8. Past a point, refused clients are all off backing off at the same time, so a freed slot sometimes has nobody standing at it. **Politeness costs a few percent of capacity** — worth knowing before tuning the retry hint, which is the knob that trades it against refusal churn.

Both runs pass the drift check (1.01× and 1.00×), so this is a comparison of two servers, not a story about the machine getting busier between them.

**It reproduces.** A later run on a busier machine (drift 1.11×, RTF 0.300 against 0.307) gave the same answer where it counts: zero stalls at every level, nobody starved, and 6 of 24 then 31 of 48 refused against 6 and 32 here. The tail was wider and the buffer margin wobbled — that's the noise showing up where noise should. The result isn't one lucky afternoon.

### 12. First-come-first-served starved two clients out of eight

The door began as plain first-come-first-served, which sounds like the fair answer. **What I tested:** instead of only counting how many requests got served, `bench.py` now counts them **per client** — because a served total reads identically whether the server rotates fairly or serves the same three people every time. The first-come-first-served run put 18 of 48 through at 8 clients; the total said nothing about who they went to.

| 8 clients, 6 requests each | requests served, per client | served in total | got nothing |
|---|---|---|---|
| first-come-first-served | 4, 4, 4, 3, 2, 1, **0, 0** | 18 of 48 | **2 of 8** |
| refusals move you up the queue | 4, 3, 2, 2, 2, 1, 1, 1 | 16 of 48 | 0 of 8 |

The two rows are separate runs, so the totals differ a little — 18 and 16 — which is exactly the point: the number that changed by two says nothing, and the number that changed from two-starved to none is the whole finding. The aged row reproduced exactly in the published run of finding 11: same split, same spread, nobody starved.

**What I found:**

**1. Being refused made the next refusal more likely.** A refused client backs off before retrying, and while it's backing off it isn't in the queue at all — so it comes back *behind* the clients that were just served, who re-join the instant they finish. Two clients were turned away all six times. **Backing off politely costs you your place, and first-come-first-served has no memory of that.**

**2. The totals hid it completely.** Served counts, latency percentiles, stall counts and throughput were all fine in both versions. Only the per-client breakdown showed two people getting nothing, which is why the check is now part of the harness and not something I looked at once.

**3. The fix is to let the door remember.** Each refusal now raises that caller's priority; a freed slot goes to whoever has been refused most, with arrival order breaking ties, and a successful admission resets them to zero. Everyone gets served at least once, and the gap between best- and worst-served client narrows from 4 to 3.

This is the point of writing the queue out by hand rather than using a semaphore. A semaphore *is* a scheduling policy — first-come-first-served, wait forever, no visibility — it just doesn't look like one, so the policy never gets chosen, measured, or found to be wrong.

---

## Architecture

**What runs today:**

```
  client ──WS──▶  FastAPI /tts
                    │  split text into chunks (sentence, or clause for the first)
                    │
                    │  ┌── the door ── max 3 streams, held for the whole request
                    │  │   full? wait up to 5.2 s, then refuse with a retry hint
                    │  │   order: most-refused first, then longest-waiting
                    │  └── refused ──▶ {"type":"busy"}, no audio sent
                    │
                    │  ├─ asyncio.Semaphore(1)  ← one model call at a time; queue is measured
                    │  └─ asyncio.to_thread     ← keeps the event loop alive
                    │
                    │  + 277 ms silence at sentence boundaries  ← free audio, restores the pause
                    │  per chunk: JSON metadata frame, then raw int16 PCM frame
  client ◀──────────┘  playback starts on chunk 0, while chunk 1 is still generating
```

Two gates, doing different jobs. The **door** decides *whether* you get served, and holds its slot for the whole request — releasing it between chunks would let a new stream in to compete with one already mid-sentence, which is the thing it exists to prevent. The **model slot** decides *when* each individual chunk runs. The door bounds the queue; the slot serialises what's in it.

Metadata as JSON, audio as a separate binary frame — **not** base64 inside the JSON, which is +33% on the one payload where bytes are latency, and int16 rather than float32, which halves it again.

**Still to come:** result cache, Prometheus `/metrics`, `TTSBackend` interface with Piper behind it, Docker + kind + HPA, Go gateway.

---

## Status

| # | Milestone | State |
|---|---|---|
| **M0** | Scaffold + model speaks | ✅ done |
| **M1** | Streaming — chunking policy, then latency floor, then WebSocket | ✅ done |
| **M2** | Scheduler + benchmark harness (p50/p95/p99) | ✅ done — harness, admission control, queue policy |
| M3 | Docker + kind + HPA + Prometheus/Grafana | |
| M4 | Go gateway + Piper backend comparison | |

**M1 was deliberately ordered chunking → latency → transport.** The obvious order is to build the WebSocket first, but transport moves TTFB by single-digit milliseconds on localhost (measured: 33 ms of 6.4 s). Building it first would have meant re-running every benchmark after the real optimisation landed.

**M2 was re-scoped after finding 7.** It was "dynamic batcher + benchmarks". The batcher half is impossible without graph surgery, so it becomes a scheduler — admission control and queue policy — plus the harness. The harness ran first on purpose: a scheduler is worth building only if the queue owns the tail, and finding 10 shows it owns 88% of it, with a hard capacity line at 3.3 streams that no queue policy can move. That bounded what the scheduler could honestly claim before a line of it was written — and findings 11 and 12 claim exactly that and no more: every stall removed, fair rotation, and not one extra second of audio per second.

---

## The honest bar

Cartesia's Sonic-3 advertises **40–90 ms** time-to-first-audio; independent measurements land nearer **166–190 ms**. That is served GPU infrastructure, and this is one laptop CPU.

The claim here is not "I matched that." It's: *here is the floor on this machine, here is what it decomposes into operator by operator, here is what each lever bought, and here is the model that predicts what different hardware would do.* Being several times off a frontier system with a full account of where the time goes is more useful than a fast number with no decomposition.

And per finding 9, most of the remaining gap isn't a serving gap at all. A one-shot graph's floor is a full forward pass; an incremental model's is one step of a loop. Knowing which of those you're holding decides whether the next win comes from engineering or from a different model.

---

## Repo map — which script proves which claim

The root holds the product; `experiments/` holds the evidence. The split is deliberate: the three files at the top are what you'd run or deploy, and the thirteen below them are one-shot measurements whose output is a number in this README.

**The server**

| file | what it is |
|---|---|
| `server.py` | The server and the door. Also findings 6, 8, 11, 12 |
| `client.py` | Reference client — streams, plays, and reports what it experienced |
| `bench.py` | Findings 10, 11, 12 — percentiles and saturation, the door on/off comparison, the per-client fairness check |

**The evidence** — `experiments/`

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
| `leadsweep.py` | Finding 6 — the first-chunk sweep that chose `lead_words=5` |
| `headline.py` | The table at the top — all three rows, one run, one methodology |
| `granularity.py` | Finding 9 — chunk-size sweep, citation-form cost, and the graph signature |

Run them from anywhere in the checkout — `experiments/_root.py` anchors the working directory to the repo root, so the relative paths inside each script stay correct and `import server` still resolves. The server itself takes the other route and reads `HEADSTART_MODEL` from the environment, because unlike these it does have to run somewhere else.

Every number in this README came from one of these on the machine described at the top. Re-running them is the point.

---

## Stack

Python 3.10 · ONNX Runtime (CPU) · kokoro-onnx · FastAPI + WebSockets · numpy
Planned: Prometheus + Grafana · Docker + kind · Go gateway
