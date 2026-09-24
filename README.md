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

## Run it

```bash
docker build -t headstart .
docker run --rm -p 8000:8000 headstart
curl localhost:8000/health
```

That is the whole setup. The image carries the 311 MB of weights, so there is nothing to download and no Python to install — which matters, because a reader who cannot run it cannot check any of the numbers below. The cost is a ~1.14 GB image. The alternative was a small image plus a setup step, and the setup step is precisely what I wanted gone.

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

To reproduce the load findings — the sweep with admission control on, then off:

```bash
.venv/bin/python bench.py --levels 1,2,3,4,8 --requests 6   # admission control on
.venv/bin/python server.py --max-inflight 0 &               # admission control off, then re-run bench.py
```

Capacity can also be moved on a running server, which is how the three-caller table above was measured without restarting between arms:

```bash
curl -X POST "localhost:8000/admission?capacity=3"    # pin it; 0 turns admission control off
curl -X POST "localhost:8000/admission?capacity=-1"   # hand it back to the measurement
```

Naming a number pins it, and `-1` hands it back to the measurement, so a reader who has just pinned capacity open to watch it fail can return to the default without a restart. Omit `--max-inflight` and capacity tracks the measured realtime speed from the start.

Model weights are gitignored (311 MB) — pull them with the commands above.

</details>

To talk to it yourself, the WebSocket protocol is written up in [`docs/api.md`](docs/api.md): the request fields, the `start` / `chunk` / `end` / `busy` frames, an annotated trace from a real run, and what to do when you get refused.

**Benchmark on a quiet machine.** `bench.py` re-runs its lowest level at the end as a control and prints the drift; anything over 15% and it declares the run void rather than publishing. It means it — a browser eating a core is enough to fail a run. That is worth the inconvenience, because a benchmark contaminated that way produces numbers that are wrong and entirely plausible-looking.

---

## What I found

### 1. The library's streaming is not a latency feature

`kokoro-onnx` exposes `create_stream()`, which reads like the answer. It isn't. It splits only on `MAX_PHONEME_LENGTH` (510) — a guard against overrunning the model's input window. **Any text under that limit yields exactly one chunk, so `create_stream()` is `create()`.** Worse, its splitter deliberately *balances* batch sizes for prosody, which is the opposite of what time-to-first-byte wants.

So the chunking had to be written here. The finding took twenty minutes in the library's source and would not have come from its documentation, which describes `create_stream()` in exactly the terms you would hope for.

### 2. Chunking has a floor, and the floor is inside the model

Every call pays a **fixed** cost (phonemization, tokenization, style lookup, session dispatch) and a **variable** cost (the forward pass, proportional to audio). Chunking shrinks the variable part only, so TTFB approaches the fixed cost and stops. Fitting a line through progressively longer prefixes:

```
gen_s = 0.35 + 0.35 × audio_seconds        (n=10, max residual 66 ms)
```

The server refits this continuously from its own completed chunks rather than trusting the constant, and on a warm box it settles near `0.39 + 0.31 × audio_s`, with about 0.30 s of audio per word. Every derived number below — chunk size, capacity, how long admission control waits — comes out of those three coefficients, and `/health` publishes them so the arithmetic can be checked rather than believed.

Phonemization is **0.3 ms** of the fixed term. The floor lives inside the ONNX session, so no amount of smarter text handling reaches it.

Profiling per-operator gives an independent decomposition — **299 ms fixed, 327 ms/audio-second.** Both terms land below the fit above — the fixed one by 1.14×, the scaling one by 1.07× — and in the direction the thread win predicts, since the fit was run untuned and the profile tuned. Two unrelated methods agreeing on the shape and on the size of both terms is the reason I trust either.

**What the floor is made of:**

| | share of the 299 ms floor |
|---|---|
| `Sin` | 32% |
| `Conv` | 29% |
| `STFT` | 12% |
| `ConvTranspose` | 11% |

**This is a convolutional vocoder, not a transformer.** `MatMul` is 4.5 ms of 299 — under 2%. Every instinct trained on LLM serving points at the wrong operator here. `Sin` is top of the list because Kokoro's vocoder builds speech by summing sine waves at the predicted pitch, over ~44 million values per sentence.

Framework overhead — Python, the ONNX dispatcher, everything that isn't math — is **2%** (669 ms of kernel time inside a 686 ms wall). There is no free win in the plumbing.

### 3. Thread tuning is free, and only one of the two knobs does anything

8 threads beats the default and beats 16. 8 is the physical core count; two SMT threads on one core share a load/store path, so the second one contends rather than helps.

The tempting way to report this is to set `intra_op` and `inter_op` together, compare against a run that sets neither, and quote the difference. That number cannot be attributed to either knob. `experiments/isolate.py` varies them one at a time — median of 3 runs, best-of-5 within each:

| config | short chunk | full sentence |
|---|---|---|
| default | 781 ms | 1914 ms |
| `intra_op=8` only | **583 ms** | **1541 ms** |
| `inter_op=1` only | 729 ms | 1864 ms |
| both | 589 ms | 1527 ms |

**`intra_op` is the entire win on both texts, and `inter_op` is worth nothing on either.** Setting both is no better than setting `intra_op` alone, within the run-to-run spread — the three runs put `intra_op` only at 585/583/583 ms on the short chunk against 589/589/582 for both. So the honest headline is **1.34× on a short chunk and 1.24× on a full sentence, from one setting**. `inter_op=1` ships anyway because it costs nothing and makes the thread budget explicit, but it is not carrying any of the number.

**What 8 threads is worth against one.** The comparison above is against ONNX Runtime's default, which is one thread per logical core — 16 here, not 1 — so it is a tuning result and not a parallel-efficiency figure. The sweep in `experiments/threads.py` has the anchor: the full sentence takes **3769 ms on one thread and 1532 ms on eight**, so eight cores buy 2.46× and roughly **31% parallel efficiency**. The short chunk behaves the same way, 1499 ms against 585 ms. That number matters later — finding 8 explains why a second concurrent request does not get its own cores, and 31% is the reason the cores are already spoken for.

### 4. int8 quantization is structurally unavailable here

Predicted ~1.27× from the matmul-family share. Then the quantized model wouldn't load:

```
NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
```

**ONNX Runtime's `ConvInteger` CPU kernel is 2-D only, and all 94 convolutions in this graph are 1-D** (audio slides a filter over time; images slide over height *and* width). So the operator holding 86 of the 93 quantizable milliseconds cannot be quantized on this runtime at all. Quantizing `MatMul` alone: **1.00× speedup**, and phoneme durations shifted by 21–64 ms — no speed, different audio.

Also worth knowing before trying: this CPU is Zen 2, which has no VNNI, so int8 was only ever going to be a memory-traffic win rather than an instruction win.

What this finding delivers is a diagnosis rather than a speedup. "int8 gave 1.27×" survives no follow-up questions. "The runtime's integer convolution kernel is 2-D and every convolution in this model is 1-D, which I found by reading weight ranks after the model refused to load" survives all of them.

### 5. Mixed-bound, which makes hardware scaling piecewise

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

Finding 2 says the floor is ~300 ms, so a 1594 ms first chunk is nowhere near it — because a long first sentence is a long forward pass. Cutting the first segment at a clause boundary is what closes that gap, and how small to cut is a policy question with a measurable answer.

Smaller fails in two directions: total time rises (every chunk pays the fixed cost again), and the buffer can run dry — a tiny first chunk starts playback almost immediately and then has to be fed faster than the model generates. Running dry is the one failure that actually breaks a streaming product. Swept with `experiments/leadsweep.py`, median of 3:

| words in the first chunk | TTFB median | spread | thinnest buffer |
|---|---|---|---|
| whole sentence | 1624 ms | 1551–2064 | +4.31 s |
| 8 | 1116 ms | 1066–1155 | +2.73 s |
| 6 | 1013 ms | 981–1015 | +2.20 s |
| 5 | 876 ms | 786–901 | +1.90 s |
| 4 | 710 ms | 683–**2185** | +1.54 s |
| 3 | 625 ms | 614–**2586** | +0.52 s |

**Below about 5 words the median keeps falling but the spread explodes** — 4 and 3 each threw a 2 s outlier in three runs, where 8, 6 and 5 threw none. 876 ms with a 115 ms spread beats 625 ms with a 2 s tail, because a serving SLO is written against p99, not p50.

So the server derives the number from its own fit — the audio that fits in the TTFB budget, divided by seconds per word, which lands between 4 and 7 on this box as the box warms up and cools — and clamps it at 4, where this sweep says the tail starts. **The fit picks the number and the measured variance sets the floor under it.** Neither half is a constant typed on a different machine.

It isn't free. The two renderings correlate at only **+0.71** and diverge 40 ms in, so prosody changes across the whole clip, not just at the seam. Sending `lead_words: 0` turns it off.

**And chunking has a cost that no latency table shows.** The model renders a pause between sentences only when it can see the boundary — generate each sentence separately and the pause silently disappears. Measured: the paragraph in one call is 17.152 s, its three sentences summed are 16.597 s. **555 ms over 2 boundaries, 277 ms each.** Not trimmed edge silence; leading/trailing silence is ~30–90 ms per chunk either way.

So the pause is re-inserted as silence at sentence boundaries — and *not* at a clause split inside a sentence, where there was never a pause to restore. It costs zero model time, and since it's audio handed over for free it **deepens** the buffer instead of spending it: the prosody is restored and the margin improves at the same time.

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

How many model calls run at once is a setting, and the first version of this server set it to one. The reasoning was that `intra_op=8` already hands every operator all 8 physical cores, so a second caller finds nothing idle and only makes both tails worse. Half of that is measured: finding 3 puts the 8-thread session at 2.46× against one thread, about 31% efficiency. The other half is the part worth testing, because 31% efficiency means the machine is mostly not working — and a second caller is one way to use the rest. **Finding 18 measures it and the slot now runs four calls at once.**

The same arithmetic is behind "why not shard the cores, four per stream?" Per-stream speed falls much more slowly than the core count does: inverting the real-time factors from the same sweep gives 2.78x at 8 threads, 2.38x at 4, 1.82x at 2, so two 4-thread instances look like 4.76x against 2.78x — if they scaled perfectly against each other, which they will not, because they share one 8 MiB L3 and one memory controller. ONNX Runtime exposes `intra_op_thread_affinities` to pin them properly. `experiments/sharding.py` runs that arm against the shared-session one; finding 18 has the table.

Calls still queue once every slot is full, so the queue is **measured**: every chunk reports `queue_ms` separately from `gen_ms`. Two clients against the serialised slot:

```
client A   time to first audio  1436 ms
client B   time to first audio  2560 ms      ← waited one full generation
```

A latency number that doesn't separate queue time from model time isn't a latency number. Findings 10 and 13 are both consequences of keeping them apart.

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

### 10. Under load it is all queueing, and the percentile lies

Every number above was measured with one client on an idle laptop. **What I tested:** 1, 2, 3, 4 and 8 clients all asking for the same paragraph at the same time, each firing its next request as soon as the last finished (`bench.py`, 6 requests per client, admission control off so the load actually lands).

| clients | TTFB p50 | p95 | time spent queueing | spare audio in buffer | requests that stalled |
|---|---|---|---|---|---|
| 1 | 707 ms | 735 ms | 0% | +1.90 s | 0 / 6 |
| 2 | 1504 ms | 3196 ms | 50% | +1.90 s | 0 / 12 |
| 3 | 3774 ms | 5481 ms | 67% | +1.85 s | 0 / 18 |
| 4 | 4555 ms | 7923 ms | 75% | **−1.03 s** | 6 / 24 |
| 8 | 10720 ms | 17404 ms | 88% | **−19.51 s** | 48 / 48 |

**What I found:**

**1. Generation never slows down; the line in front of it gets longer.** Model time per request never moved — 5163, 5303, 5200, 5228, 5222 ms — while waiting grew from 0% to 88% of a request's life. The server isn't degrading under load; requests are standing in line.

**2. The percentile lies. The buffer doesn't.** Spare audio in hand is how far ahead of the listener the server is. While it's positive, playback is smooth. It goes negative at 4 clients, and at 8 *every* request stalls mid-sentence. At 4 clients p95 reads 7.9 s — bad, but it doesn't look fatal, and by then a quarter of listeners are hearing the audio break up. **For streaming audio the SLO belongs on buffer margin, not on TTFB percentiles.**

**3. The ceiling is already hit by one client.** Extra clients buy queue, not audio. No scheduler or queue policy raises it: ordering decides who waits, not how much the box finishes. That bounds what any scheduler here can honestly claim, and findings 11 through 16 claim no more than it.

What *does* raise it is how much of the machine is running at once, which is a different lever from who goes next — finding 18 lifts this same measurement to 5.12× by widening the slot. Past the widened line the only remaining move is more audio per forward pass, and per finding 7 this graph's batch dimension is a literal 1.

Levels run one after another, so the first one sets the baseline everything else is compared against. `bench.py` re-runs that level again at the end and prints the drift; more than 15% apart and it reports the run as void, because a benchmark measured across a changing machine is describing the machine.

### 11. Capacity is the measured realtime speed, not the asymptote

Finding 10 gives a wall but not a number, and the number is what admission control needs. **What I tested:** the two ways of computing it, against where the buffers actually break.

The arithmetic answer is `1 / slope`. Generation costs about 0.31 s of compute per second of audio, so one machine feeds `1 / 0.31 ≈ 3.2` listeners in real time. The measured answer is the server's own record of what it produced: total audio made over total time spent making it, which read **2.2 to 3.0** depending on how busy the box was. (This finding was measured with the slot serialised, so "time spent making it" and "time one call took" were the same quantity. Finding 18 separates them and capacity now reads off the delivered one — the argument below is unaffected, because at width 1 the two agree.)

**What I found: the measurement is right and the arithmetic is optimistic, by exactly one stream.** Three callers, same server, capacity moved through `/admission` between arms:

| admission control | streams served | refused | underruns | silence, per run |
|---|---|---|---|---|
| 3 | 3 | 0 | 5, 6, 9, 10, 11 | **2.7 s, 6.4 s, 7.3 s, 20.0 s, 21.9 s** |
| 2 | 2 | 1 | 0, 0, 1, 1, 2, 2, 2 | **0 ms, 0 ms, 0.1 s, 0.8 s, 0.8 s, 1.7 s, 2.3 s** |

Slope is the *marginal* cost of one more second of speech, so `1 / slope` is the capacity of an infinitely long chunk. This server does not serve infinitely long chunks — it serves short ones deliberately, because that is what buys TTFB, and it re-pays the fixed cost on every single one. **The missing stream is the fixed cost, charged once per chunk.** The asymptote is a real number about the model; it is the wrong number about this server.

So capacity is read off the fit window rather than derived, and admission control walks toward it one slot at a time — never in a jump, because a capacity number that moves on every sample is one nobody can act on, and never by evicting anyone, because shrinking simply refuses the next arrival. An operator who names a number with `--max-inflight` or `/admission` keeps it; `/admission?capacity=-1` hands it back. Auto-tracking is for the default, where the alternative is a constant typed on a different box.

**It is a measurement of this box right now, and it moves.** On a quiet machine it reads 3.0x and capacity opens to 3; the third stream then lands, contention drops the measured speed back under 3.0x, and capacity closes to 2 again. That self-correction is the mechanism working — but it works *after* three streams have already taken damage, which is the honest limit of measuring capacity from the inside. **Every other bound here carries a safety margin and this one does not**, so a server sitting exactly on 3.0 will flap. Rounding down is the only protection it has, and rounding down is not a margin.

**This is what the capacity ceiling costs when it is guessed high.** One stream over the line is not a slightly worse experience for three people. It is five to eleven underruns, in all three of them, on every run.

### 12. Latency is paid for out of the buffer

Chunking small enough to start speaking early leaves the listener holding less finished audio, and that margin is the only thing standing between a busy server and an underrun. **What I tested:** the 39-word-opener text from the top of this README, on the same server, one request field apart — whole sentences, or chunks sized for TTFB. It is a longer opener than the sweep in finding 6 uses, so the buffer depths here are deeper than that table's and the two are not directly comparable.

| first chunk | TTFB, one stream | thinnest buffer | silence, two streams |
|---|---|---|---|
| whole sentences | 4088 ms (4004–4108) | 12.1 s | **0 ms** |
| sized for TTFB | 1393 ms (1329–1430) | 2.7 s | **1514 ms** |

**What I found: the bill is not capacity, it is buffer depth.** Realtime speed was 2.17x and 2.32x across these two arms, and 2.54x and 2.39x on an earlier pair — the difference changes sign between runs, so chunking costs essentially nothing in raw capacity. The chunk count barely moves either: 4 turns at the model slot instead of 5.

What moves by 4.4× is the thinnest the buffer ever gets. Whole sentences hand the listener 12.1 s of audio up front and never dip below it; chunks sized for TTFB never build more than 2.7 s. On an idle box both are fine. With a second caller the wait for the model slot runs past a second and keeps growing, and **a queue wait longer than the buffer is an underrun by definition** — 2.7 s of margin is thin enough to lose, 12.1 s is not.

So the trade is legible and it is the listener's to make: **starting 2.9× sooner costs 4.4× the safety margin.** A voice agent answering a question wants the early start and gets it. A long narration nobody is waiting on should send `lead_words: 0` and keep the margin. The server takes the field per request rather than picking for everyone.

### 13. The listener waits through the queue, so the plan has to see it

The chunk planner sizes each chunk against the audio already in the listener's hands: don't promise more speech than there is time to make before the buffer runs out. That works on an idle box and drifts on a busy one, because the cost model it plans against is deliberately queue-free.

**What I tested:** whether the queue belongs in the fit, in the plan, or in both. The measurement is a rolling wait for the model slot, recorded on every chunk alongside generation time and reported at `/health`.

**What I found: it belongs in the plan and must stay out of the fit.** Folding the wait into the fit makes a busy box read as a slow box, and the coefficients then inflate themselves — every stream that queues teaches the model that generation got more expensive, which it didn't. But the plan cannot ignore it, because the listener's buffer drains on wall-clock time and does not care which part of the wait was queueing. Measured on one side, added back on the other.

Two things fall out once the queue is in the plan:

**The ceiling falls and the floor rises at the same time.** A chunk earns `a` seconds of buffer and spends `queue + fixed + slope × a` getting there, so it only leaves the listener better off while

```
a × (1 − slope) > fixed + queue
```

Below that line every chunk is a net withdrawal and the stream drains however carefully the next one is sized. **Cutting smaller to catch up is exactly backwards**, because each extra chunk pays the queue again. Under contention there is a shortest chunk that is worth generating at all, and it gets longer as the box gets busier — `/health` publishes it as `min_chunk_s`, and it moves from 0.53 s idle to 2.25 s with three callers arriving.

**When the floor rises above the ceiling there is no chunk size that works.** That is a statement about admission, not about chunking, and it cannot be chunked around: the box is taking in more work than it can finish. `/health` reports it as `sustainable`, and it is the same conclusion finding 11 reaches from the other direction.

Both bounds carry the same 30% margin, for the same reason. The shortest worthwhile chunk is exactly break-even, so a chunk sized there grows the buffer by nothing and the first jitter is an underrun; sizing above break-even by the same factor the ceiling holds back is what makes the margin grow instead of merely holding.

`HEADSTART_QUEUE_AWARE=0` turns the queue term off, for the same reason `--max-inflight 0` exists: the claim is a comparison, and a comparison that needs two builds is one nobody reproduces.

### 14. Saying no is the feature. Every stall goes away.

Findings 10 and 11 say the machine carries two streams and the third listener doesn't get slower audio, they get silence in the middle of a sentence. So the server refuses work it can't finish: past capacity a request waits a few seconds for a slot and is then turned away, **before any audio has been sent**, with a retry hint. How long it waits is itself measured — the average time a request holds its slot, divided by the number of slots — rather than a constant.

**What I tested:** a sweep at 4 and 8 clients, run twice back to back — once with admission control switched off (`--max-inflight 0`) and once on. Same binary both times, one flag apart, so the arms can't differ by a code change.

| clients | | TTFB p50 | p95 | spare audio in buffer | requests that stalled | turned away |
|---|---|---|---|---|---|---|
| 4 | off | 4573 ms | 8014 ms | **−1.38 s** | **6 / 24** | 0 |
| 4 | on | 6151 ms | 6361 ms | +1.90 s | **0 / 18** | 6 / 24 |
| 8 | off | 10707 ms | 17828 ms | **−21.54 s** | **48 / 48** | 0 |
| 8 | on | 6102 ms | 7457 ms | +1.90 s | **0 / 16** | 32 / 48 |

**What I found:**

**1. Nobody hears a stall any more.** Stalls went from 6 of 24 and 48 of 48 to zero at both levels. Spare audio in the buffer holds at +1.90 s under 8 clients — the same margin a single client gets. **Whoever gets served now gets the single-client experience, and everyone else is told up front instead of finding out mid-sentence.**

**2. The 4th client's median is worse on purpose.** At 4 clients, p50 is 6151 ms with admission control on against 4573 ms with it off, because an admitted client waits for admission before its first byte. That is the trade being bought: a longer wait is legible to a listener as loading, a gap in a sentence is not — they can't tell server load from a broken product, so they conclude the product is broken.

**3. Past capacity it is better on every measure at once.** At 8 clients p50 falls 10707 → 6102 ms and p95 falls 17828 → 7457 ms. Nothing was optimised; the queue simply stopped being allowed to grow. **The unbounded queue was worse for the people standing in it than being turned away would have been.**

**4. It costs no capacity.** Peak 3.23x vs 3.27x realtime, and 3.20x vs 3.27x at 8 clients — unchanged within run-to-run noise. Admission control reallocates who waits; it does not create or destroy capacity, and per finding 10 nothing at this layer could.

**5. The share served tracks capacity, and the shortfall is the price of backing off.** At 4 clients, 18 of 24 got through — exactly 3/4, the capacity share. At 8 clients it's 16 of 48, a little under what the arithmetic predicts. The reason shows up in admission control wait: admitted clients wait 2341 ms at 4 clients but only 616 ms at 8. Past a point, refused clients are all off backing off at the same time, so a freed slot sometimes has nobody standing at it. **Politeness costs a few percent of capacity** — worth knowing before tuning the retry hint, which is the knob that trades it against refusal churn.

Both runs pass the drift check (1.01× and 1.00×), so this is a comparison of two servers, not a story about the machine getting busier between them.

**It reproduces.** A later run on a busier machine (drift 1.11×, RTF 0.300 against 0.307) gave the same answer where it counts: zero stalls at every level, nobody starved, and 6 of 24 then 31 of 48 refused against 6 and 32 here. The tail was wider and the buffer margin wobbled — that's the noise showing up where noise should. The result isn't one lucky afternoon.

> Two notes on reading this table against finding 11. These rows were measured with capacity pinned at 3 rather than tracking the measurement, so the "on" rows are one stream more generous than the shipped default; the comparison is admission control on against off, so the level it is pinned at does not change the conclusion. And the 3.23x above is not the same quantity as the 2.2–3.0x the server reads for itself: `bench.py` counts every second of audio the listener receives, including the inserted sentence pauses that cost no model time, and it measures a saturated box where chunks are long. The server's own number counts only what came out of the model, at the chunk sizes it is actually serving. **The higher number is the friendlier one, which is why the server does not use it.**

### 15. The order of the queue decides who starves

First-come-first-served sounds like the fair answer at admission control. **What I tested:** counting served requests **per client** rather than in total — because a served total reads identically whether the server rotates fairly or serves the same three people every time.

| 8 clients, 6 requests each | requests served, per client | served in total | got nothing |
|---|---|---|---|
| first-come-first-served | 4, 4, 4, 3, 2, 1, **0, 0** | 18 of 48 | **2 of 8** |
| refusals move you up the queue | 4, 3, 2, 2, 2, 1, 1, 1 | 16 of 48 | 0 of 8 |

The two rows are separate runs, so the totals differ a little — 18 and 16 — which is exactly the point: the number that changed by two says nothing, and the number that changed from two-starved to none is the whole finding. The aged row reproduced exactly in the published run of finding 14: same split, same spread, nobody starved.

**What I found:**

**1. Being refused makes the next refusal more likely.** A refused client backs off before retrying, and while it's backing off it isn't in the queue at all — so it comes back *behind* the clients that were just served, who re-join the instant they finish. Two clients were turned away all six times. **Backing off politely costs you your place, and first-come-first-served has no memory of that.**

**2. The totals hid it completely.** Served counts, latency percentiles, stall counts and delivered audio per second were all healthy in both arms. Only the per-client breakdown showed two people getting nothing, which is why the check is now part of the harness and not something I looked at once.

**3. So admission control remembers.** Each refusal raises that caller's priority; a freed slot goes to whoever has been refused most, with arrival order breaking ties, and a successful admission resets them to zero. Everyone gets served at least once, and the gap between best- and worst-served client narrows from 4 to 3.

This is the point of writing the queue out by hand rather than using a semaphore. A semaphore *is* a scheduling policy — first-come-first-served, wait forever, no visibility — it just doesn't look like one, so the policy never gets chosen, measured, or found to be wrong.

### 16. The buffer is the deadline

The same argument runs one layer down. Two streams sharing the model slot are not symmetric: one can be holding five seconds of finished audio and the other none, and handing the slot to whichever called first is what leaves the second one silent.

**What I tested:** two callers under a slot handed out in arrival order. **What I found: the total silence barely moves, but it picks a different victim each run** — one stream finished clean and the other lost 2.8 s, and which one it was changed between runs. A summary metric that adds them together cannot see this at all, which is the same blind spot finding 15 found at the admission layer.

So the slot goes to whoever has the least audio in hand. That is earliest-deadline-first under another name, and it is the right rule here because the buffer is exactly the deadline — it is how long this stream can wait before the listener hears silence. **Chunk 0 of a new stream has no buffer at all, so it sorts to the front for free**: giving new arrivals priority is not a special case bolted on, it is what the general rule does with a zero.

**The honest caveat: at the capacity this server actually runs at, it never fires.** `/health` counts every occasion need-order beat arrival-order. With the model serialised and admission control at 2, eight callers arriving together leave the counter at **0** — two admitted streams sharing one slot are never both waiting, so there is nothing to reorder. Pin admission control one stream over and the same eight callers make it fire **7 to 13** times, because a third admitted stream is what puts a second waiter in the queue. Turn admission control off and it fires **42 to 69** times. The rule is busiest exactly where the server is least defensible, and silent where it is behaving.

So on CPU the rule is free insurance against a state admission control exists to prevent. It earns its place when waiters are the normal condition rather than a misconfiguration — which is a claim about a GPU I don't have, written down as a prediction rather than a result.

**Finding 18 moved this further from firing, not closer.** Widening the model slot to four concurrent calls means two admitted streams never queue behind each other at all, so the counter now stays at 0 for a reason the old text did not anticipate: not "one slot, never two waiters" but "more slots than admitted streams". The rule survives the change untouched, which is the argument for writing it as a policy over need rather than as a semaphore — but it is now idle for a second reason, and both are worth knowing before anyone quotes it as a load-bearing feature.

### 17. Changing the voice changes TTFB, and speaking rate is the whole of it

Swapping the voice id moves time to first audio by half again, with the text, the machine and the chunk plan all held still. That looks like some voices being more expensive to run, which would be a problem, because the fix for an expensive model is not the fix for a long one.

**What I tested:** seven voices, one sentence, one stream at a time (`--max-inflight 1` on a freshly started server), three runs each, medians, `experiments/voice_ttfb.py`. The first chunk is the same nine words for every voice — the clause rule in finding 6 cuts at the same `that` whether `lead_words` is 5 or the 9–11 the server derives — so the only thing that varies is the voice. TTFB came out equal to the first chunk's generation time to within 1 ms everywhere, so there is no queue or admission wait hiding in these numbers.

| voice | first chunk | TTFB | ms per second of audio |
|---|---|---|---|
| am_adam | 2.50 s | 792 ms | 317 |
| bf_emma | 2.60 s | 831 ms | 319 |
| af_sarah | 2.62 s | 860 ms | 328 |
| af_bella | 2.77 s | 997 ms | 359 |
| am_michael | 3.03 s | 900 ms | 297 |
| bm_george | 3.24 s | 966 ms | 298 |
| af_nicole | 3.73 s | 1193 ms | 319 |

**What I found: the same nine words are 2.50 s of speech in one voice and 3.73 s in another, a 1.50× spread, and TTFB moves 1.51× in step with it.** The cost of producing one second of audio spans only 1.21×, and that last bit is run-to-run noise rather than a property of the voice: across runs the per-voice figures move by about that much and the ordering scrambles — af_nicole read 275 then 319, af_bella 298 then 359 — while the audio lengths repeat to the centisecond every time.

So no voice is harder to synthesise. A slow voice is slow because it says the same words for longer, and the generator is asked for more audio before the first chunk can go out. That puts the fix in the chunk plan, not in the model: TTFB is the voice's speaking rate times a roughly constant cost per second, so the lever is how much speech the opening words become.

Which the server currently gets wrong in one place. `s_per_word` is a single global EWMA, so the word budget for the next request is set by whatever voice ran last — measured here at 9 to 11 words for the same text. This sentence absorbed it, because the nearest clause boundary sat outside every budget in that range and finding 6 overshoots to the boundary rather than cutting to the count. On text whose boundaries fall inside the range it would not absorb it, and a slow voice would be handed a lead sized by a fast one. Making the estimate per-voice is what closes that, and is not done yet.

### 18. One call at a time was leaving a third of the machine unused

Finding 5 found that the operators which dominate the floor — `Sin`, `ConvTranspose`, `STFT` — sit under neither the bandwidth roof nor the compute roof. That is a third of the model's time in which the machine is waiting rather than working, and finding 3 showed you cannot thread one inference into it: past 8 threads the curve goes flat, then negative. The remaining way to use waiting time is to put a second piece of work in it, which means running more than one model call at once. The server was not doing that, and the reason was an assumption rather than a measurement.

**What I tested:** the same server twice, one flag apart, admission control off so nothing is turned away and the queue has to absorb everything. Four clients, three requests each.

| 4 clients, admission control off | one call at a time | four calls at once |
|---|---|---|
| wait for the model slot, p50 | 14760 ms | 0 ms |
| time to generate one chunk, p50 | 4850 ms | 13131 ms |
| worst buffer margin | −15.57 s | +4.59 s |
| streams whose buffer ran dry | 6 of 12 | 0 of 12 |
| TTFB p50 | 2640 ms | 3610 ms |
| TTFB p95 | 4532 ms | 3827 ms |
| audio delivered per second | 3.44× | 5.12× |

Both runs pass the drift check (1.02× and 1.07×), and at one client the two are the same server: 1315 ms against 1305 ms TTFB, 3.52× against 3.56×. Width costs nothing when there is nobody to share with.

**What I found:**

**1. The queue was the whole problem, and it was self-inflicted.** Serialising the model does not make the work smaller, it makes it wait — 14.8 seconds of waiting per chunk at four clients, against 4.9 seconds of actually generating. Half the clients ran their buffers dry, and the worst was fifteen seconds behind the listener. Letting four calls run at once deletes the queue outright, and every underrun with it.

**2. Each call gets slower, and that is the trade rather than a defect.** Generating one chunk goes from 4.9 s to 13.1 s, because four calls now share eight cores instead of owning them. TTFB p50 pays for it: 2640 ms becomes 3610 ms. What improves is the tail — p95 drops from 4532 ms to 3827 ms and the worst case from 5252 ms to 3858 ms, because the spread between the luckiest and unluckiest client was the queue's doing. A predictable 3.6 s is worth more to a listener than a median of 2.6 s with a fifteen-second hole behind it.

**3. So the per-call speed is the wrong number to size capacity with.** `realtime_speed` is audio over *model* time for one call, and widening the slot makes it fall — 2.85× down to 1.25× — at the exact moment the box starts carrying more listeners. `/health` now publishes `delivered_speed` beside it: audio over *wall* time while at least one call was running. The two agree at width 1 and separate above it, in opposite directions, and capacity derives from the second. Measuring the part instead of the whole is an easy way to build a server that throttles itself hardest when it is doing best.

**4. The same idle time is worth about 1.5×, whichever way you take it.** `experiments/sharding.py` measures the two shapes head to head, 25 seconds per leg on an otherwise idle box:

| layout | audio per second | extra memory | per-stream latency |
|---|---|---|---|
| 1 process × 8 threads | 3.43× | — | 1674 ms |
| 2 processes × 4 threads, pinned | 4.64× | +326 MB | 2508 ms |
| 4 processes × 2 threads, pinned | 5.30× | +978 MB | 4364 ms |
| 8 processes × 1 thread, pinned | 5.34× | +2282 MB | 8689 ms |

Sharding across processes tops out near 5.3×, and the eighth shard buys 0.04× for another 1.3 GB. One shared session with four threads calling into it reaches the same ceiling — 5.8× on the best of four runs, about 5.3× at the median — for no extra memory at all, because there is still one copy of the weights. On rented CPU that decides it: memory is the line item, cores are what you are already paying for.

**5. Pinning is worth doing and easy to do backwards.** The first sharded run came out *slower* than the baseline, 3.33× against 3.43×, because a mask of `0-3` looked like four cores and was two: `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` shows this box pairs SMT siblings adjacently, so half the machine sat out. Stepping the masks by 2 turned the same test into 4.61×. The idea was right the first time; the core numbering was not.

**6. One caveat on the in-process arm.** Its run-to-run spread is wide — one caller reads 3.94× to 4.40× across four runs, and the two-pinned-sessions leg swings from 3.64× to 5.69×. The ordering holds across repeats but the individual figures do not, so `sharding.py` takes a repeat count and prints every run rather than a best row. The server-level table above is the number worth defending; the microbenchmark is what explains it.

The default is `--model-width` at half the intra-op thread count, 4 here, because that is where it was measured. Other widths are extrapolation from one box.

---

## Architecture

```
  client ──WS──▶  FastAPI /tts
                    │  plan the chunks, against the buffer and the queue
                    │  (sizes are read from the live fit, not from constants)
                    │
                    │  ┌── admission control ── as many streams as the measured speed allows
                    │  │   full? wait, then refuse with a retry hint
                    │  │   order: most-refused first, then longest-waiting
                    │  └── refused ──▶ {"type":"busy"}, no audio sent
                    │
                    │  ├─ the model slot ── a measured number of calls at once
                    │  │  order: least audio in hand first
                    │  └─ asyncio.to_thread ── keeps the event loop alive
                    │
                    │  + 277 ms silence at sentence boundaries  ← free audio, deepens the buffer
                    │  per chunk: JSON metadata frame, then raw int16 PCM frame
  client ◀──────────┘  playback starts on chunk 0, while chunk 1 is still generating
```

Two gates, doing different jobs. Admission control decides *whether* you get served, and holds its slot for the whole request — releasing it between chunks would let a new stream in to compete with one already mid-sentence, which is the thing it exists to prevent. The model slot decides *when* each individual chunk runs, and how many run together. Admission control bounds the queue; the slot bounds how much of the machine is in use at once. Both are ordered by need rather than arrival, for the reasons in findings 15 and 16.

**Nothing in that diagram is a tuned constant.** Chunk size, how many streams fit, how long admission control waits, and the shortest chunk worth generating are all derived from one three-coefficient fit that the server refits from its own completed work. `/health` publishes the coefficients next to every number derived from them, so the arithmetic can be checked. Constants in the source are seeds for a cold start, and an operator who names a number keeps it.

Metadata as JSON, audio as a separate binary frame — **not** base64 inside the JSON, which is +33% on the one payload where bytes are latency, and int16 rather than float32, which halves it again.

---

## Status

| # | Milestone | State |
|---|---|---|
| **M0** | Scaffold + model speaks | ✅ done |
| **M1** | Streaming — chunking policy, then latency floor, then WebSocket | ✅ done |
| **M2** | Scheduler + benchmark harness (p50/p95/p99) | ✅ done — harness, admission control, queue policy, contention-aware planning |
| M3 | Docker + kind + HPA + Prometheus/Grafana | Docker done |
| M4 | Go gateway + Piper backend comparison | |

**Still to come:** result cache, Prometheus `/metrics`, a `TTSBackend` interface with Piper behind it, kind + HPA, Go gateway. The open measurement is a runtime bake-off — ONNX Runtime against PyTorch on chunk-0-sized inputs, where the published real-time factors differ enough (0.72 against 0.49) to move the term that dominates TTFB.

**M1 was deliberately ordered chunking → latency → transport.** The obvious order is to build the WebSocket first, but transport moves TTFB by single-digit milliseconds on localhost (measured: 33 ms of 6.4 s). Building it first would have meant re-running every benchmark after the real optimisation landed.

**M2 is a scheduler rather than a batcher because of finding 7.** The batcher half is impossible without graph surgery. The harness ran before the scheduler on purpose: a scheduler is worth building only if the queue owns the tail, and finding 10 shows it owns 88% of it, against a hard capacity line no queue policy can move. That bounded what the scheduler could honestly claim before a line of it was written — and findings 11 through 16 claim exactly that and no more: every stall removed, nobody starved, chunks planned against real contention, and not one extra second of audio per second.

The ceiling itself moved later, and not by scheduling. Finding 18 takes delivered audio from 3.44× to 5.12× at four clients by running several model calls at once, which is a claim about how much of the machine is in use rather than about who goes next. The scheduler bound stands: no ordering policy moves that line, and finding 7 still rules out the batcher that would move it further.

---

## The honest bar

Cartesia's Sonic-3 advertises **40–90 ms** time-to-first-audio; independent measurements land nearer **166–190 ms**. That is served GPU infrastructure, and this is one laptop CPU.

The claim here is not "I matched that." It's: *here is the floor on this machine, here is what it decomposes into operator by operator, here is what each lever bought, and here is the model that predicts what different hardware would do.* Being several times off a frontier system with a full account of where the time goes is more useful than a fast number with no decomposition.

And per finding 9, most of the remaining gap isn't a serving gap at all. A one-shot graph's floor is a full forward pass; an incremental model's is one step of a loop. Knowing which of those you're holding decides whether the next win comes from engineering or from a different model.

---

## Repo map — which script proves which claim

The root holds the product; `experiments/` holds the evidence. The split is deliberate: the files at the top are what you'd run or deploy, and the fifteen below them are one-shot measurements whose output is a number in this README.

**The server**

| file | what it is |
|---|---|
| `server.py` | The server, admission control, the model slot and the cost model. Also findings 6, 8, 11, 12, 13, 15, 16, 17, 18 |
| `client.py` | Reference client — streams, plays, and reports what it experienced |
| `bench.py` | Findings 10, 14, 15, 18 — percentiles and saturation, admission control on/off comparison, the per-client fairness check, the model-width matched pair |
| `demo/` | The console at `/demo` — the race between audio received and audio played, with a runtime capacity toggle |

**The evidence** — `experiments/`

| file | what it establishes |
|---|---|
| `speak.py` | M0 baseline — cold start 909 ms, synthesis 1785 ms, RTF 0.41 |
| `stream.py` | Finding 1 — `create_stream()` vs `create()` vs our chunking |
| `floor.py` | Finding 2 — the fixed/variable fit |
| `threads.py` | Finding 3 — the thread sweep |
| `isolate.py` | Finding 3 — attributing it to `intra_op` alone |
| `profile_ops.py` | Finding 2 — per-operator fixed/scaling decomposition |
| `quant.py` | Finding 4 — int8, including the audio-quality check |
| `roofline.py` | Finding 5 — machine ceilings and per-operator placement |
| `kernels.py` | Finding 5 — numpy head-to-head proving the kernel loss |
| `overhead.py` | Finding 8 — cost of the thread hop |
| `leadsweep.py` | Finding 6 — the first-chunk sweep that set the floor at 4 words |
| `headline.py` | The table at the top — both rows, one run, one methodology |
| `granularity.py` | Finding 9 — chunk-size sweep, citation-form cost, and the graph signature |
| `voice_ttfb.py` | Finding 17 — seven voices, one stream at a time, speaking rate against cost per second |
| `sharding.py` | Finding 18 — process sharding against in-process concurrency, output per second against memory |

Run them from anywhere in the checkout — `experiments/_root.py` anchors the working directory to the repo root, so the relative paths inside each script stay correct and `import server` still resolves. The server itself takes the other route and reads `HEADSTART_MODEL` from the environment, because unlike these it does have to run somewhere else.

Every number in this README came from one of these, or from `/health` on a running server, on the machine described at the top. Re-running them is the point.

---

## Stack

Python 3.10 · ONNX Runtime (CPU) · kokoro-onnx · FastAPI + WebSockets · numpy
Planned: Prometheus + Grafana · kind · Go gateway
