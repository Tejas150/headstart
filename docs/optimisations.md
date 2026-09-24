# Optimisations, what was tried, in order

Every line here is a change that was measured before and after on the same machine (Ryzen 7 4800H, 8C/16T, CPU only). The finding numbers point at the section of `findings.md` that holds the table.

## M0, the measuring stick

Nothing was optimised here. `experiments/speak.py` established the baseline: 909 ms cold start, 1785 ms to synthesise, real-time factor 0.41. Everything below is measured against a version of this number.

## M1, getting to first sound

Thread count. ONNX Runtime defaults to one thread per logical core, which is 16 here and the wrong answer: SMT siblings share a load/store path, so the second thread on a core contends instead of helping. Setting `intra_op=8` took a full sentence from 1914 ms to 1541 ms and a short chunk from 781 ms to 583 ms. The other knob, `inter_op=1`, moved nothing measurable on either text. It ships anyway because it makes the thread budget explicit, but it is not carrying any of the number. Finding 3.

Chunking at sentence boundaries. The single largest change. Instead of generating 17 s of audio and sending it when finished, generate a sentence, send it, generate the next. Time to first audio went from 5658 ms to 1940 ms, and the total clip takes 1.27x longer because every chunk re-pays a fixed per-call cost. The number at the top of the README.

Splitting the first chunk at a clause boundary. A sentence-sized chunk 0 is still a long forward pass, 1624 ms on the sweep text against a 300 ms floor. Cutting the first segment early takes it to 876 ms at five words. Below five the median keeps falling but the spread explodes: four and three words each threw a 2 s outlier in three runs. So the server derives the word count from its own cost fit and clamps it at four, where the tail starts. Finding 6.

Re-inserting the sentence pause. Not a speed change. The model renders the gap between sentences only when it can see the boundary, so chunking silently deletes it: 555 ms across two boundaries, 277 ms each. Putting it back as explicit silence costs zero model time and, because it is audio handed over for free, deepens the listener's buffer rather than spending it. Finding 6.

Sizing chunks against the buffer. Chunk length is chosen from how much audio the listener already has in hand, not from a constant. Findings 12 and 13.

Building the transport last. Connect, WebSocket and framing together are 33 ms of a 6.4 s run. Building it first would have meant re-running every benchmark after the real work landed.

## M2, serving more than one person

Admission control. The server measures how fast it delivers audio, rounds down, and admits that many streams. At eight clients, stalls went from 48 of 48 to 0, p50 from 10707 ms to 6102 ms and p95 from 17828 ms to 7457 ms. It costs no capacity, 3.23x against 3.27x, because it reallocates who waits rather than creating or destroying throughput. Finding 14.

Ageing refusals. Plain first-come-first-served starved two of eight clients completely: a refused caller backs off, and while backing off it is not in the queue, so it loses its place to whoever was just served. Each refusal now raises that caller's priority. Everyone gets served at least once and the gap between best- and worst-served narrows from 4 to 3. Finding 15.

Ordering the model slot by least audio in hand. Whoever is closest to running dry goes next, rather than whoever asked first. Finding 16.

Running four model calls at once. The server originally ran one, on the assumption that eight threads already had the machine busy. Finding 3 says otherwise: eight cores buy 2.46x over one, about 31% efficiency, and finding 5 puts three of the dominant operators under neither the compute roof nor the bandwidth roof. Widening the slot to four takes delivered audio from 3.44x to 5.12x at four clients. The queue goes from 14760 ms per chunk to zero, and underruns from 6 of 12 streams to none. Each individual call gets slower, 4850 ms to 13131 ms per chunk and TTFB p50 2640 ms to 3610 ms, but the tail improves, p95 from 4532 ms to 3827 ms, because the spread was the queue's doing. Finding 18.

Pinning to real cores rather than core numbers. The first sharded run came out slower than the baseline because a mask of `0-3` looked like four cores and was two; this box numbers SMT siblings adjacently. Stepping the masks by two turned 3.33x into 4.61x. Finding 18.

## Tried, and it did not work

int8 quantization. Predicted about 1.27x from the matmul-family share. The quantized model would not load: ONNX Runtime's `ConvInteger` CPU kernel is 2-D only and all 94 convolutions in this graph are 1-D, so the operator holding 86 of the 93 quantizable milliseconds cannot be quantized on this runtime at all. Quantizing `MatMul` alone gave 1.00x and shifted phoneme durations by 21 to 64 ms. No speed, different audio. This CPU is Zen 2 with no VNNI, so int8 was only ever going to be a memory-traffic win. Finding 4.

The library's own `create_stream()`. It chunks for memory, not for latency, and does not get the first sample out sooner. Finding 1.

Process sharding instead of one shared session. Measured head to head against the in-process arm: both recover about the same 1.5x, and the shared session wins on memory because it holds one copy of a 326 MB model instead of several. The idle time is worth the same either way; the tiebreak is resident set size. Finding 18.

## Not done, and the reason

Dynamic batching. Structurally closed rather than deferred. The exported graph's batch dimension is a literal 1 and the output has no batch dimension at all, so there is nothing to stack requests into. This does not change on a GPU. Finding 7.

Replacing the STFT kernel. The best unclaimed win on the board and the one most likely to break something. Reimplemented in numpy on the same shapes, `STFT` runs 43x faster, 50.9 ms against 1.2 ms, and about 110 ms of a 686 ms forward pass, 16%, is recoverable kernel quality rather than a busy machine. Taking it means splitting the graph and running part of it outside ONNX Runtime, either as a custom operator or as a pre/post-process hop. That is a change that can alter the audio, and the payoff is capped at 16%. Parked with the measurement attached, which is the part worth having. Finding 5.

Incremental or token-by-token generation. This is a one-shot graph, with no state tensors in the signature, so its floor is a full forward pass rather than one step of a loop. Getting past that needs a different model, not different serving. Worth knowing because it decides whether the next win comes from engineering at all. Finding 9.

GPU, and CUDA graphs on top of it. There is no GPU on this machine, which is the whole premise. The scaling model in finding 5 predicts what a different box does and is piecewise on purpose: 16% of the time does not move at all, so "twice the hardware, twice the speed" is wrong here and the formula says precisely where.

Probing above the measured capacity. Capacity derives from delivered speed, and delivered speed is only ever sampled under the admission limit it already set, so the loop reads 2.91x forever while the door-off run does 5.12x. Climbing out means admitting one more than the reading says and keeping it only if the buffers hold. That needs a rollback path and a way to run the experiment without a live listener paying for it, so it is written down as open rather than built.

Chunk-0 priority in the model slot, and a rolling door-wait estimate. Both are small and both would need their own before/after run to be worth claiming. Neither has one yet.

Runtime bake-off, ONNX Runtime against PyTorch. Published real-time factors differ enough, 0.72 against 0.49, to move the term that dominates TTFB. Not yet run.

Result cache, Prometheus metrics, kind + HPA, Go gateway. Nothing technical is blocking these. They are M3/M4 and are sequenced after the serving work on purpose.
