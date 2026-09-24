# headstart API

Everything a client needs is on two endpoints: `GET /health` and a WebSocket at `/tts`. There is also `POST /admission` for moving capacity on a running server, and the browser console at `/demo`. Base URL is `http://localhost:8000` or `ws://localhost:8000` unless you moved it.

Audio is 24000 Hz, mono, signed 16-bit little-endian, headerless. No container and no WAV header, so if you want a file you write the header yourself. The `start` frame announces the format on every request, so a client can check it instead of hardcoding it.

---

## `GET /health`

Answers as soon as the model is loaded and warmed, which takes one to two seconds from process start: roughly 1.2 s to build the session and 0.4 s to warm the graph, measured. Before that the port is not listening at all.

```json
{
  "ok": true,
  "sample_rate": 24000,
  "intra_op_threads": 8,
  "inter_op_threads": 1,
  "fit_fixed_s": 0.028,
  "fit_slope": 0.824,
  "s_per_word": 0.275,
  "fit_samples": 64,
  "queue_s": 0.0,
  "realtime_speed": 1.25,
  "model_width": 4,
  "delivered_speed": 2.91,
  "busy_s": 60.0,
  "max_inflight": 2,
  "derived_capacity": 2,
  "capacity_pinned": false,
  "admission_wait_s": 5.76,
  "hold_s": 11.51,
  "hold_samples": 47,
  "lead_words": 4,
  "min_chunk_s": 0.16,
  "sustainable": true,
  "ttfb_target_s": 1.0,
  "queue_aware": true,
  "slot_preempted": 0,
  "in_flight": 0,
  "admitted": 47,
  "refused": 1
}
```

`ok` is false until the model is in memory.

The fit. `fit_fixed_s`, `fit_slope` and `s_per_word` are the server's live model of its own cost: a chunk of `a` seconds of audio takes `fit_fixed_s + fit_slope × a` seconds to generate, and a word is worth about `s_per_word` seconds of speech. They are refitted from the last 64 completed chunks, so they drift with the text, the voice and whatever else the box is doing; `fit_samples` says how many chunks are in the window. `realtime_speed` is how much faster than realtime one call generates, in seconds of audio produced per second of model time, so 1.25 means a single chunk comes out 1.25x faster than it plays. It is the same idea as the `rtf` on the `end` frame turned upside down: `rtf` is wall time over audio for one request, `realtime_speed` is audio over model time averaged over the last 64 chunks. `queue_s` is the rolling wait for the model slot, which is tracked separately and kept out of the fit, since a busy box would otherwise read as a slow one.

Two speeds, and they are not interchangeable. `model_width` is how many model calls are allowed to run at once. Above 1 the calls share the thread pool, so each one gets slower and `realtime_speed` falls, which is the opposite of what the box is doing. `delivered_speed` is the honest total: seconds of audio produced per second of wall time during which at least one call was running, over a rolling `busy_s` window of working time. The two agree at width 1 and separate above it, in opposite directions. It reads `null` until the window holds a few seconds of real work, since a ratio over a near-zero denominator is not a measurement. Finding 18 has the numbers behind the split.

What the fit decides. The next block is all derived from the three coefficients above rather than configured. `derived_capacity` is `delivered_speed` rounded down, falling back to `realtime_speed` before the busy window has filled: one listener consumes one second of speech per second of real time, so a server delivering 2.91x can carry two of them with margin. `lead_words` is how many words fit inside `ttfb_target_s`. `min_chunk_s` is the shortest chunk still worth generating once the queue wait is paid, and `sustainable` goes false when that floor rises above the ceiling, which is the server saying no chunk size works at this load. `max_inflight` is the capacity actually in force, which equals `derived_capacity` unless somebody pinned it, and `capacity_pinned` says which. `admission_wait_s` is how long a request waits for admission before it is refused, computed from `hold_s`, the average time a request holds its slot, over the number of slots; `hold_samples` is that window's size. `queue_aware` reports whether the queue term is switched on.

The counters. `in_flight` is how many streams are generating right now; `admitted` and `refused` are cumulative since start. `slot_preempted` counts how often the model slot went to a stream that did not ask first (finding 16). For monitoring, `refused / (admitted + refused)` is the ratio worth graphing, and the note at the bottom explains why CPU% cannot do that job.

Every number the server derives is published beside the coefficients it was derived from, so the arithmetic can be rechecked instead of taken on trust.

---

## `WS /tts`

Open the socket once and send as many requests as you like down it. The socket is a session rather than a single request, and the server keeps a small amount of state per connection (see "Refusals are per connection" below).

### Request

One JSON text frame:

```json
{
  "text": "The transcript is not the product. The audio is.",
  "voice": "af_sarah",
  "speed": 1.0,
  "lead_words": 5
}
```

| field | required | default | meaning |
|---|---|---|---|
| `text` | yes | | What to say. Split into chunks on sentence boundaries. |
| `voice` | no | `af_sarah` | Any voice in `voices-v1.0.bin`. |
| `speed` | no | `1.0` | Playback rate passed to the model. |
| `lead_words` | no | derived | Max words in the first chunk. `0` turns it off and cuts on sentences only. Omitted or `null` lets the server size it. |

`lead_words` is the one knob that changes what the listener experiences. The first chunk is cut short, at a comma, semicolon, colon, or conjunction if one fits and otherwise a hard word cut, so sound starts before the first sentence has finished generating. Only the first segment is cut this way. Everything after it keeps whole-sentence prosody, because nobody is waiting on those.

Leave the field out and the server sizes chunk 0 itself: it asks how many words of speech fit inside its TTFB target at the cost it is currently measuring, which lands between 4 and 7 on this box as it warms up and cools. The lower clamp of 4 comes from the sweep in finding 6, where the median kept falling but the spread blew up. So the fit picks the number and the measured variance sets the floor under it. Whatever it picked comes back on the `start` frame. The request above pins it at 5 so the trace below is reproducible.

`null` and `0` are different answers. `null` means you have no opinion. `0` means you would rather keep the buffer than start early, which is the trade in finding 12, worth sending for a long narration nobody is waiting on.

### Response, served

```
start   (JSON)
chunk 0 (JSON)  →  audio 0 (binary)
chunk 1 (JSON)  →  audio 1 (binary)
...
end     (JSON)
```

Every `chunk` frame is immediately followed by exactly one binary frame carrying that chunk's audio. Read them as a pair. The metadata arrives first so a client knows how many bytes are coming and what they represent before it has them.

Here is a real trace of the request above, binary frames elided:

```
{"type": "start", "sample_rate": 24000, "format": "s16le", "chunks": 3, "lead_words": 5, "admission_wait_ms": 0.0, "in_flight": 1}
{"type": "chunk", "index": 0, "text": "The transcript is not the", "bytes": 77824, "audio_s": 1.621, "gap_s": 0.0,   "queue_ms": 0.0, "gen_ms": 850.3, "elapsed_ms": 850.6,  "lead_s": 1.621}
{"type": "chunk", "index": 1, "text": "product.",                  "bytes": 51184, "audio_s": 1.066, "gap_s": 0.277, "queue_ms": 0.0, "gen_ms": 507.5, "elapsed_ms": 1363.4, "lead_s": 2.175}
{"type": "chunk", "index": 2, "text": "The audio is.",             "bytes": 48128, "audio_s": 1.003, "gap_s": 0.0,   "queue_ms": 0.0, "gen_ms": 624.5, "elapsed_ms": 1991.1, "lead_s": 2.55}
{"type": "end", "ttfb_ms": 850.6, "total_ms": 1994.9, "admission_wait_ms": 0.0, "audio_s": 3.69, "rtf": 0.541, "overhead_ms": 12.6}
```

Three things in that trace are worth pointing at. Two sentences became three chunks, because `lead_words` cut the first one short, which is why sound starts at 851 ms instead of at 1358 ms. `gap_s` is 0.277 on chunk 1 and zero on the others, because chunk 1 is the one that ends at a real sentence boundary. And `lead_s` climbs across the request, 1.6 then 2.2 then 2.6, which is the buffer filling faster than it drains: the listener is never going to hear it run dry.

Metadata is JSON and audio is raw binary in two separate frames rather than base64 inside one. Base64 is 33% more bytes on the one payload where bytes are latency, and int16 instead of float32 halves it again.

`start`:

```json
{"type": "start", "sample_rate": 24000, "format": "s16le", "chunks": 3, "lead_words": 5, "admission_wait_ms": 0.0, "in_flight": 1}
```

`chunks` is how many `chunk` plus binary pairs to expect. `lead_words` is what chunk 0 was actually sized to, either the value you sent or the one the server derived if you sent none. Read it here rather than from `/health`, which may have refitted since your request landed. `admission_wait_ms` is time spent waiting for admission, and it is kept separate from the two costs reported per chunk.

`chunk`, one per chunk, each followed by its binary frame:

| field | meaning |
|---|---|
| `bytes` | Length of the binary frame that follows. |
| `audio_s` | Seconds of audio in it, including any appended gap. |
| `gap_s` | Silence appended after this chunk. See "The gap" below. |
| `queue_ms` | Waiting for the model slot, once already admitted. |
| `gen_ms` | Inside the model. |
| `elapsed_ms` | Since the request arrived. |
| `lead_s` | Audio handed over, minus audio the listener has already played. |

`queue_ms` and `gen_ms` are reported separately on purpose. Under one client `queue_ms` is near zero; under load it is the whole story. A latency number that adds them together tells you the server was slow without telling you whether the model or the queue was the reason.

`lead_s` is the number this project exists to keep positive. Positive means the buffer never ran dry. If it goes negative, playback has caught up with generation and the audio has already broken up, and no other field in the response says so.

`end`: `rtf` is total time over audio produced, and below 1.0 is faster than realtime. It is close to the reciprocal of `realtime_speed` on `/health`, since 0.541 inverts to 1.85x, but it is not the same number. `rtf` is wall time for one request, so it carries the queue and admission wait with it, while `realtime_speed` counts model time only, averaged over the last 64 chunks. On an idle box they nearly coincide; under load `rtf` rises while `realtime_speed` barely moves. `overhead_ms` is everything that was not the model: transport, framing, framework. In the trace above it is 12.6 ms of a 1995 ms request, which is the measured answer to "should I worry about the WebSocket?"

### Response, refused

```json
{
  "type": "busy", "in_flight": 3, "capacity": 3, "retry_after_s": 5.2,
  "waited_ms": 5201.3, "refusals": 2,
  "detail": "at capacity (3 streams); no slot freed in 5.2s"
}
```

A `busy` arrives instead of `start`, never after it. Nothing was generated and no audio was sent. This is the point of admission control: a request is either served cleanly or refused before the listener has been told anything is coming. The server will not accept a stream it cannot finish, because a wait reads as loading but a stutter reads as a broken product.

The socket stays open. Send another request when you are ready.

`retry_after_s` is measured rather than constant: it is the average time a request holds a slot, divided by the number of slots, so it tracks how long the server currently takes to free one. `refusals` is your consecutive-refusal count, which is also your queue priority. See below.

### Refusals are per connection

Each time you are refused, your priority goes up by one. A freed slot goes to the highest-priority waiter, with arrival order only breaking ties. Being admitted resets it to zero.

This exists because plain first-come-first-served starves people here. A refused caller backs off, and while backing off it is not in the queue at all, so it comes back behind the callers that were just served and re-queued instantly. Measured at 8 clients against 3 slots, 2 of the 8 were refused every single time. With ageing, none were. Finding 15 in `docs/findings.md` has the per-client numbers.

The practical consequence for you: if you get a `busy`, back off and retry on the same socket. The counter lives on the connection, so reconnecting throws away the priority you earned by waiting and puts you at the back again.

### The gap

`gap_s` is silence the server appends after a chunk. Chunking has a cost that no latency metric shows: the model only renders the pause between two sentences when it can see the boundary, so generating sentences separately silently deletes it. Measured at 277 ms per boundary. It is put back as silence, which costs no model time and raises `lead_s` rather than spending it.

A clause split inside a sentence gets `gap_s: 0.0`. There was no pause there to restore, and inserting one would be audibly wrong.

---

## Errors and edges

Malformed JSON, or `text` missing: the handler raises and the connection drops without a close frame, so a client sees an abnormal closure rather than an error message. Worth naming as a real gap rather than a design choice. It should send a typed error, and it does not.

Empty or whitespace-only `text`: produces zero chunks, so you get `start` with `"chunks": 0` and then `end` with `"rtf": null`. No audio frames, and the socket stays open for the next request.

Disconnect mid-generation: the slot is released. Capacity does not leak.

Requests are serialised per connection. The server reads one request, serves it, and only then reads the next. Pipelining several requests down one socket does not make them concurrent; open more sockets for that, up to the admission limit.

---

## Configuration

Command line flags, all of which have measured defaults:

```
--host          127.0.0.1     loopback by default; exposing the model is an explicit act
--port          8000
--max-inflight    (measured)  streams served at once; 0 turns admission control off entirely
--admission-wait  (measured)  seconds a request waits for admission before it is refused
--model-width     cores / 2   model calls allowed to run at once; 1 serialises them
```

`--model-width` is the throughput knob. At 1 the model runs one call at a time, which is the least surprising behaviour and the slowest; above 1 each call is slower and the box delivers more audio, for the reason in finding 18. The default is half the intra-op thread count, which is 4 here.

Omit `--max-inflight` and capacity tracks the measured delivered speed, which is `derived_capacity` in `/health` and reads 2 on this box. Naming a number pins it for the life of the process. `POST /admission?capacity=N` moves it on a running server and `POST /admission?capacity=-1` hands it back to the measurement.

Environment variables, which the container uses:

```
HEADSTART_MODEL     models/kokoro-v1.0.onnx
HEADSTART_VOICES    models/voices-v1.0.bin
HEADSTART_HOST      127.0.0.1
HEADSTART_PORT      8000
HEADSTART_INTRA_OP  8    physical cores, not logical; see below
HEADSTART_INTER_OP  1
```

`--max-inflight 0` is how the before/after arms of the benchmark are produced. The two arms then differ by one flag rather than by a code version, so "was it the same build?" stops being a question about the result.

`HEADSTART_INTRA_OP` should be the host's physical core count. It is not autodetected, and that is deliberate: inside a container `os.cpu_count()` reports the host rather than the CPU quota, so autodetection would confidently pick the wrong number. If you constrain CPU, set it:

```
docker run --cpus 4 -e HEADSTART_INTRA_OP=4 -p 8000:8000 headstart
```

---

## A note on monitoring

The obvious autoscaling signal is CPU utilization, and here it does not work.

`intra_op_num_threads=8` hands every operator all eight physical cores, and a stream only ever has one chunk in the model at a time, so a single client already uses everything the server will ever use. Widening the model slot does not change that: concurrent calls share the same eight cores rather than finding more. Measured on this box, one client draws 806%, 764% and 771% of a core; six clients draw 673%, 787% and 775%. Those are the same number. CPU% reads identically at one healthy stream and at six where three are being turned away, so it cannot tell you which you have.

`refused / (admitted + refused)` from `/health` is the signal that does. It tracks demand against the capacity the server has measured for itself, which is `delivered_speed` and `derived_capacity` in the same payload, and it moves for exactly one reason.

---

## Minimal client

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/tts") as ws:
        await ws.send(json.dumps({"text": "Hello. This is a second sentence."}))
        header = json.loads(await ws.recv())
        if header["type"] == "busy":
            print("refused, retry in", header["retry_after_s"], "s")
            return
        pcm = b""
        for _ in range(header["chunks"]):
            meta = json.loads(await ws.recv())     # chunk metadata
            pcm += await ws.recv()                 # the audio it describes
            print(meta["index"], meta["elapsed_ms"], "ms  lead", meta["lead_s"], "s")
        print(json.loads(await ws.recv()))         # end
        # pcm is now 24 kHz mono s16le, ready for a WAV header or a sound device

asyncio.run(main())
```

`client.py` in the repo root is the fuller version. It plays as it receives and reports what it actually experienced, including whether the buffer ever ran dry.
