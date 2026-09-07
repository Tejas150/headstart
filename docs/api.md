# headstart — API

Two endpoints: `GET /health` and a WebSocket at `/tts`. Base URL is `http://localhost:8000` / `ws://localhost:8000` unless you moved it.

Audio is **24000 Hz, mono, signed 16-bit little-endian, headerless**. No container, no WAV header — if you want a file you write the header yourself. That format is announced in the `start` frame rather than assumed, so a client can check rather than hardcode.

---

## `GET /health`

Answers as soon as the model is loaded and warmed, which takes one to two seconds from process start — roughly 1.2 s to build the session and 0.4 s to warm the graph, measured. Before that the port is not listening at all. It reports live door counters, so it doubles as the thing you look at when you want to know whether the server is turning people away.

```json
{
  "ok": true,
  "sample_rate": 24000,
  "intra_op_threads": 8,
  "inter_op_threads": 1,
  "max_inflight": 3,
  "door_wait_s": 5.2,
  "in_flight": 2,
  "admitted": 41,
  "refused": 7
}
```

`ok` is false until the model is in memory. `in_flight` is how many streams are generating right now; `admitted` and `refused` are cumulative since start. `refused / (admitted + refused)` is the number worth graphing — see the note at the bottom on why it beats CPU%.

---

## `WS /tts`

Open the socket once and send as many requests as you like down it. The socket is not one request; it is a session, and the server keeps a small amount of state per connection (see *Refusals are per connection* below).

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
| `text` | yes | — | What to say. Split into chunks on sentence boundaries. |
| `voice` | no | `af_sarah` | Any voice in `voices-v1.0.bin`. |
| `speed` | no | `1.0` | Playback rate passed to the model. |
| `lead_words` | no | `5` | Max words in the first chunk. `0` turns it off and cuts on sentences only. Omitted or `null` means "use the server default". |

`lead_words` is the one knob that changes what the listener experiences. The first chunk is cut short — at a comma, semicolon, colon, or conjunction if one fits, otherwise a hard word cut — so sound starts before the first sentence has finished generating. Only the first segment is cut this way; everything after it keeps whole-sentence prosody, because nobody is waiting on those. The default of 5 was swept, not guessed (finding 6 in the README).

The distinction between `null` and `0` matters: `null` means you have no opinion, `0` is an opinion.

### Response — served

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
{"type": "start", "sample_rate": 24000, "format": "s16le", "chunks": 3, "door_ms": 0.0, "in_flight": 1}
{"type": "chunk", "index": 0, "text": "The transcript is not the", "bytes": 77824, "audio_s": 1.621, "gap_s": 0.0,   "queue_ms": 0.0, "gen_ms": 850.3, "elapsed_ms": 850.6,  "lead_s": 1.621}
{"type": "chunk", "index": 1, "text": "product.",                  "bytes": 51184, "audio_s": 1.066, "gap_s": 0.277, "queue_ms": 0.0, "gen_ms": 507.5, "elapsed_ms": 1363.4, "lead_s": 2.175}
{"type": "chunk", "index": 2, "text": "The audio is.",             "bytes": 48128, "audio_s": 1.003, "gap_s": 0.0,   "queue_ms": 0.0, "gen_ms": 624.5, "elapsed_ms": 1991.1, "lead_s": 2.55}
{"type": "end", "ttfb_ms": 850.6, "total_ms": 1994.9, "door_ms": 0.0, "audio_s": 3.69, "rtf": 0.541, "overhead_ms": 12.6}
```

Three things in that trace are worth pointing at. Two sentences became **three** chunks, because `lead_words` cut the first one short — that is why sound starts at 851 ms instead of at 1358 ms. `gap_s` is 0.277 on chunk 1 and zero on the others, because chunk 1 is the one that ends at a real sentence boundary. And `lead_s` climbs across the request, 1.6 → 2.2 → 2.6, which is the buffer filling faster than it drains: the listener is never going to hear a hole.

Metadata is JSON and audio is raw binary in **two separate frames** rather than base64 inside one. Base64 is +33% bytes on the one payload where bytes are latency, and int16 instead of float32 halves it again.

**`start`**

```json
{"type": "start", "sample_rate": 24000, "format": "s16le", "chunks": 3, "door_ms": 0.0, "in_flight": 1}
```

`chunks` is how many `chunk`+binary pairs to expect. `door_ms` is time spent waiting for admission, which is separate from the two costs reported per chunk.

**`chunk`** — one per chunk, each followed by its binary frame.

| field | meaning |
|---|---|
| `bytes` | Length of the binary frame that follows. |
| `audio_s` | Seconds of audio in it, including any appended gap. |
| `gap_s` | Silence appended after this chunk — see *The gap* below. |
| `queue_ms` | Waiting for the model slot, once already admitted. |
| `gen_ms` | Inside the model. |
| `elapsed_ms` | Since the request arrived. |
| `lead_s` | **Audio handed over, minus audio the listener has already played.** |

`queue_ms` and `gen_ms` are reported separately on purpose. Under one client `queue_ms` is near zero; under load it is the whole story. A latency number that adds them together tells you the server was slow without telling you whether the model or the queue was the reason.

`lead_s` is the number this project exists to keep positive. Positive means the listener never hears a hole. If it goes negative, playback has caught up with generation and the audio has already broken up — no other field in the response says so.

**`end`** — `rtf` is total time ÷ audio produced; below 1.0 is faster than realtime. `overhead_ms` is everything that was not the model: transport, framing, framework. In the trace above it is 12.6 ms of a 1995 ms request, which is the measured answer to "should I worry about the WebSocket?"

### Response — refused

```json
{
  "type": "busy", "in_flight": 3, "capacity": 3, "retry_after_s": 5.2,
  "waited_ms": 5201.3, "refusals": 2,
  "detail": "at capacity (3 streams); no slot freed in 5.2s"
}
```

**A `busy` arrives instead of `start`, never after it.** Nothing was generated and no audio was sent. This is the point of the door: a request is either served cleanly or refused before the listener has been told anything is coming. The server will not accept a stream it cannot finish, because a wait reads as loading but a stutter reads as a broken product.

The socket stays open. Send another request when you are ready.

`retry_after_s` is a real hint, not a constant plucked from the air: it is the measured time for one of the three slots to come free. `refusals` is your consecutive-refusal count, which is also your queue priority — see below.

### Refusals are per connection

Each time you are refused, your priority goes up by one. A freed slot goes to the highest-priority waiter, with arrival order only breaking ties. Being admitted resets it to zero.

This exists because plain first-come-first-served starves people here. A refused caller backs off, and while backing off it is not in the queue at all — so it comes back behind the callers that were just served and re-queued instantly. Measured at 8 clients against 3 slots, 2 of the 8 were refused every single time. With ageing, none were. Findings 11 and 12.

**The practical consequence for you: if you get a `busy`, back off and retry on the same socket.** The counter lives on the connection, so reconnecting throws away the priority you earned by waiting and puts you at the back again.

### The gap

`gap_s` is silence the server appends after a chunk. Chunking has a cost that no latency metric shows: the model only renders the pause between two sentences when it can see the boundary, so generating sentences separately silently deletes it. Measured at 277 ms per boundary. It is put back as silence, which costs no model time and *raises* `lead_s` rather than spending it.

A clause split inside a sentence gets `gap_s: 0.0` — there was no pause there to restore, and inserting one would be audibly wrong.

---

## Errors and edges

- **Malformed JSON, or `text` missing** — the handler raises and the connection drops without a close frame, so a client sees an abnormal closure rather than an error message. Worth naming as a real gap rather than a design choice: it should send a typed error, and it does not.
- **Empty or whitespace-only `text`** — produces zero chunks, so you get `start` with `"chunks": 0` and then `end` with `"rtf": null`. No audio frames, and the socket stays open for the next request.
- **Disconnect mid-generation** — the slot is released. Capacity does not leak.
- **Requests are serialised per connection.** The server reads one request, serves it, and only then reads the next. Pipelining several requests down one socket does not make them concurrent; open more sockets for that, up to the door.

---

## Configuration

Command line flags, all of which have measured defaults:

```
--host          127.0.0.1     loopback by default; exposing the model is an explicit act
--port          8000
--max-inflight  3             streams served at once; 0 turns the door off entirely
--door-wait     5.2           seconds waited at the door before refusal
```

Environment variables, which the container uses:

```
HEADSTART_MODEL     models/kokoro-v1.0.onnx
HEADSTART_VOICES    models/voices-v1.0.bin
HEADSTART_HOST      127.0.0.1
HEADSTART_PORT      8000
HEADSTART_INTRA_OP  8    physical cores, not logical — see below
HEADSTART_INTER_OP  1
```

`--max-inflight 0` is how the before/after arms of the benchmark are produced. The two arms then differ by one flag rather than by a code version, so "was it the same build?" stops being a question about the result.

`HEADSTART_INTRA_OP` should be the host's **physical** core count. It is not autodetected, and that is deliberate: inside a container `os.cpu_count()` reports the host rather than the CPU quota, so autodetection would confidently pick the wrong number. If you constrain CPU, set it:

```
docker run --cpus 4 -e HEADSTART_INTRA_OP=4 -p 8000:8000 headstart
```

---

## A note on monitoring

The obvious autoscaling signal is CPU utilization, and here it does not work.

`intra_op_num_threads=8` hands every operator all eight physical cores, and the model slot runs one call at a time — so a single client already uses everything the server will ever use, and extra clients queue rather than consuming more. Measured on this box: **one client draws 806%, 764%, 771% of a core; six clients draw 673%, 787%, 775%.** Those are the same number. CPU% reads identically at one healthy stream and at six where three are being turned away, so it cannot tell you which you have.

**`refused / (admitted + refused)` from `/health` is the signal that does.** It tracks demand against a measured capacity of ~3.3 concurrent realtime streams, and it moves for exactly one reason.

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

`client.py` in the repo root is the fuller version — it plays as it receives and reports what it actually experienced, including whether the buffer ever ran dry.
