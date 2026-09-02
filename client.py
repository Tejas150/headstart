"""M1, block 3 — the client. Plays audio while the rest is still being made.

This is the demo. Everything before it was a table of numbers; this is the
part where the claim becomes audible, and the part a reviewer runs first.

HOW PLAYBACK WORKS
    PCM is piped straight into `aplay` as it arrives. aplay starts playing the
    instant the first bytes hit its stdin and keeps consuming as more arrive,
    so playback begins on chunk 0 while the server is still generating chunk 1.
    No audio library, no PortAudio, no device setup -- which also means the
    demo runs on a machine where `pip install sounddevice` would fail.

    The failure mode this is watching for: if generation ever falls behind
    playback, aplay drains its pipe and the listener hears a gap. That is what
    `lead` measures -- seconds of audio in hand minus seconds already played.
    While lead stays positive there is no gap. It going negative is the single
    thing that breaks a streaming TTS product, so it is printed per chunk
    rather than averaged away.

WHAT IT MEASURES, AND WHY THESE COLUMNS
    ttfb        client-side time to first audio *byte in hand*. The honest
                number: it includes connect, request, generation and transport,
                because that is what the listener actually waits through.
    queue/gen   from the server -- waiting for the model slot vs running the
                model. Separating them is what makes M2's batcher provable
                instead of asserted.
    transport   total client time minus total server time. On localhost this
                should be small; the point of printing it is that "should be"
                is not a measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time

import websockets

M0_SENTENCE = (
    "Headstart streams the first audio chunk before the rest of the clip is generated."
)
PARAGRAPH = (
    "Headstart streams the first audio chunk before the rest of the clip is generated. "
    "It is a way to reduce latency and make the model feel more responsive. "
    "The first chunk is generated in parallel with the rest of the clip, so it can be "
    "played back immediately while the rest of the audio is still being generated."
)


def player(sample_rate: int) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(
            ["aplay", "-q", "-t", "raw", "-f", "S16_LE",
             "-r", str(sample_rate), "-c", "1", "-"],
            stdin=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("  (aplay not found — measuring only, no playback)")
        return None


async def run(url: str, text: str, lead_words: int | None,
              play: bool, out: str | None) -> None:
    print(f"\n{'=' * 74}")
    print(f"{text[:68]}{'...' if len(text) > 68 else ''}")
    print(f"lead_words={lead_words or 'off (sentence boundaries only)'}\n")

    t0 = time.perf_counter()
    async with websockets.connect(url, max_size=None) as ws:
        connect_ms = (time.perf_counter() - t0) * 1000
        await ws.send(json.dumps({"text": text, "lead_words": lead_words}))

        header = json.loads(await ws.recv())
        sink = player(header["sample_rate"]) if play else None
        collected = bytearray()

        print(f"  {'#':>2} {'ttfb/arr':>10} {'queue':>8} {'gen':>9} "
              f"{'audio':>7} {'lead':>7}  text")

        ttfb = None
        meta = None
        while True:
            frame = await ws.recv()
            if isinstance(frame, str):
                meta = json.loads(frame)
                if meta["type"] == "end":
                    break
                continue

            arrived = (time.perf_counter() - t0) * 1000
            if ttfb is None:
                ttfb = arrived
            if sink:
                sink.stdin.write(frame)
                sink.stdin.flush()
            collected += frame

            label = meta["text"]
            print(f"  {meta['index']:>2} {arrived:>7.0f} ms {meta['queue_ms']:>6.1f} ms "
                  f"{meta['gen_ms']:>6.0f} ms {meta['audio_s']:>6.2f}s "
                  f"{meta['lead_s']:>+6.2f}s  {label[:34]}")

        wall_ms = (time.perf_counter() - t0) * 1000

    if sink:
        sink.stdin.close()
        sink.wait()

    print(f"\n  connect .............. {connect_ms:>8.0f} ms")
    print(f"  TIME TO FIRST AUDIO .. {ttfb:>8.0f} ms   <- the number")
    print(f"  total ................ {wall_ms:>8.0f} ms")
    print(f"  audio produced ....... {meta['audio_s']:>8.2f} s")
    print(f"  RTF .................. {meta['rtf']:>8.3f}      "
          f"({'faster' if meta['rtf'] < 1 else 'SLOWER'} than realtime)")
    print(f"  server total ......... {meta['total_ms']:>8.0f} ms")
    print(f"  transport + connect .. {wall_ms - meta['total_ms']:>8.0f} ms")
    print(f"  non-model overhead ... {meta['overhead_ms']:>8.0f} ms  (server side)")

    if out:
        import soundfile as sf
        import numpy as np
        pcm = np.frombuffer(bytes(collected), dtype="<i2").astype("float32") / 32767
        sf.write(out, pcm, header["sample_rate"])
        print(f"  wrote {out}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/tts")
    parser.add_argument("--text", default=None)
    parser.add_argument("--paragraph", action="store_true",
                        help="use the three-sentence paragraph")
    parser.add_argument("--lead-words", type=int, default=None,
                        help="split the first segment at a clause boundary "
                             "under N words, to get sound out sooner")
    parser.add_argument("--compare", action="store_true",
                        help="run with and without the lead split, back to back")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    text = args.text or (PARAGRAPH if args.paragraph else M0_SENTENCE)

    if args.compare:
        for lead in (None, 6):
            await run(args.url, text, lead, not args.no_play, None)
        print("\n  Same audio, same model, same machine. The only difference is "
              "where\n  the first cut is made — which is the entire chunking policy.")
    else:
        await run(args.url, text, args.lead_words, not args.no_play, args.out)


asyncio.run(main())
