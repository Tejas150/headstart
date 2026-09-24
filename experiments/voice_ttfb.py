#!/usr/bin/env python3
"""Why does TTFB move when only the voice changes?

Two arms, same text, one voice at a time and nothing else on the box.

  pinned   lead_words fixed at 5, so chunk 0 is the same five words for every
           voice. Whatever moves here is the model, not the planner.
  derived  lead_words left to the server, which is how real requests arrive.

What to read: audio_s on chunk 0 is how much speech those words became, and
gen_ms is what it cost to make it. If audio_s moves with the voice and gen_ms
tracks audio_s, the voice is not slower to generate -- it is saying the same
words for longer, and the generator is being asked for more audio.
"""
import asyncio, json, statistics, sys
import websockets

URL = "ws://localhost:8000/tts"
TEXT = ("A listener cannot tell the difference between a server that is under "
        "load and a product that is simply broken. The audio is the product.")
VOICES = ["af_sarah", "af_bella", "af_nicole", "am_michael", "am_adam",
          "bf_emma", "bm_george"]
REPEATS = 3


async def once(ws, voice, lead):
    req = {"text": TEXT, "voice": voice}
    if lead is not None:
        req["lead_words"] = lead
    await ws.send(json.dumps(req))
    head = json.loads(await ws.recv())
    if head["type"] == "busy":
        raise RuntimeError("refused -- run this on an idle server")
    first = None
    for _ in range(head["chunks"]):
        meta = json.loads(await ws.recv())
        await ws.recv()
        if first is None:
            first = meta
    end = json.loads(await ws.recv())
    return head, first, end


async def arm(lead, label):
    print(f"\n  {label}")
    print(f"  {'voice':<12}{'lead':>5}{'chunk 0 audio':>15}{'gen':>9}"
          f"{'TTFB':>9}{'s/word':>9}")
    print("  " + "-" * 59)
    rows = []
    async with websockets.connect(URL) as ws:
        await once(ws, "af_sarah", 5)                      # warm, discarded
        for v in VOICES:
            got = [await once(ws, v, lead) for _ in range(REPEATS)]
            audio = statistics.median(f["audio_s"] for _, f, _ in got)
            gen = statistics.median(f["gen_ms"] for _, f, _ in got)
            ttfb = statistics.median(e["ttfb_ms"] for _, _, e in got)
            words = statistics.median(h["lead_words"] for h, _, _ in got)
            rows.append((v, words, audio, gen, ttfb))
            print(f"  {v:<12}{words:>5.0f}{audio:>13.2f} s{gen:>7.0f} ms"
                  f"{ttfb:>7.0f} ms{audio / max(words, 1):>9.3f}")
    return rows


def spread(rows, i):
    vals = [r[i] for r in rows]
    return min(vals), max(vals), max(vals) / min(vals)


async def main():
    pinned = await arm(5, "lead_words pinned at 5 — same words for every voice")
    alo, ahi, ax = spread(pinned, 2)
    glo, ghi, gx = spread(pinned, 3)
    tlo, thi, tx = spread(pinned, 4)
    print(f"\n    chunk 0 audio  {alo:.2f} s to {ahi:.2f} s   {ax:.2f}x")
    print(f"    gen            {glo:.0f} ms to {ghi:.0f} ms   {gx:.2f}x")
    print(f"    TTFB           {tlo:.0f} ms to {thi:.0f} ms   {tx:.2f}x")

    # Cost per second of audio produced, which is the number that should NOT
    # move if the voice is only changing how long the speech is.
    print(f"\n    cost per second of audio produced:")
    for v, w, a, g, t in pinned:
        print(f"      {v:<12}{g / a:>8.0f} ms per audio-second")
    per_s = [g / a for _, _, a, g, _ in pinned]
    print(f"      spread {min(per_s):.0f} to {max(per_s):.0f} ms"
          f"   {max(per_s) / min(per_s):.2f}x")

    derived = await arm(None, "lead_words left to the server — how real requests arrive")
    tlo, thi, tx = spread(derived, 4)
    print(f"\n    TTFB           {tlo:.0f} ms to {thi:.0f} ms   {tx:.2f}x")


asyncio.run(main())
