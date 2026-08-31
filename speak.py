"""M0 — prove the model loads and speaks, and time both halves.

Cold start (loading 311 MB of weights off disk) and synthesis are
separate costs. They get separate numbers from day one, because
every metric in this project is a number in the README later.
"""

import time

import soundfile as sf
from kokoro_onnx import Kokoro

TEXT = "Headstart streams the first audio chunk before the rest of the clip is generated. It is a way to reduce latency and make the model feel more responsive. The first chunk is generated in parallel with the rest of the clip, so it can be played back immediately while the rest of the audio is still being generated."
VOICE = "af_sarah"

t0 = time.perf_counter()
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
cold_start = time.perf_counter() - t0

t1 = time.perf_counter()
samples, sample_rate = kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
synth = time.perf_counter() - t1

sf.write("out.wav", samples, sample_rate)

audio_seconds = len(samples) / sample_rate
print(f"cold start      {cold_start * 1000:8.0f} ms   (load weights, paid once per process)")
print(f"synthesis       {synth * 1000:8.0f} ms   (one full forward pass, no streaming yet)")
print(f"audio produced  {audio_seconds:8.2f} s")
print(f"RTF             {synth / audio_seconds:8.2f}     (<1.0 = faster than realtime)")
print("wrote out.wav")
