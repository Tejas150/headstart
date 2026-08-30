# headstart

A low-latency **streaming TTS inference server**. The listener starts hearing audio while the rest of the clip is still generating — the name is the architecture.

Built to answer one question with real numbers: *how fast can you serve a neural TTS model on a laptop CPU, and where exactly does the latency live?*

**No GPU.** Everything here runs on a Ryzen 7 4800H (8C/16T, DDR4-3200). Optimising inference on constrained hardware is the interesting problem; throwing a GPU at it isn't.

---

## Target architecture

```
                    ┌──────────────────────────────────────────┐
  client ──WS──────▶│  GATEWAY                                 │
         └─HTTP────▶│  · validate · admission control          │
                    │  · result cache · /metrics · /healthz    │
                    └────────────────┬─────────────────────────┘
                                     │ async request queue
                    ┌────────────────▼─────────────────────────┐
                    │  BATCHER (dynamic)                       │
                    │  · max_batch_size OR max_wait_ms         │
                    │  · the throughput-vs-latency knob        │
                    └────────────────┬─────────────────────────┘
                    ┌────────────────▼─────────────────────────┐
                    │  MODEL WORKERS (ONNX Runtime, CPU)       │
                    │  · Kokoro | Piper behind TTSBackend      │
                    │  · thread-pool tuned, emits chunk-by-chunk│
                    └────────────────┬─────────────────────────┘
                                     │ PCM chunks
  client ◀───────────────────────────┘  first chunk ASAP
```

## Status

| # | Milestone | State |
|---|---|---|
| **M0** | Scaffold + model speaks | ✅ **done** |
| M1 | Streaming over WebSocket | next |
| M2 | Dynamic batcher + benchmark harness (p50/p95/p99) | |
| M3 | Docker + kind + HPA + Prometheus/Grafana | |
| M4 | Go gateway + Piper backend comparison | |

## M0 numbers

Kokoro-82M, fp32 ONNX (311 MB), single request, no streaming, no batching:

| Metric | Value |
|---|---|
| Cold start (load weights into RAM) | **909 ms** |
| Synthesis, one full forward pass | **1785 ms** |
| Audio produced | 4.31 s |
| **RTF** (real-time factor) | **0.41** |

RTF below 1.0 means generation outruns playback — a necessary condition for streaming to be worth building, since the stream can't keep up otherwise. That's the whole point of measuring it first.

Note what these numbers *don't* include: time-to-first-audio-chunk. Right now it equals total synthesis time, because nothing is emitted until the clip is complete. Collapsing that gap is M1.

## Run it

```bash
virtualenv -p python3.10 .venv
.venv/bin/pip install kokoro-onnx soundfile

B=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
curl -L -o models/kokoro-v1.0.onnx "$B/kokoro-v1.0.onnx"
curl -L -o models/voices-v1.0.bin  "$B/voices-v1.0.bin"

.venv/bin/python speak.py   # writes out.wav
```

Model weights are gitignored — pull them with the commands above.

## Stack

Python 3.10 · ONNX Runtime (CPU) · kokoro-onnx · FastAPI + WebSockets · Prometheus + Grafana · Docker + kind · Go gateway (M4)
