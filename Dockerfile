# headstart: one `docker run` to a server that answers on /health.
#
# The point of this file is reproducibility, not deployment. Every latency
# number in the README is a property of a specific runtime build as much as of
# a specific machine (block 2 found kernel quality to be ~16% of the floor), so
# a reader who cannot pin the runtime cannot check the claims. requirements.txt
# pins the Python packages; this pins the interpreter and the OS libraries
# under them.
#
#     docker build -t headstart .
#     docker run --rm -p 8000:8000 headstart
#     curl localhost:8000/health
#
# The image carries the 311 MB of model weights. That makes it large (~1.6 GB)
# and it is the deliberate trade: the alternative is a small image plus a
# download step, which is exactly the setup friction this file exists to
# remove. To skip the bake and mount weights you already have:
#
#     docker build --build-arg FETCH_MODELS=0 -t headstart .
#     docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" headstart

# 3.10 to match the interpreter every measurement was taken on. slim rather
# than alpine: onnxruntime ships manylinux wheels built against glibc, and on
# musl pip would fall back to building from source or simply fail.
FROM python:3.10-slim AS base

# curl is needed by HEALTHCHECK below and by the weight fetch. Removed from the
# final image only if it were a security boundary; here it stays because being
# able to curl /health from inside the container is worth more than 3 MB.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source. Docker caches layers in order, so putting the
# 200 MB of wheels ahead of the code means editing server.py rebuilds in
# seconds instead of re-resolving pip every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Weights in their own layer for the same reason, and ahead of the source for
# the stronger version of it: 311 MB should be fetched once per image, not once
# per code change.
ARG FETCH_MODELS=1
ARG MODEL_BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
RUN mkdir -p /app/models \
 && if [ "$FETCH_MODELS" = "1" ]; then \
      curl -fsSL -o /app/models/kokoro-v1.0.onnx "$MODEL_BASE/kokoro-v1.0.onnx" \
   && curl -fsSL -o /app/models/voices-v1.0.bin  "$MODEL_BASE/voices-v1.0.bin" ; \
    fi

COPY server.py bench.py client.py ./
# The console is served from /demo on the same origin as the socket, so the
# image is the whole demo: run it, open the port, press play.
COPY demo/ ./demo/

# Not root. The server reads two files and writes nothing, so there is no
# reason for it to be able to write anything either.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV HEADSTART_MODEL=/app/models/kokoro-v1.0.onnx \
    HEADSTART_VOICES=/app/models/voices-v1.0.bin \
    HEADSTART_HOST=0.0.0.0 \
    HEADSTART_PORT=8000 \
    PYTHONUNBUFFERED=1

# INTRA_OP is deliberately not set here. 8 is the measured default and the
# right value is the host's *physical* core count -- under a CPU limit that is
# neither 8 nor what the container can autodetect, since os.cpu_count() reports
# the host rather than the quota. Set it explicitly when you constrain CPU:
#
#     docker run --cpus 4 -e HEADSTART_INTRA_OP=4 -p 8000:8000 headstart

EXPOSE 8000

# start-period covers model load: ~1.2 s to build the session plus ~0.4 s to
# warm it. Without it the first two probes fail on a container that is fine.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["python", "server.py"]
