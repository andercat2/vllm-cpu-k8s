# vLLM CPU inference image. Build from the repository root:
#   docker build -f docker/vllm/Dockerfile -t vllm-cpu-private:0.32.0 .

# trixie: the vLLM CPU wheel is manylinux_2_39 and needs glibc >= 2.39.
FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS builder

COPY docker/vllm/requirements.txt /build/requirements.txt

# --no-deps with --require-hashes: only what is in the lock file, byte for byte.
# The package indexes, including the one with CPU builds of torch, are named in
# the lock file: an --index-url there would override any given on this line.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps --require-hashes \
        -r /build/requirements.txt


FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# A compiler is a runtime dependency of the vLLM CPU backend, with no flag to turn
# it off: Triton builds its kernels with cc on first use, and torch.compile builds
# C++ kernels on startup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ libnuma1 libgomp1 libtcmalloc-minimal4t64 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 vllm \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin vllm

COPY --from=builder /opt/venv /opt/venv
COPY src/vllm_privacy /opt/vllm-privacy/vllm_privacy
COPY docker/vllm/logging.json docker/vllm/entrypoint.sh docker/vllm/warmup.py /opt/vllm-privacy/

# The offline switches make any attempt to reach the Hub fail instead of quietly
# re-downloading: the weights come from the volume only. The server has no need
# for telemetry either. LD_PRELOAD is required by the vLLM CPU backend.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/vllm-privacy \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so

USER 10001:10001

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "/opt/vllm-privacy/entrypoint.sh"]
