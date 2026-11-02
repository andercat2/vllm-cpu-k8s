#!/usr/bin/env bash
# Regenerates both requirements.txt files, with hashes, inside a Linux container
# that matches the base images.
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_VERSION=0.32.0
UV_VERSION=0.13.6

host_dir=$(pwd -W 2>/dev/null || pwd)

MSYS_NO_PATHCONV=1 docker run --rm -v "${host_dir}/docker:/work" -w /work python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 sh -euc "
    pip install --quiet --root-user-action=ignore uv==${UV_VERSION}

    uv pip compile --quiet vllm/requirements.in -o vllm/requirements.txt \
        --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
        --torch-backend cpu --generate-hashes --emit-index-url --no-header \
        --extra-index-url https://wheels.vllm.ai/${VLLM_VERSION}/cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match

    uv pip compile --quiet model-downloader/requirements.in \
        -o model-downloader/requirements.txt \
        --python-version 3.12 --python-platform x86_64-unknown-linux-musl \
        --generate-hashes --no-header
"
