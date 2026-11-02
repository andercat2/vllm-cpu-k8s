# Lightweight image for the initContainer that fetches the model. Build from the
# repository root:
#   docker build -f docker/model-downloader/Dockerfile -t vllm-model-downloader:1.0.0 .
FROM python:3.12-alpine3.22@sha256:a190708a2dec1bd18b1decb539f8e8f5407abaa9bf39cacda583f7f8c11db322

COPY docker/model-downloader/requirements.txt /build/requirements.txt
COPY docker/model-downloader/download_model.py /opt/downloader/download_model.py

RUN pip install --no-cache-dir --no-deps --require-hashes --root-user-action=ignore \
        -r /build/requirements.txt \
    && rm -rf /build

ENV HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 10001:10001

ENTRYPOINT ["python", "/opt/downloader/download_model.py"]
