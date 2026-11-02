#!/usr/bin/env python3
"""Puts a Hugging Face model on the shared volume, once.

A marker file is written after a complete download. While it matches the
requested model and revision the script exits without touching the network, so
pod restarts load the weights from the volume. An interrupted download has no
marker and resumes on the next run.
"""

import json
import os
import sys
from pathlib import Path

# Other weight formats next to the safetensors would only waste disk.
IGNORE_PATTERNS = ["*.bin", "*.pth", "*.pt", "*.h5", "*.msgpack", "*.onnx", "*.gguf"]

MARKER_NAME = ".download-complete"


def log(message: str) -> None:
    print(f"[model-downloader] {message}", flush=True)


def is_cached(marker: Path, expected: dict[str, str]) -> bool:
    try:
        return json.loads(marker.read_text()) == expected
    except (OSError, ValueError):
        return False


def main() -> int:
    model_id = os.environ["MODEL_ID"]
    revision = os.environ["MODEL_REVISION"]
    target = Path(os.environ["MODEL_ROOT"]) / os.environ["MODEL_DIR"]

    marker = target / MARKER_NAME
    expected = {"model_id": model_id, "revision": revision}

    if is_cached(marker, expected):
        log(f"cache hit: {model_id}@{revision} is in {target}, skipping the network")
        return 0

    # Imported here so that a cache hit cannot depend on anything Hub-related.
    from huggingface_hub import snapshot_download

    log(f"cache miss: downloading {model_id}@{revision} to {target}")
    marker.unlink(missing_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=target,
        token=os.environ.get("HF_TOKEN") or False,
        ignore_patterns=IGNORE_PATTERNS,
    )

    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected))
    temporary.replace(marker)
    log("download complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
