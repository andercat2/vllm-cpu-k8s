#!/usr/bin/env python3
"""Startup probe: succeeds once the server has completed real requests.

Triton compiles a kernel the first time it meets a new batch shape, which takes
a minute or more each time, so the pod must not receive traffic before the common
shapes are compiled: single requests and a full batch. Needs only the standard
library, and exits with an error while the server is not listening yet.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 280
BATCH_SIZES = (1, 2, 4, 8)


def call(path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def main() -> int:
    try:
        model = call("/v1/models")["data"][0]["id"]
        body = {"model": model, "prompt": "warm up", "max_tokens": 4}
        for size in BATCH_SIZES:
            with ThreadPoolExecutor(size) as pool:
                list(pool.map(lambda _: call("/v1/completions", body), range(size)))
    except (OSError, urllib.error.URLError) as error:
        print(f"Not ready: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
