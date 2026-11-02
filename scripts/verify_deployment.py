#!/usr/bin/env python3
"""Checks a running deployment against the task's requirements.

Needs only the standard library and kubectl. Start a port-forward first:

    kubectl -n vllm port-forward svc/vllm 18000:8000
    python scripts/verify_deployment.py

Each probe carries a random marker. After the requests are sent, neither the
marker nor the text the model generated may appear in the pod's logs, while the
fact of each request must.
"""

import argparse
import json
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = ROOT / "k8s/overlays/minikube/secrets.env"

# Routes that vLLM itself leaves open even with --api-key set.
OPEN_UPSTREAM_ROUTES = [
    ("POST", "/invocations"),
    ("POST", "/tokenize"),
    ("POST", "/detokenize"),
    ("POST", "/pooling"),
    ("GET", "/metrics"),
    ("GET", "/server_info"),
]

failures = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global failures
    failures += not passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{f': {detail}' if detail else ''}")


def request(base, method, path, token=None, body=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except TimeoutError:
        return 0, "timed out"


def kubectl(namespace, *args):
    command = ["kubectl", "-n", namespace, *args]
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.strip()


def read_token() -> str:
    for line in SECRETS_FILE.read_text().splitlines():
        if line.startswith("VLLM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"VLLM_API_KEY not found in {SECRETS_FILE}")


def verify_auth(base, token):
    status, _ = request(base, "GET", "/health")
    check("/health is open for kubelet probes", status == 200, str(status))

    status, _ = request(base, "GET", "/v1/models")
    check("/v1/models without a token is rejected", status == 401, str(status))

    status, _ = request(base, "GET", "/v1/models", token="wrong-token")
    check("/v1/models with a wrong token is rejected", status == 401, str(status))

    status, _ = request(base, "GET", "/v1/models", token=token)
    check("/v1/models with the token works", status == 200, str(status))

    # No request body: the server answers 401 without reading it, and the reset
    # that follows would kill the kubectl port-forward these checks run through.
    for method, path in OPEN_UPSTREAM_ROUTES:
        status, _ = request(base, method, path, timeout=15)
        check(f"{path} without a token is rejected", status == 401, str(status))


def verify_privacy(base, token, namespace, pod, model):
    marker = f"MARKER{secrets.token_hex(8)}"
    prompt = f"Repeat this code exactly and nothing else: {marker}"
    messages = [{"role": "user", "content": prompt}]

    status, body = request(
        base,
        "POST",
        "/v1/chat/completions",
        token,
        {"model": model, "messages": messages, "max_tokens": 48, "temperature": 0},
    )
    answer = json.loads(body)["choices"][0]["message"]["content"]
    check("chat completion succeeds", status == 200, repr(answer))

    status, _ = request(
        base,
        "POST",
        "/v1/chat/completions",
        token,
        {
            "model": model,
            "messages": messages,
            "max_tokens": 48,
            "temperature": 0,
            "stream": True,
        },
    )
    check("streaming chat completion succeeds", status == 200, str(status))

    # Error paths, which is where prompts tend to leak through exception text.
    oversized = {
        "model": model,
        "prompt": f"{marker} " * 20000,
        "max_tokens": 4,
    }
    status, _ = request(base, "POST", "/v1/completions", token, oversized)
    check("oversized prompt is rejected", status == 400, str(status))

    invalid = {"model": marker, "messages": messages}
    status, _ = request(base, "POST", "/v1/chat/completions", token, invalid)
    check("unknown model is rejected", status in (400, 404), str(status))

    broken_template = {
        "model": model,
        "messages": messages,
        "chat_template": f"{{{{ raise_exception('{marker}') }}}}",
    }
    status, _ = request(base, "POST", "/v1/chat/completions", token, broken_template)
    check("failing chat template is rejected", status >= 400, str(status))

    request(base, "GET", f"/v1/models?query={marker}", token)
    request(base, "GET", f"/{marker}", token)

    logs = kubectl(namespace, "logs", pod, "--all-containers", "--tail=-1")
    check("the prompt is absent from the logs", marker not in logs)
    check("the generated answer is absent from the logs", answer not in logs)
    check("the API token is absent from the logs", token not in logs)
    check(
        "the fact of the request is logged",
        "API request: method=POST path=/v1/chat/completions status=200" in logs,
    )
    check("the request id is logged", "Received request chatcmpl-" in logs)
    check("the finish reason is logged", "finish_reason=" in logs)


def verify_pod(namespace, pod):
    uid = kubectl(namespace, "exec", pod, "-c", "vllm", "--", "id", "-u")
    check("the container runs as non-root", uid != "0", f"uid={uid}")

    env = kubectl(namespace, "exec", pod, "-c", "vllm", "--", "env")
    check("HF_TOKEN does not reach the inference container", "HF_TOKEN=" not in env)
    check("the Hub is switched off", "HF_HUB_OFFLINE=1" in env)

    write = subprocess.run(
        ["kubectl", "-n", namespace, "exec", pod, "-c", "vllm", "--", "touch", "/x"],
        capture_output=True,
    )
    check("the root filesystem is read-only", write.returncode != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--namespace", default="vllm")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    args = parser.parse_args()

    token = read_token()
    pod = kubectl(
        args.namespace,
        "get",
        "pods",
        "-l",
        "app.kubernetes.io/component=inference",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    )
    print(f"Pod: {pod}")

    verify_auth(args.base_url, token)
    verify_privacy(args.base_url, token, args.namespace, pod, args.model)
    verify_pod(args.namespace, pod)

    print(f"\n{failures} check(s) failed" if failures else "\nAll checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
