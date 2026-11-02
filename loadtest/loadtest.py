#!/usr/bin/env python3
"""Closed-loop load test for an OpenAI-compatible streaming endpoint.

Measures, per scenario:
  TTFT        time to the first generated token
  Latency     time to the last token
  Throughput  completed requests and generated tokens per second

Every prompt starts with a random nonce, so the prefix cache cannot hide the
cost of processing the prompt. Scenarios with "shared" in the name put the nonce
at the end instead, to measure that cache.

    pip install httpx==0.28.1
    kubectl -n vllm port-forward svc/vllm 18000:8000
    VLLM_API_KEY=... python loadtest/loadtest.py --output loadtest/results/baseline.json
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

SHORT_TEXT = "Explain in one paragraph why the sky is blue."

# Approximately 2 tokens per word on this text.
LONG_WORDS = (
    "kubernetes schedules containers across nodes while the kernel isolates "
    "processes using namespaces and cgroups so that memory limits hold under "
    "pressure and noisy neighbours stay away from each other during peak load"
).split()


@dataclass
class Scenario:
    name: str
    prompt_words: int
    max_tokens: int
    concurrency: int
    requests: int


@dataclass
class Sample:
    ttft: float
    latency: float
    prompt_tokens: int
    completion_tokens: int


DEFAULT_SCENARIOS = [
    Scenario("short", 0, 64, 1, 4),
    Scenario("short", 0, 64, 4, 12),
    Scenario("short", 0, 64, 8, 24),
    Scenario("long", 1000, 64, 1, 3),
    Scenario("long", 1000, 64, 4, 8),
    Scenario("long", 1000, 64, 8, 16),
]


def build_prompt(scenario: Scenario) -> str:
    nonce = uuid.uuid4().hex
    words = scenario.prompt_words
    if words == 0:
        return f"[{nonce}] {SHORT_TEXT}"
    question = "Summarise the text above in one paragraph."

    # "shared" scenarios model many questions about one big document: the same
    # context with a different ending, which is what the prefix cache is for.
    if "shared" in scenario.name:
        body = " ".join(random.Random(0).choices(LONG_WORDS, k=words))
        return f"{body}\n\n[{nonce}] {question}"

    body = " ".join(random.choices(LONG_WORDS, k=words))
    return f"[{nonce}] {body}\n\n{question}"


async def one_request(client: httpx.AsyncClient, model: str, s: Scenario) -> Sample:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(s)}],
        "max_tokens": s.max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token = None
    usage = None

    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if first_token is None and choices and choices[0]["delta"].get("content"):
                first_token = time.perf_counter()

    finished = time.perf_counter()
    return Sample(
        ttft=first_token - started,
        latency=finished - started,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


async def run_scenario(client, model, scenario: Scenario) -> dict:
    remaining = list(range(scenario.requests))
    samples: list[Sample] = []
    errors = 0

    async def worker() -> None:
        nonlocal errors
        while remaining:
            remaining.pop()
            try:
                samples.append(await one_request(client, model, scenario))
            except (httpx.HTTPError, KeyError, TypeError) as error:
                errors += 1
                print(f"Request failed: {error!r}", flush=True)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(scenario.concurrency)))
    elapsed = time.perf_counter() - started

    if not samples:
        return {**asdict(scenario), "errors": errors}

    ttft = [s.ttft for s in samples]
    latency = [s.latency for s in samples]
    return {
        **asdict(scenario),
        "errors": errors,
        "prompt_tokens": round(statistics.mean(s.prompt_tokens for s in samples)),
        "ttft_p50": statistics.median(ttft),
        "ttft_p95": percentile(ttft, 0.95),
        "latency_p50": statistics.median(latency),
        "latency_p95": percentile(latency, 0.95),
        "requests_per_second": len(samples) / elapsed,
        "output_tokens_per_second": sum(s.completion_tokens for s in samples) / elapsed,
    }


def print_table(results: list[dict]) -> None:
    print(
        "| Prompt | Prompt tok | Concurrency | Requests | TTFT p50, s | TTFT p95, s "
        "| Latency p50, s | Latency p95, s | Req/s | Output tok/s | Errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "ttft_p50" not in r:
            print(f"| {r['name']} | - | {r['concurrency']} | {r['requests']} "
                  f"| - | - | - | - | - | - | {r['errors']} |")
            continue
        print(
            f"| {r['name']} | {r['prompt_tokens']} | {r['concurrency']} "
            f"| {r['requests']} | {r['ttft_p50']:.2f} | {r['ttft_p95']:.2f} "
            f"| {r['latency_p50']:.2f} | {r['latency_p95']:.2f} "
            f"| {r['requests_per_second']:.2f} "
            f"| {r['output_tokens_per_second']:.1f} | {r['errors']} |"
        )


def parse_scenario(text: str) -> Scenario:
    name, words, max_tokens, concurrency, requests = text.split(":")
    return Scenario(name, int(words), int(max_tokens), int(concurrency), int(requests))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run each scenario once unmeasured: Triton compiles a kernel the "
        "first time it meets a batch shape, which would otherwise end up in "
        "the numbers",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        help="name:prompt_words:max_tokens:concurrency:requests, repeatable",
    )
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    timeout = httpx.Timeout(900, connect=10)
    limits = httpx.Limits(max_connections=None)
    results = []

    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=timeout, limits=limits
    ) as client:
        for scenario in args.scenario or DEFAULT_SCENARIOS:
            print(f"Running {scenario}", flush=True)
            if args.warmup:
                await run_scenario(client, args.model, scenario)
            results.append(await run_scenario(client, args.model, scenario))

    print_table(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
