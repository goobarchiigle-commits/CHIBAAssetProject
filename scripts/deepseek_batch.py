from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-pro"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_SLEEP_SECONDS = 1.0


@dataclass
class BatchResult:
    index: int
    prompt: str
    status: str
    attempt_count: int
    response_text: str
    error_message: str
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/prompts.txt"))
    parser.add_argument("--output", type=Path, default=Path("results/deepseek_batch_results.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=50)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def extract_response_text(payload: dict[str, Any]) -> str:
    return payload.get("choices", [{}])[0].get("message", {}).get("content", "")


def call_deepseek(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    temperature: float,
    max_tokens: int,
) -> BatchResult:

    url = f"{base_url}/chat/completions"
    headers = build_headers(api_key)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, headers=headers, json=body, timeout=timeout)
            res.raise_for_status()

            return BatchResult(
                index=-1,
                prompt=prompt,
                status="success",
                attempt_count=attempt,
                response_text=extract_response_text(res.json()),
                error_message="",
                elapsed_seconds=time.perf_counter() - start,
            )

        except Exception as e:
            last_error = str(e)

            if attempt < max_retries:
                sleep = retry_backoff ** (attempt - 1)
                time.sleep(sleep)

    return BatchResult(
        index=-1,
        prompt=prompt,
        status="error",
        attempt_count=max_retries,
        response_text="",
        error_message=last_error,
        elapsed_seconds=time.perf_counter() - start,
    )


def main():
    args = parse_args()

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("NVIDIA_API_KEY not set")
        return

    prompts = load_prompts(args.input)
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"{i}/{len(prompts)} processing")

        r = call_deepseek(
            prompt,
            api_key,
            args.base_url,
            args.model,
            args.timeout,
            args.max_retries,
            args.retry_backoff,
            args.temperature,
            args.max_tokens,
        )

        r.index = i
        results.append(r)

        time.sleep(args.sleep_seconds)

    Path(args.output).parent.mkdir(exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BatchResult.__annotations__.keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    print("DONE")


if __name__ == "__main__":
    main()