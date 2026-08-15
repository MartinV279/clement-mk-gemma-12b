#!/usr/bin/env python3
"""Shared teacher-API client (OpenAI-compatible, stdlib-only).

The provider runs a peak/off-peak schedule and throttles harder during peak
windows, so batch generation is scheduled off-peak. Peak (UTC): 01:00-04:00 and
06:00-10:00 (Skopje/UTC+2: 03:00-06:00 and 08:00-12:00). is_peak_hour() guards
every bulk run; allow_peak=True overrides it.

Reads TEACHER_API_KEY from .env. Tracks token usage across calls so every run
can report what it consumed and stop at a cap.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://api.deepseek.com"
PEAK_RANGES_UTC = [(1, 4), (6, 10)]


class Usage:
    """Token accounting shared by every generator, so a run can cap itself."""

    def __init__(self):
        self.prompt = self.prompt_cached = self.completion = self.calls = 0

    def add(self, u: dict) -> None:
        self.calls += 1
        self.prompt += u.get("prompt_tokens", 0)
        self.prompt_cached += u.get("prompt_cache_hit_tokens", 0)
        self.completion += u.get("completion_tokens", 0)

    def total(self) -> int:
        return self.prompt + self.completion

    def report(self) -> str:
        return (f"{self.calls} calls | in {self.prompt:,} "
                f"(cached {self.prompt_cached:,}) | out {self.completion:,} "
                f"| total {self.total():,}")


def is_peak_hour(now=None) -> bool:
    h = (now or datetime.now(timezone.utc)).hour
    return any(lo <= h < hi for lo, hi in PEAK_RANGES_UTC)


def api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("TEACHER_API_KEY")
    if not key:
        raise SystemExit("TEACHER_API_KEY empty — add it to .env")
    return key


def list_models(key: str) -> list:
    req = urllib.request.Request(f"{BASE_URL}/models",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def chat(key: str, model: str, messages: list, usage: Usage,
         temperature: float = 0.8, max_tokens: int = 2048, retries: int = 4) -> str:
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            usage.add(data.get("usage", {}))
            choice = data["choices"][0]
            content = choice["message"]["content"]
            # Reasoning tiers spend max_tokens on hidden reasoning FIRST: an
            # empty/truncated content with finish_reason=length means the budget
            # was too small — the caller must raise max_tokens, not retry.
            if not content or choice.get("finish_reason") == "length":
                raise ValueError(f"incomplete answer (finish_reason="
                                 f"{choice.get('finish_reason')}, {len(content or '')} chars) — "
                                 f"raise max_tokens (reasoning models need headroom)")
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry {attempt+1} in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("unreachable")
