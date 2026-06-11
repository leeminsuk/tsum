from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time
import requests


@dataclass
class HttpClient:
    timeout: int = 15
    max_retries: int = 2
    backoff_sec: float = 0.75

    def get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                # 429(rate limit)는 분 단위 한도라 짧은 백오프로는 안 풀린다 — Retry-After 존중
                if resp.status_code == 429 and attempt < self.max_retries:
                    wait = float(resp.headers.get("Retry-After") or 20 * (attempt + 1))
                    time.sleep(min(wait, 60.0))
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # requests has many subclasses; keep caller surface simple.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (attempt + 1))
        raise RuntimeError(f"GET {url} failed: {last_error}")
