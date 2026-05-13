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
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # requests has many subclasses; keep caller surface simple.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * (attempt + 1))
        raise RuntimeError(f"GET {url} failed: {last_error}")
