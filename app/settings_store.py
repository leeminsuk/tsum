from __future__ import annotations

import json
import os
import threading
from pathlib import Path

SETTINGS_PATH = Path(os.getenv("SETTINGS_FILE", "/tmp/tsum_settings.json"))
_lock = threading.Lock()

DEFAULTS = {
    "coin": os.getenv("DEFAULT_COIN", "bitcoin"),
    "interval_hours": int(os.getenv("INTERVAL_HOURS", "5")),
    "min_whale_usd": int(os.getenv("MIN_WHALE_USD", "1000000")),
    "lookback_hours": int(os.getenv("DEFAULT_HOURS", "24")),
}

SUPPORTED_COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]


def load() -> dict:
    with _lock:
        try:
            if SETTINGS_PATH.exists():
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                return {**DEFAULTS, **data}
        except Exception:
            pass
        return dict(DEFAULTS)


def save(data: dict) -> dict:
    with _lock:
        current = load()
        merged = {**current, **data}
        SETTINGS_PATH.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return merged
