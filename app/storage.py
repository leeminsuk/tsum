from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os

STORAGE_PATH = Path(os.getenv("STORAGE_FILE", "/tmp/tsum_signals.json"))
MAX_STACK = 5
_lock = threading.Lock()


def _load() -> list[dict]:
    try:
        if STORAGE_PATH.exists():
            return json.loads(STORAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save(signals: list[dict]) -> None:
    STORAGE_PATH.write_text(
        json.dumps(signals, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_signals() -> list[dict]:
    with _lock:
        return _load()


def push_signal(data: dict[str, Any]) -> list[dict]:
    """Insert new signal at front (newest first). Trim to MAX_STACK."""
    with _lock:
        signals = _load()
        entry = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        signals.insert(0, entry)
        if len(signals) > MAX_STACK:
            signals = signals[:MAX_STACK]
        _save(signals)
        return signals
