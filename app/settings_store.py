from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]

DEFAULTS = {
    "coin": os.getenv("DEFAULT_COIN", "bitcoin"),
    "interval_hours": int(os.getenv("INTERVAL_HOURS", "5")),
    "min_whale_usd": int(os.getenv("MIN_WHALE_USD", "1000000")),
    "lookback_hours": int(os.getenv("DEFAULT_HOURS", "24")),
}

# ── Supabase (reuses the same client from storage) ────────────────────────────

def _supabase():
    from app.storage import _supabase as _get_sb
    return _get_sb()


# ── File fallback ─────────────────────────────────────────────────────────────

_file_lock = threading.Lock()
_FILE = Path(os.getenv("SETTINGS_FILE", "/tmp/tsum_settings.json"))


def _file_load() -> dict:
    try:
        if _FILE.exists():
            data = json.loads(_FILE.read_text(encoding="utf-8"))
            return {**DEFAULTS, **data}
    except Exception:
        pass
    return dict(DEFAULTS)


def _file_save(data: dict) -> None:
    _FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Public API ────────────────────────────────────────────────────────────────

def load() -> dict:
    sb = _supabase()
    if sb:
        try:
            res = sb.table("app_settings").select("data").eq("id", 1).single().execute()
            if res.data:
                return {**DEFAULTS, **res.data["data"]}
        except Exception as exc:
            logger.warning(f"Supabase settings read failed: {exc}")

    with _file_lock:
        return _file_load()


def save(updates: dict) -> dict:
    current = load()
    merged = {**current, **updates}

    sb = _supabase()
    if sb:
        try:
            sb.table("app_settings").upsert({"id": 1, "data": merged}).execute()
            return merged
        except Exception as exc:
            logger.warning(f"Supabase settings write failed: {exc}")

    with _file_lock:
        _file_save(merged)
    return merged
