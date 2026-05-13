from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_STACK = 5

# ── Supabase client (lazy, optional) ─────────────────────────────────────────

_sb_lock = threading.Lock()
_sb_client = None
_sb_tried = False


def _supabase():
    """Return Supabase client if env vars are set, else None."""
    global _sb_client, _sb_tried
    with _sb_lock:
        if _sb_tried:
            return _sb_client
        _sb_tried = True
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        if not url or not key:
            logger.info("Supabase not configured — using file fallback")
            return None
        try:
            from supabase import create_client
            _sb_client = create_client(url, key)
            logger.info("Supabase connected")
        except Exception as exc:
            logger.warning(f"Supabase init failed: {exc} — falling back to file")
            _sb_client = None
        return _sb_client


# ── File fallback ─────────────────────────────────────────────────────────────

_file_lock = threading.Lock()
_FILE = Path(os.getenv("STORAGE_FILE", "/tmp/tsum_signals.json"))


def _file_load() -> list[dict]:
    try:
        if _FILE.exists():
            return json.loads(_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _file_save(signals: list[dict]) -> None:
    _FILE.write_text(json.dumps(signals, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Public API ────────────────────────────────────────────────────────────────

def get_signals() -> list[dict]:
    sb = _supabase()
    if sb:
        try:
            res = sb.table("signals").select("*").order("generated_at", desc=True).limit(MAX_STACK).execute()
            return res.data or []
        except Exception as exc:
            logger.warning(f"Supabase read failed: {exc}")

    with _file_lock:
        return _file_load()


def push_signal(data: dict[str, Any]) -> list[dict]:
    """Insert newest signal; keep at most MAX_STACK rows (delete oldest if over)."""
    now = datetime.now(timezone.utc)
    entry = {
        "id": now.strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": now.isoformat(),
        "coin": data.get("coin", "bitcoin"),
        "price_usd": data.get("price_usd"),
        "price_change_24h": data.get("price_change_24h"),
        "signal": data.get("signal", {}),
        "summary": data.get("summary", {}),
    }

    sb = _supabase()
    if sb:
        try:
            sb.table("signals").insert(entry).execute()
            # Enforce max stack: delete oldest rows beyond MAX_STACK
            all_rows = (
                sb.table("signals")
                .select("id, generated_at")
                .order("generated_at", desc=True)
                .execute()
                .data or []
            )
            if len(all_rows) > MAX_STACK:
                ids_to_delete = [r["id"] for r in all_rows[MAX_STACK:]]
                sb.table("signals").delete().in_("id", ids_to_delete).execute()
            return get_signals()
        except Exception as exc:
            logger.warning(f"Supabase write failed: {exc}")

    # File fallback
    with _file_lock:
        signals = _file_load()
        signals.insert(0, entry)
        if len(signals) > MAX_STACK:
            signals = signals[:MAX_STACK]
        _file_save(signals)
        return signals
