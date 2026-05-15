from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_STACK = 5  # per-coin

# ── Supabase client (lazy, optional) ─────────────────────────────────────────

_sb_lock = threading.Lock()
_sb_client = None
_sb_tried = False


def _supabase():
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

def get_signals(coin: str | None = None) -> list[dict]:
    """coin 지정 시 해당 코인만, 없으면 전체 최신 MAX_STACK개."""
    sb = _supabase()
    if sb:
        try:
            q = sb.table("signals").select("*").order("generated_at", desc=True)
            if coin:
                q = q.eq("coin", coin.lower())
            res = q.limit(MAX_STACK).execute()
            return res.data or []
        except Exception as exc:
            logger.warning(f"Supabase read failed: {exc}")

    with _file_lock:
        all_signals = _file_load()

    if coin:
        filtered = [s for s in all_signals if s.get("coin") == coin.lower()]
        return filtered[:MAX_STACK]
    return all_signals[:MAX_STACK]


def has_recent_signal(coin: str, within_hours: float = 1.0) -> bool:
    """최근 within_hours 시간 내 해당 코인 신호가 있는지 확인."""
    signals = get_signals(coin=coin)
    if not signals:
        return False
    try:
        ts_str = signals[0].get("generated_at", "")
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        diff = (datetime.now(timezone.utc) - ts).total_seconds()
        return diff < within_hours * 3600
    except Exception:
        return False


def push_signal(data: dict[str, Any]) -> list[dict]:
    now = datetime.now(timezone.utc)
    coin = data.get("coin", "bitcoin").lower()
    entry = {
        "id": f"{coin}_{now.strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": now.isoformat(),
        "coin": coin,
        "price_usd": data.get("price_usd"),
        "price_change_24h": data.get("price_change_24h"),
        "signal": data.get("signal", {}),
        "summary": data.get("summary", {}),
    }

    sb = _supabase()
    if sb:
        try:
            sb.table("signals").insert(entry).execute()
            # 코인별로 MAX_STACK 유지
            all_rows = (
                sb.table("signals")
                .select("id, generated_at")
                .eq("coin", coin)
                .order("generated_at", desc=True)
                .execute()
                .data or []
            )
            if len(all_rows) > MAX_STACK:
                ids_to_delete = [r["id"] for r in all_rows[MAX_STACK:]]
                sb.table("signals").delete().in_("id", ids_to_delete).execute()
            return get_signals(coin=coin)
        except Exception as exc:
            logger.warning(f"Supabase write failed: {exc}")

    # File fallback — 코인별 MAX_STACK 유지
    with _file_lock:
        all_signals = _file_load()
        all_signals.insert(0, entry)
        # 코인별로 최대 MAX_STACK개만 유지
        coin_counts: dict[str, int] = defaultdict(int)
        trimmed = []
        for s in all_signals:
            c = s.get("coin", "bitcoin")
            if coin_counts[c] < MAX_STACK:
                trimmed.append(s)
                coin_counts[c] += 1
        _file_save(trimmed)
        return [s for s in trimmed if s.get("coin") == coin]
