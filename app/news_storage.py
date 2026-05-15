from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_STACK = 5
_lock = threading.Lock()
_NEWS_FILE = Path(os.getenv("NEWS_FILE", "/tmp/tsum_news.json"))


def _load_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def get_news() -> list[dict]:
    sb = _load_supabase()
    if sb:
        try:
            res = sb.table("news_summaries").select("*").order("generated_at", desc=True).limit(MAX_STACK).execute()
            return res.data or []
        except Exception as e:
            logger.warning(f"Supabase news read failed: {e}")
    if _NEWS_FILE.exists():
        try:
            return json.loads(_NEWS_FILE.read_text())
        except Exception:
            pass
    return []


def push_news(item: dict) -> None:
    with _lock:
        sb = _load_supabase()
        if sb:
            try:
                sb.table("news_summaries").insert(item).execute()
                rows = sb.table("news_summaries").select("id").order("generated_at", desc=True).execute()
                ids = [r["id"] for r in (rows.data or [])]
                for old_id in ids[MAX_STACK:]:
                    sb.table("news_summaries").delete().eq("id", old_id).execute()
                return
            except Exception as e:
                logger.warning(f"Supabase news write failed: {e}")

        stack = get_news()
        stack.insert(0, item)
        stack = stack[:MAX_STACK]
        try:
            _NEWS_FILE.write_text(json.dumps(stack, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"News file write failed: {e}")
