"""
tools/cg.py
===========
CoinGecko 공유 헬퍼 — 무료 한도(분당 호출) 안에서 4개 코인 실시간 값을 보장한다.

- batch_prices(): /simple/price를 코인 4종 묶음 1회 호출 + 60초 TTL 캐시
- market_chart(): (cg_id, days)별 5분 TTL 캐시 — technical/exchange_flow가 같은 차트를 공유
- 429는 tools.http의 Retry-After 백오프가 처리하고, 여기는 호출 수 자체를 줄인다
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from tools.config import CONFIG, env
from tools.http import HttpClient

logger = logging.getLogger(__name__)

PRICE_TTL_SEC = 60      # 가격은 1분 캐시 — 4코인이 같은 실시간 스냅샷 공유
CHART_TTL_SEC = 300     # 30일/7일 일봉 차트는 5분 캐시면 충분

_lock = threading.Lock()
_price_cache: dict[str, Any] = {"ts": 0.0, "data": {}}
_chart_cache: dict[tuple[str, int], dict[str, Any]] = {}


def cg_headers() -> dict:
    key = env("COINGECKO_API_KEY", "")
    return {"x-cg-demo-api-key": key} if key else {}


def all_cg_ids() -> list[str]:
    coins = CONFIG.get("coins", {})
    return [v.get("coingecko_id", k) for k, v in coins.items()]


def batch_prices(http: HttpClient, cg_id: str) -> dict[str, Any]:
    """설정된 전체 코인을 한 번에 조회해 캐시하고, 요청한 cg_id의 행을 반환."""
    with _lock:
        fresh = time.time() - _price_cache["ts"] < PRICE_TTL_SEC
        if fresh and cg_id in _price_cache["data"]:
            return _price_cache["data"][cg_id]

    ids = all_cg_ids()
    if cg_id not in ids:
        ids = ids + [cg_id]
    data = http.get_json(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": ",".join(ids),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        },
        headers=cg_headers() or None,
    )
    with _lock:
        _price_cache["ts"] = time.time()
        _price_cache["data"] = data if isinstance(data, dict) else {}
        return _price_cache["data"].get(cg_id, {})


def market_chart(http: HttpClient, cg_id: str, days: int) -> dict:
    """market_chart 공유 캐시 — 같은 (코인, 기간) 차트를 모듈 전체가 1회만 호출."""
    key = (cg_id, days)
    with _lock:
        hit = _chart_cache.get(key)
        if hit and time.time() - hit["ts"] < CHART_TTL_SEC:
            return hit["data"]

    data = http.get_json(
        f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart",
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        headers=cg_headers() or None,
    )
    chart = data if isinstance(data, dict) else {}
    with _lock:
        _chart_cache[key] = {"ts": time.time(), "data": chart}
    return chart
