from __future__ import annotations

import logging
import math
import os
from tools.http import HttpClient

logger = logging.getLogger(__name__)
_http = HttpClient(timeout=10, max_retries=1)

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"


def fetch_liquidation_map(symbol: str = "BTC") -> dict:
    """
    청산 레벨 히트맵.
    Coinglass API 키 있으면 실데이터, 없으면 Binance 공개 API로 추정.
    """
    coinglass_key = os.getenv("COINGLASS_API_KEY", "")
    if coinglass_key:
        try:
            data = _http.get_json(
                f"{COINGLASS_BASE}/liquidation_map",
                params={"ex": "Binance", "pair": f"{symbol}USDT", "interval": "12h"},
                headers={"coinglassSecret": coinglass_key},
            )
            if data.get("code") == "0" and data.get("data"):
                raw = data["data"]
                return {
                    "source": "coinglass",
                    "current_price": raw.get("currentPrice"),
                    "levels": [
                        {"price": float(p), "long_usd": float(l), "short_usd": float(s)}
                        for p, l, s in zip(
                            raw.get("prices", []),
                            raw.get("longLiquidation", []),
                            raw.get("shortLiquidation", []),
                        )
                    ],
                }
        except Exception as e:
            logger.warning(f"Coinglass liquidation failed: {e}")

    # Binance public 데이터로 포지션 분포 추정
    return _estimated_liq_map(symbol)


def _estimated_liq_map(symbol: str) -> dict:
    """
    Binance futures 공개 데이터 (가격·OI·펀딩레이트) 로 청산 레벨 추정.
    실제 포지션 DB 없이 레버리지 분포 모델로 계산.
    """
    pair = f"{symbol}USDT"
    try:
        price_info = _http.get_json(f"{BINANCE_FAPI}/ticker/price", params={"symbol": pair})
        current_price = float(price_info["price"])

        oi_info = _http.get_json(f"{BINANCE_FAPI}/openInterest", params={"symbol": pair})
        oi_btc = float(oi_info.get("openInterest", 0))
        oi_usd = oi_btc * current_price

        fr_list = _http.get_json(f"{BINANCE_FAPI}/fundingRate", params={"symbol": pair, "limit": 3})
        funding = float(fr_list[-1].get("fundingRate", 0)) if fr_list else 0.0

        # 펀딩레이트 > 0 → 롱 우세 (롱이 숏에게 비용 지불)
        long_bias   = min(0.75, max(0.25, 0.55 + funding * 800))
        short_bias  = 1.0 - long_bias

        # 레버리지 분포 가정 (소매 거래자 평균)
        leverage_dist = {5: 0.10, 10: 0.30, 20: 0.30, 50: 0.20, 100: 0.10}

        levels = []
        # 현재가 기준 -25% ~ +25%  (1% 단위)
        for pct_i in range(-25, 26):
            pct = pct_i / 100
            level_price = round(current_price * (1 + pct))
            long_liq = 0.0
            short_liq = 0.0

            for lev, w in leverage_dist.items():
                # 롱 청산: 진입가 대비 -1/L 하락 시
                long_liq_pct = -1.0 / lev   # 예: -0.10 @ 10x
                # 숏 청산: 진입가 대비 +1/L 상승 시
                short_liq_pct = 1.0 / lev

                sigma = 0.025  # 분포 폭 (2.5%)

                if pct_i < 0:
                    # 현재가 아래 → 롱 청산 구간
                    z = (pct - long_liq_pct) / sigma
                    g = math.exp(-0.5 * z * z)
                    long_liq += oi_usd * long_bias * w * g * 0.25

                if pct_i > 0:
                    # 현재가 위 → 숏 청산 구간
                    z = (pct - short_liq_pct) / sigma
                    g = math.exp(-0.5 * z * z)
                    short_liq += oi_usd * short_bias * w * g * 0.25

            levels.append({
                "price": level_price,
                "long_usd": round(long_liq),
                "short_usd": round(short_liq),
            })

        return {
            "source": "estimated",
            "current_price": current_price,
            "open_interest_usd": round(oi_usd),
            "funding_rate": round(funding * 100, 4),
            "long_ratio": round(long_bias * 100, 1),
            "note": f"추정값 · OI ${oi_usd/1e9:.1f}B · 펀딩 {funding*100:.4f}% · 롱 {round(long_bias*100)}%",
            "levels": [l for l in levels if l["long_usd"] > 0 or l["short_usd"] > 0],
        }

    except Exception as e:
        logger.error(f"Estimated liq map failed: {e}")
        return {"source": "unavailable", "current_price": None, "levels": []}


def fetch_bubble_coins(limit: int = 20) -> list:
    """CoinGecko 시가총액 상위 코인 버블맵 데이터."""
    headers = {}
    cg_key = os.getenv("COINGECKO_API_KEY", "")
    if cg_key:
        headers["x-cg-demo-api-key"] = cg_key
    try:
        data = _http.get_json(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h",
            },
            headers=headers or None,
        )
        return [
            {
                "id": c["id"],
                "symbol": c["symbol"].upper(),
                "name": c["name"],
                "price": c.get("current_price"),
                "market_cap": c.get("market_cap") or 0,
                "change_24h": c.get("price_change_percentage_24h") or 0,
                "volume_24h": c.get("total_volume") or 0,
            }
            for c in data
        ]
    except Exception as e:
        logger.error(f"Bubble data fetch failed: {e}")
        return []
