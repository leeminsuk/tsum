"""
tools/technical.py
==================
CoinGecko 무료 OHLC + market_chart 데이터로 실시간 기술적 지표 계산.
API 키 불필요. 모든 데이터 실시간.

지표:
  - RSI (14)          : 과매수(>70) / 과매도(<30)
  - MACD              : 골든크로스 / 데드크로스
  - 볼린저 밴드 위치   : 상단 돌파 / 하단 이탈
  - 거래량 모멘텀      : 현재 거래량 vs 7일 평균
  - 가격 모멘텀        : 7일 / 30일 수익률
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any

from tools.config import CONFIG, env
from tools.http import HttpClient

logger = logging.getLogger(__name__)

COINGECKO_DELAY = 1.2  # 무료 30 req/min 대응


class TechnicalAnalyzer:
    def __init__(self) -> None:
        cfg = CONFIG.get("agent", {})
        self.http = HttpClient(timeout=int(cfg.get("request_timeout_sec", 15)))

    # ── 공개 메서드 ────────────────────────────────────────────────────────────

    def analyze(self, coin: str) -> dict[str, Any]:
        """RSI, MACD, BB, 거래량, 모멘텀을 종합해 0~1 score + 근거 반환."""
        cg_id = self._cg_id(coin)
        headers = self._cg_headers()

        ohlc   = self._fetch_ohlc(cg_id, days=14, headers=headers)
        chart  = self._fetch_market_chart(cg_id, days=30, headers=headers)

        if not ohlc and not chart:
            return self._fallback(coin, "CoinGecko 데이터 없음")

        closes  = [c[4] for c in ohlc] if ohlc else []
        volumes = [v[1] for v in chart.get("total_volumes", [])] if chart else []
        prices  = [p[1] for p in chart.get("prices", [])]         if chart else []

        signals: dict[str, float] = {}
        notes:   list[str]        = []
        flags:   list[str]        = []

        # RSI
        if len(closes) >= 15:
            rsi = self._rsi(closes)
            signals["rsi"] = self._rsi_score(rsi)
            notes.append(f"RSI(14): {rsi:.1f}  →  {'과매수⚠️' if rsi>70 else '과매도🟢' if rsi<30 else '중립'}")
            if rsi > 75:
                flags.append(f"RSI {rsi:.0f}: 과매수 구간, 단기 조정 가능")
            elif rsi < 25:
                flags.append(f"RSI {rsi:.0f}: 극단적 과매도, 반등 기대")

        # MACD
        if len(closes) >= 26:
            macd_line, signal_line = self._macd(closes)
            signals["macd"] = 0.65 if macd_line > signal_line else 0.35
            cross = "골든크로스 🟢" if macd_line > signal_line else "데드크로스 🔴"
            notes.append(f"MACD: {cross} ({macd_line:+.4f} vs {signal_line:+.4f})")

        # 볼린저 밴드
        if len(closes) >= 20:
            bb_pos = self._bb_position(closes)
            if bb_pos is not None:
                signals["bb"] = self._bb_score(bb_pos)
                pct = bb_pos * 100
                notes.append(f"볼린저 밴드 위치: {pct:.0f}%  ({'상단' if pct>80 else '하단' if pct<20 else '중간'})")
                if pct > 90:
                    flags.append("볼린저 상단 돌파: 과열 신호")
                elif pct < 10:
                    flags.append("볼린저 하단 이탈: 낙폭 과대")

        # 거래량 모멘텀
        if len(volumes) >= 8:
            vol_ratio = self._volume_ratio(volumes)
            signals["volume"] = self._volume_score(vol_ratio)
            trend = "급증📈" if vol_ratio > 1.8 else "증가" if vol_ratio > 1.2 else "감소📉" if vol_ratio < 0.7 else "보통"
            notes.append(f"거래량: 7일 평균 대비 {vol_ratio:.2f}x  ({trend})")

        # 가격 모멘텀
        if len(prices) >= 7:
            mom7  = (prices[-1] - prices[-7])  / prices[-7]  * 100 if prices[-7]  else 0.0
            mom30 = (prices[-1] - prices[0])   / prices[0]   * 100 if prices[0]   else 0.0
            signals["momentum"] = self._momentum_score(mom7, mom30)
            notes.append(f"모멘텀: 7일 {mom7:+.1f}%  /  30일 {mom30:+.1f}%")
            if abs(mom7) > 20:
                flags.append(f"7일 급{'등' if mom7>0 else '락'} ({mom7:+.1f}%): 변동성 주의")

        if not signals:
            return self._fallback(coin, "지표 계산 데이터 부족")

        # 종합 score: 단순 평균
        score = statistics.mean(signals.values())

        return {
            "coin":     coin,
            "score":    round(score, 4),
            "signal":   "bullish" if score >= 0.58 else "bearish" if score <= 0.42 else "neutral",
            "components": signals,
            "rationale":  notes,
            "risk_flags": flags,
            "source":   "coingecko_ohlc",
        }

    # ── 지표 계산 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> float:
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [max(d, 0.0) for d in deltas]
        losses = [max(-d, 0.0) for d in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data: list[float], period: int) -> float:
        k = 2.0 / (period + 1)
        val = data[0]
        for price in data[1:]:
            val = price * k + val * (1 - k)
        return val

    def _macd(self, closes: list[float]) -> tuple[float, float]:
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        # signal line = EMA9 of last-9 MACD values (simplified: single-pass EMA of closes)
        signal_line = self._ema(closes[-9:], 9) * (2 / (12 + 1)) - self._ema(closes[-9:], 9) * (2 / (26 + 1))
        return macd_line, signal_line

    @staticmethod
    def _bb_position(closes: list[float], period: int = 20) -> float | None:
        window = closes[-period:]
        if len(window) < period:
            return None
        mean  = statistics.mean(window)
        stdev = statistics.stdev(window)
        if stdev == 0:
            return 0.5
        upper = mean + 2 * stdev
        lower = mean - 2 * stdev
        pos   = (closes[-1] - lower) / (upper - lower)
        return max(0.0, min(1.0, pos))

    @staticmethod
    def _volume_ratio(volumes: list[float]) -> float:
        if len(volumes) < 2:
            return 1.0
        avg7 = statistics.mean(volumes[-8:-1]) or 1.0
        return volumes[-1] / avg7

    # ── 점수 변환 (0~1) ───────────────────────────────────────────────────────

    @staticmethod
    def _rsi_score(rsi: float) -> float:
        # 과매도(낮은 RSI) → bullish(높은 score), 과매수 → bearish
        if rsi <= 30:   return 0.80
        if rsi <= 40:   return 0.65
        if rsi <= 60:   return 0.50
        if rsi <= 70:   return 0.40
        return 0.25  # >70 과매수

    @staticmethod
    def _bb_score(pos: float) -> float:
        # 하단(0) → bullish, 상단(1) → bearish
        return 1.0 - pos

    @staticmethod
    def _volume_score(ratio: float) -> float:
        # 거래량 단독으론 방향 모름 → 중립 근처 유지하되 극단 제외
        if ratio > 2.5:  return 0.55  # 급증: 방향성 확인 필요
        if ratio > 1.3:  return 0.55
        if ratio < 0.5:  return 0.45  # 거래량 급감: 추세 약화
        return 0.50

    @staticmethod
    def _momentum_score(mom7: float, mom30: float) -> float:
        # 수익률을 sigmoid로 0~1 변환
        import math
        blended = mom7 * 0.6 + mom30 * 0.4
        return 1.0 / (1.0 + math.exp(-blended / 15.0))

    # ── CoinGecko 요청 ────────────────────────────────────────────────────────

    def _fetch_ohlc(self, cg_id: str, days: int, headers: dict) -> list:
        try:
            data = self.http.get_json(
                f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc",
                params={"vs_currency": "usd", "days": days},
                headers=headers or None,
            )
            time.sleep(COINGECKO_DELAY)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"OHLC fetch failed ({cg_id}): {e}")
            return []

    def _fetch_market_chart(self, cg_id: str, days: int, headers: dict) -> dict:
        try:
            data = self.http.get_json(
                f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart",
                params={"vs_currency": "usd", "days": days, "interval": "daily"},
                headers=headers or None,
            )
            time.sleep(COINGECKO_DELAY)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"market_chart fetch failed ({cg_id}): {e}")
            return {}

    @staticmethod
    def _cg_headers() -> dict:
        key = env("COINGECKO_API_KEY", "")
        return {"x-cg-demo-api-key": key} if key else {}

    @staticmethod
    def _cg_id(coin: str) -> str:
        return CONFIG.get("coins", {}).get(coin.lower(), {}).get("coingecko_id", coin.lower())

    @staticmethod
    def _fallback(coin: str, reason: str) -> dict:
        logger.warning(f"Technical fallback ({coin}): {reason}")
        return {
            "coin": coin, "score": 0.5, "signal": "neutral",
            "components": {}, "rationale": [f"기술 분석 unavailable: {reason}"],
            "risk_flags": [], "source": "fallback",
        }
