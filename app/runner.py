from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sentiment import SentimentAnalyzer
from tools.onchain import OnchainAnalyzer
from tools.technical import TechnicalAnalyzer
from tools.signal_engine import SignalEngine
from tools.config import CONFIG
from app.storage import push_signal

logger = logging.getLogger(__name__)

_sentiment  = SentimentAnalyzer()
_onchain    = OnchainAnalyzer()
_technical  = TechnicalAnalyzer()
_engine     = SignalEngine()


def run_analysis(coin: str | None = None) -> dict:
    coin    = (coin or os.getenv("DEFAULT_COIN", "bitcoin")).lower()
    hours   = int(os.getenv("DEFAULT_HOURS", "24"))
    min_usd = int(os.getenv("MIN_WHALE_USD", "1000000"))

    logger.info(f"분석 시작: coin={coin}")

    # ── 데이터 수집 ──────────────────────────────────────────────────────────
    tech   = _technical.analyze(coin)              # RSI/MACD/BB/거래량 (실시간)
    fear   = _sentiment.get_fear_greed_index(1)    # alternative.me (실시간)
    price  = _onchain.get_price(coin)              # CoinGecko (실시간)
    flow   = _onchain.analyze_exchange_flow(coin)  # 거래량 기반 (실시간)
    chain  = CONFIG.get("coins", {}).get(coin, {}).get("chain", "ethereum")
    defi   = _onchain.get_defi_metrics("ethereum" if chain == "bitcoin" else chain)  # DeFiLlama (실시간)

    # 뉴스/소셜: API 키 있으면 실시간, 없으면 rule-based fallback
    news   = _sentiment.news_sentiment(coin, hours)
    social = _sentiment.social_sentiment(coin, hours)

    # API 키 없는 mock 데이터는 감성 점수 중립으로 처리
    news_score   = news.get("avg_score")   if not _is_all_mock(news)   else None
    social_score = social.get("avg_score") if not _is_all_mock(social) else None

    # ── 시그널 생성 ──────────────────────────────────────────────────────────
    signal = _engine.combine(
        technical=       tech,
        fear_greed=      fear,
        sentiment_score= news_score,
        social_score=    social_score,
        exchange_flow=   flow,
        defi_metrics=    defi,
        whale_activity=  _onchain.fetch_whale_txs(coin, min_usd),
    )

    result = {
        "coin":             coin,
        "price_usd":        price.get("usd"),
        "price_change_24h": price.get("usd_24h_change"),
        "signal":           signal,
        "summary": {
            "technical":        tech,
            "fear_greed":       fear,
            "exchange_flow":    flow,
            "defi":             defi,
            "news_avg_score":   news_score,
            "social_avg_score": social_score,
            "top_headlines":    news.get("top_headlines", []),
            "has_real_news":    not _is_all_mock(news),
        },
    }
    push_signal(result)
    logger.info(f"분석 완료: {signal.get('action')}  score={signal.get('score'):.3f}  conf={signal.get('confidence'):.3f}")
    return result


def _is_all_mock(sentiment_result: dict) -> bool:
    """뉴스/소셜 결과가 전부 mock 데이터인지 확인."""
    items = sentiment_result.get("items", [])
    if not items:
        return True
    return all(item.get("source", "").startswith("mock") for item in items)
