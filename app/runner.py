from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on path so tools/ and models/ are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sentiment import SentimentAnalyzer
from tools.onchain import OnchainAnalyzer
from tools.signal_engine import SignalEngine
from tools.config import CONFIG
from app.storage import push_signal

logger = logging.getLogger(__name__)

_sentiment = SentimentAnalyzer()
_onchain = OnchainAnalyzer()
_engine = SignalEngine()


def run_analysis(coin: str | None = None) -> dict:
    coin = (coin or os.getenv("DEFAULT_COIN", "bitcoin")).lower()
    hours = int(os.getenv("DEFAULT_HOURS", "24"))
    min_usd = int(os.getenv("MIN_WHALE_USD", "1000000"))

    logger.info(f"Running analysis: coin={coin} hours={hours}")

    news = _sentiment.news_sentiment(coin, hours)
    social = _sentiment.social_sentiment(coin, hours)
    fear = _sentiment.get_fear_greed_index(1)
    whales = _onchain.fetch_whale_txs(coin, min_usd)
    flow = _onchain.analyze_exchange_flow(coin)
    price = _onchain.get_price(coin)
    chain = CONFIG.get("coins", {}).get(coin, {}).get("chain", "ethereum")
    defi = _onchain.get_defi_metrics("ethereum" if chain == "bitcoin" else chain)

    signal = _engine.combine(
        sentiment_score=news.get("avg_score"),
        social_score=social.get("avg_score"),
        fear_greed=fear,
        whale_activity=whales,
        exchange_flow=flow,
        defi_metrics=defi,
    )

    result = {
        "coin": coin,
        "price_usd": price.get("usd"),
        "price_change_24h": price.get("usd_24h_change"),
        "signal": signal,
        "summary": {
            "news_avg_score": news.get("avg_score"),
            "social_avg_score": social.get("avg_score"),
            "fear_greed": fear,
            "whale_count": len(whales),
            "exchange_flow": flow,
            "defi": defi,
            "top_headlines": news.get("top_headlines", []),
            "model_using_transformer": news.get("model_using_transformer", False),
        },
    }
    push_signal(result)
    logger.info(f"Analysis done: {signal.get('action')} score={signal.get('score'):.3f}")
    return result
