from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import random

from tools.config import CONFIG, env
from tools.http import HttpClient
from models.inference import CryptoSentimentModel, aggregate_predictions


class SentimentAnalyzer:
    def __init__(self, model_path: str | None = None) -> None:
        cfg = CONFIG.get("agent", {})
        sent_cfg = CONFIG.get("sentiment", {})
        self.http = HttpClient(timeout=int(cfg.get("request_timeout_sec", 15)))
        self.mock_when_no_key = bool(cfg.get("mock_when_no_key", True))
        self.model = CryptoSentimentModel(
            model_path=model_path or sent_cfg.get("model_path"),
            base_model=sent_cfg.get("base_model", "ElKulako/cryptobert"),
        )

    def fetch_news(self, coin: str, hours: int = 24, limit: int = 30) -> list[dict[str, Any]]:
        """Fetch recent crypto news using CryptoPanic first, NewsAPI second, then mock fallback."""
        panic_key = env("CRYPTOPANIC_API_KEY")
        if panic_key:
            try:
                data = self.http.get_json(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={"auth_token": panic_key, "currencies": self._symbol(coin), "kind": "news", "filter": "rising"},
                )
                return [self._normalize_cryptopanic(x) for x in data.get("results", [])[:limit]]
            except Exception as exc:
                if not self.mock_when_no_key:
                    raise
                print(f"[warn] CryptoPanic failed; using fallback: {exc}")

        newsapi_key = env("NEWSAPI_KEY")
        if newsapi_key:
            try:
                data = self.http.get_json(
                    "https://newsapi.org/v2/everything",
                    params={"q": f"{coin} cryptocurrency", "sortBy": "publishedAt", "pageSize": limit, "apiKey": newsapi_key},
                )
                return [self._normalize_newsapi(x) for x in data.get("articles", [])[:limit]]
            except Exception as exc:
                if not self.mock_when_no_key:
                    raise
                print(f"[warn] NewsAPI failed; using fallback: {exc}")
        return self._mock_news(coin, limit=min(limit, 8))

    def fetch_social(self, coin: str, hours: int = 24, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch public social posts. Reddit/X require app credentials; fallback is deterministic mock posts."""
        posts: list[dict[str, Any]] = []
        # Lightweight unauthenticated Reddit JSON endpoint often works for prototypes.
        try:
            data = self.http.get_json(
                f"https://www.reddit.com/search.json",
                params={"q": f"{coin} crypto", "sort": "new", "limit": min(limit, 25)},
                headers={"User-Agent": env("REDDIT_USER_AGENT", "crypto-intel-agent/0.1") or "crypto-intel-agent/0.1"},
            )
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                posts.append({
                    "source": "reddit",
                    "title": d.get("title", ""),
                    "text": d.get("selftext", ""),
                    "url": "https://reddit.com" + d.get("permalink", ""),
                    "published_at": d.get("created_utc"),
                })
        except Exception:
            pass
        if not posts:
            posts = self._mock_social(coin, min(limit, 10))
        return posts[:limit]

    def analyze_texts(self, items: list[dict[str, Any]], text_keys: tuple[str, ...] = ("title", "text")) -> dict:
        texts: list[str] = []
        for item in items:
            parts = [str(item.get(k, "")) for k in text_keys if item.get(k)]
            text = " | ".join(parts).strip()
            if text:
                texts.append(text)
        preds = self.model.predict_many(texts)
        aggregate = aggregate_predictions(preds)
        enriched = []
        for item, pred in zip(items, preds):
            row = dict(item)
            row.update({"sentiment_label": pred.label, "sentiment_score": pred.score, "confidence": pred.confidence})
            enriched.append(row)
        aggregate.update({"items": enriched, "model_using_transformer": self.model.using_transformer})
        if self.model.load_error and not self.model.using_transformer:
            aggregate["model_warning"] = self.model.load_error
        return aggregate

    def news_sentiment(self, coin: str, hours: int = 24) -> dict:
        news = self.fetch_news(coin, hours)
        result = self.analyze_texts(news, ("title", "text"))
        result.update({"coin": coin, "hours": hours, "top_headlines": [n.get("title", "") for n in news[:5]]})
        return result

    def social_sentiment(self, coin: str, hours: int = 24) -> dict:
        posts = self.fetch_social(coin, hours)
        result = self.analyze_texts(posts, ("title", "text"))
        result.update({"coin": coin, "hours": hours, "top_posts": [p.get("title") or p.get("text", "")[:120] for p in posts[:5]]})
        return result

    def get_fear_greed_index(self, limit: int = 1) -> dict:
        try:
            data = self.http.get_json("https://api.alternative.me/fng/", params={"limit": limit, "format": "json"})
            rows = data.get("data", [])
            if rows:
                latest = rows[0]
                value = int(latest.get("value", 50))
                return {
                    "value": value,
                    "classification": latest.get("value_classification", "Neutral"),
                    "score": value / 100.0,
                    "timestamp": latest.get("timestamp"),
                    "source": "alternative.me",
                    "history": rows,
                }
        except Exception as exc:
            if not self.mock_when_no_key:
                raise
            return {"value": 50, "classification": "Neutral", "score": 0.5, "source": "mock", "error": str(exc)}
        return {"value": 50, "classification": "Neutral", "score": 0.5, "source": "fallback"}

    @staticmethod
    def _normalize_cryptopanic(x: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": x.get("source", {}).get("title", "CryptoPanic"),
            "title": x.get("title", ""),
            "text": x.get("metadata", {}).get("description", ""),
            "url": x.get("url"),
            "published_at": x.get("published_at"),
        }

    @staticmethod
    def _normalize_newsapi(x: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": x.get("source", {}).get("name", "NewsAPI"),
            "title": x.get("title", ""),
            "text": x.get("description", "") or x.get("content", ""),
            "url": x.get("url"),
            "published_at": x.get("publishedAt"),
        }

    @staticmethod
    def _symbol(coin: str) -> str:
        coin_lower = coin.lower()
        coins = CONFIG.get("coins", {})
        return coins.get(coin_lower, {}).get("symbol", coin[:3].upper())

    @staticmethod
    def _mock_news(coin: str, limit: int) -> list[dict[str, Any]]:
        templates = [
            f"{coin.title()} ETF inflows rise as institutional adoption improves",
            f"Analysts warn {coin.title()} may face liquidation pressure after rapid rally",
            f"Large wallets accumulate {coin.title()} while retail sentiment remains cautious",
            f"Developers announce upgrade for {coin.title()} network scalability",
            f"Market volatility increases as traders debate {coin.title()} breakout",
            f"Exchange reserves for {coin.title()} decline amid long-term holder accumulation",
            f"Regulatory uncertainty weighs on short-term {coin.title()} sentiment",
            f"{coin.title()} rebounds after broader crypto market recovery",
        ]
        return [{"source": "mock", "title": t, "text": "", "url": None, "published_at": datetime.now(timezone.utc).isoformat()} for t in templates[:limit]]

    @staticmethod
    def _mock_social(coin: str, limit: int) -> list[dict[str, Any]]:
        samples = [
            f"I am bullish on {coin}; whales look like they are accumulating.",
            f"{coin} price action looks risky after this pump, watching support.",
            f"Neutral on {coin} until volume confirms the breakout.",
            f"Fear is high but long-term holders keep buying {coin}.",
            f"Liquidations could hit {coin} if BTC rejects resistance.",
        ]
        return [{"source": "mock_social", "title": s, "text": "", "url": None, "published_at": datetime.now(timezone.utc).isoformat()} for s in samples[:limit]]
