from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import os
import re
import numpy as np


BULLISH_WORDS = {
    "surge", "rally", "breakout", "adoption", "institutional", "accumulate", "inflow",
    "etf", "approval", "bull", "bullish", "buy", "long", "record", "ath", "upgrade",
    "partnership", "rebound", "상승", "호재", "매집", "급등", "반등", "승인", "강세",
}
BEARISH_WORDS = {
    "crash", "dump", "selloff", "hack", "exploit", "lawsuit", "ban", "liquidation",
    "bear", "bearish", "short", "outflow", "fraud", "decline", "plunge", "risk",
    "급락", "악재", "해킹", "소송", "금지", "청산", "약세", "매도", "하락",
}


@dataclass
class Prediction:
    label: str
    score: float  # 0 bearish -> 1 bullish
    confidence: float
    probabilities: dict[str, float]


class CryptoSentimentModel:
    """Transformer inference wrapper with a deterministic rule-based fallback.

    The fallback makes the MCP server usable before fine-tuning or without GPU/model downloads.
    """

    def __init__(self, model_path: str | Path | None = None, base_model: str = "ElKulako/cryptobert") -> None:
        self.model_path = str(model_path) if model_path else None
        self.base_model = base_model
        self._pipeline = None
        self._load_error: str | None = None
        self._try_load_transformer()

    @property
    def using_transformer(self) -> bool:
        return self._pipeline is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _try_load_transformer(self) -> None:
        candidates: list[str] = []
        if self.model_path and Path(self.model_path).exists() and any(Path(self.model_path).iterdir()):
            candidates.append(self.model_path)
        # Avoid unexpected network/model downloads during local MCP startup.
        # Set LOAD_BASE_MODEL=true when you want to load Hugging Face base_model automatically.
        if os.getenv("LOAD_BASE_MODEL", "false").lower() == "true":
            candidates.append(self.base_model)
        if not candidates:
            self._load_error = "No local fine-tuned model found; using rule-based fallback."
            return
        for candidate in candidates:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    "text-classification",
                    model=candidate,
                    tokenizer=candidate,
                    top_k=None,
                    truncation=True,
                    max_length=256,
                )
                self._load_error = None
                return
            except Exception as exc:
                self._load_error = f"Could not load {candidate}: {exc}"
        self._pipeline = None

    def predict(self, text: str) -> Prediction:
        if self._pipeline is not None:
            return self._predict_transformer(text)
        return self._predict_rules(text)

    def predict_many(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]

    def _predict_transformer(self, text: str) -> Prediction:
        raw = self._pipeline(text)
        # pipeline(top_k=None) returns [[{label, score}, ...]] for a single input in recent transformers.
        items = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        probs = {str(x["label"]).lower(): float(x["score"]) for x in items}
        bearish = self._pick_prob(probs, ["bearish", "label_0", "negative", "neg"])
        neutral = self._pick_prob(probs, ["neutral", "label_1", "neu"])
        bullish = self._pick_prob(probs, ["bullish", "label_2", "positive", "pos"])
        total = bearish + neutral + bullish
        if total <= 0:
            return self._predict_rules(text)
        bearish, neutral, bullish = bearish / total, neutral / total, bullish / total
        score = bullish + 0.5 * neutral
        label = "bullish" if score >= 0.58 else "bearish" if score <= 0.42 else "neutral"
        confidence = max(bearish, neutral, bullish)
        return Prediction(label, float(score), float(confidence), {"bearish": bearish, "neutral": neutral, "bullish": bullish})

    @staticmethod
    def _pick_prob(probs: dict[str, float], keys: list[str]) -> float:
        for key in keys:
            if key in probs:
                return probs[key]
        return 0.0

    def _predict_rules(self, text: str) -> Prediction:
        tokens = set(re.findall(r"[A-Za-z가-힣]+", text.lower()))
        bull = len(tokens & BULLISH_WORDS)
        bear = len(tokens & BEARISH_WORDS)
        raw = bull - bear
        score = 1.0 / (1.0 + math.exp(-raw / 2.0))
        # Keep neutral-looking content near 0.5.
        if bull == 0 and bear == 0:
            score = 0.5
        label = "bullish" if score >= 0.58 else "bearish" if score <= 0.42 else "neutral"
        confidence = abs(score - 0.5) * 2 if label != "neutral" else 0.55
        probs = {
            "bearish": float(max(0.0, 1 - score - 0.15)),
            "neutral": float(0.30 if label != "neutral" else 0.70),
            "bullish": float(max(0.0, score - 0.15)),
        }
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}
        return Prediction(label, float(score), float(confidence), probs)


def aggregate_predictions(preds: list[Prediction]) -> dict:
    if not preds:
        return {
            "avg_score": 0.5,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "neutral_ratio": 1.0,
            "avg_confidence": 0.0,
            "n": 0,
        }
    scores = np.array([p.score for p in preds], dtype=float)
    labels = [p.label for p in preds]
    return {
        "avg_score": float(scores.mean()),
        "bullish_ratio": float(labels.count("bullish") / len(labels)),
        "bearish_ratio": float(labels.count("bearish") / len(labels)),
        "neutral_ratio": float(labels.count("neutral") / len(labels)),
        "avg_confidence": float(np.mean([p.confidence for p in preds])),
        "n": len(preds),
    }
