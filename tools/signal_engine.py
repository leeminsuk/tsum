from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import statistics

from tools.config import CONFIG


@dataclass
class SignalResult:
    action: str
    score: float
    confidence: float
    rationale: list[str]
    risk_flags: list[str]
    raw_components: dict[str, Any]


class SignalEngine:
    def __init__(self) -> None:
        cfg = CONFIG.get("signal_engine", {})
        self.weights = cfg.get("weights", {"sentiment": 0.35, "fear_greed": 0.15, "whale": 0.20, "exchange_flow": 0.20, "defi": 0.10})
        self.buy_threshold = float(cfg.get("buy_threshold", 0.62))
        self.sell_threshold = float(cfg.get("sell_threshold", 0.38))
        self.conflict_penalty = float(cfg.get("conflict_penalty", 0.12))

    def combine(
        self,
        sentiment_score: float | None = None,
        social_score: float | None = None,
        fear_greed: dict | None = None,
        whale_activity: list[dict] | None = None,
        exchange_flow: dict | None = None,
        smart_money: dict | None = None,
        defi_metrics: dict | None = None,
    ) -> dict[str, Any]:
        components: dict[str, float] = {}
        rationale: list[str] = []
        risk_flags: list[str] = []

        sentiment_values = [x for x in [sentiment_score, social_score] if x is not None]
        if sentiment_values:
            components["sentiment"] = float(statistics.mean(sentiment_values))
            rationale.append(self._describe_score("뉴스/SNS 감성", components["sentiment"]))

        if fear_greed:
            fg = float(fear_greed.get("score", 0.5))
            components["fear_greed"] = fg
            rationale.append(f"공포·탐욕 지수 {fear_greed.get('value', int(fg*100))}({fear_greed.get('classification', 'Neutral')})")
            if fg > 0.82:
                risk_flags.append("탐욕 과열: 조정 리스크")
            elif fg < 0.18:
                risk_flags.append("극단적 공포: 변동성 확대 리스크")

        if whale_activity is not None:
            whale_score = self._score_whales(whale_activity)
            components["whale"] = whale_score
            rationale.append(self._describe_score("고래 트랜잭션", whale_score))

        if exchange_flow:
            flow_score = self._score_exchange_flow(exchange_flow)
            components["exchange_flow"] = flow_score
            rationale.append(f"거래소 플로우: {exchange_flow.get('signal', 'neutral')} / net_flow_usd={exchange_flow.get('net_flow_usd')}")

        if smart_money:
            sm_score = {"accumulation": 0.72, "distribution": 0.28, "neutral": 0.5}.get(str(smart_money.get("bias", "neutral")).lower(), 0.5)
            components["smart_money"] = sm_score
            rationale.append(f"스마트머니 바이어스: {smart_money.get('bias', 'neutral')}")

        if defi_metrics:
            defi_score = {"bullish": 0.65, "bearish": 0.35, "neutral": 0.5}.get(str(defi_metrics.get("signal", "neutral")).lower(), 0.5)
            components["defi"] = defi_score
            rationale.append(f"DeFi TVL 변화: {defi_metrics.get('tvl_30d_change_pct', 0):.2f}%")

        if not components:
            components["sentiment"] = 0.5
            risk_flags.append("입력 데이터 부족: 중립값으로 처리")

        weighted_sum = 0.0
        weight_total = 0.0
        for name, score in components.items():
            w = float(self.weights.get(name, self.weights.get("defi", 0.1)))
            weighted_sum += score * w
            weight_total += w
        score = weighted_sum / weight_total if weight_total else 0.5

        # Conflicting signals reduce confidence and slightly move toward neutral.
        bullish_count = sum(1 for x in components.values() if x >= 0.62)
        bearish_count = sum(1 for x in components.values() if x <= 0.38)
        conflict = bullish_count > 0 and bearish_count > 0
        if conflict:
            risk_flags.append("시그널 충돌: 심리/온체인 방향이 엇갈림")
            score = 0.5 + (score - 0.5) * (1 - self.conflict_penalty)

        action = "BUY" if score >= self.buy_threshold else "SELL" if score <= self.sell_threshold else "HOLD"
        dispersion = statistics.pstdev(list(components.values())) if len(components) > 1 else 0.0
        confidence = max(0.0, min(1.0, abs(score - 0.5) * 2 + 0.45 - dispersion - (0.15 if conflict else 0.0)))

        if action == "BUY" and conflict:
            rationale.append("종합 점수는 매수 쪽이나 충돌이 있어 분할 접근 권장")
        elif action == "SELL" and conflict:
            rationale.append("종합 점수는 매도 쪽이나 충돌이 있어 손절/헤지 기준 명확화 권장")

        return SignalResult(action, float(score), float(confidence), rationale, risk_flags, components). __dict__

    @staticmethod
    def _score_whales(txs: list[dict]) -> float:
        if not txs:
            return 0.5
        score = 0.5
        for tx in txs:
            text = " ".join(str(tx.get(k, "")).lower() for k in ["from", "to", "interpretation"])
            usd = float(tx.get("amount_usd") or 0)
            impact = min(0.08, usd / 50_000_000)
            if "exchange" in text and ("to" in tx and str(tx.get("to", "")).lower() == "exchange" or "inflow" in text):
                score -= impact
            if "cold" in text or "outflow" in text or "accumulation" in text:
                score += impact
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_exchange_flow(flow: dict) -> float:
        signal = str(flow.get("signal", "neutral")).lower()
        if signal == "bullish":
            return 0.66
        if signal == "bearish":
            return 0.34
        return 0.5

    @staticmethod
    def _describe_score(name: str, score: float) -> str:
        label = "강세" if score >= 0.62 else "약세" if score <= 0.38 else "중립"
        return f"{name}: {label}({score:.2f})"
