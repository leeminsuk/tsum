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
        # technical 추가, 가중치 재조정
        self.weights = cfg.get("weights", {
            "technical":     0.30,  # RSI/MACD/BB — 실시간, 신뢰도 높음
            "fear_greed":    0.20,  # alternative.me — 실시간 무료
            "sentiment":     0.20,  # 뉴스/소셜 — API 키 있을 때 품질↑
            "exchange_flow": 0.15,  # 거래량 기반 proxy
            "defi":          0.10,  # DeFiLlama — 실시간 무료
            "whale":         0.05,  # API 키 없으면 mock이라 낮게 유지
        })
        self.buy_threshold  = float(cfg.get("buy_threshold",  0.60))
        self.sell_threshold = float(cfg.get("sell_threshold", 0.40))
        self.conflict_penalty = float(cfg.get("conflict_penalty", 0.10))

    def combine(
        self,
        sentiment_score:  float | None       = None,
        social_score:     float | None       = None,
        fear_greed:       dict  | None       = None,
        whale_activity:   list[dict] | None  = None,
        exchange_flow:    dict  | None       = None,
        smart_money:      dict  | None       = None,
        defi_metrics:     dict  | None       = None,
        technical:        dict  | None       = None,   # ← 신규
    ) -> dict[str, Any]:
        components: dict[str, float] = {}
        rationale:  list[str]        = []
        risk_flags: list[str]        = []

        # ── 기술적 지표 (최우선) ────────────────────────────────────────────
        if technical and technical.get("source") != "fallback":
            components["technical"] = float(technical["score"])
            rationale += technical.get("rationale", [])
            risk_flags += technical.get("risk_flags", [])

        # ── 공포·탐욕 ───────────────────────────────────────────────────────
        if fear_greed:
            fg = float(fear_greed.get("score", 0.5))
            components["fear_greed"] = fg
            val   = fear_greed.get("value", int(fg * 100))
            cls   = fear_greed.get("classification", "Neutral")
            rationale.append(f"공포·탐욕 지수: {val} ({cls})")
            if fg > 0.82:
                risk_flags.append("탐욕 과열: 조정 리스크")
            elif fg < 0.18:
                risk_flags.append("극단적 공포: 변동성 확대 리스크")

        # ── 감성 ────────────────────────────────────────────────────────────
        sent_vals = [x for x in [sentiment_score, social_score] if x is not None]
        if sent_vals:
            components["sentiment"] = float(statistics.mean(sent_vals))
            rationale.append(self._describe_score("뉴스/SNS 감성", components["sentiment"]))

        # ── 거래소 플로우 ────────────────────────────────────────────────────
        if exchange_flow:
            flow_score = self._score_exchange_flow(exchange_flow)
            components["exchange_flow"] = flow_score
            vol_ratio = exchange_flow.get("volume_ratio")
            vol_str   = f" / 거래량 {vol_ratio:.2f}x" if vol_ratio else ""
            rationale.append(f"거래소 플로우: {exchange_flow.get('signal', 'neutral')}{vol_str}")

        # ── DeFi ────────────────────────────────────────────────────────────
        if defi_metrics:
            defi_score = {"bullish": 0.65, "bearish": 0.35, "neutral": 0.50}.get(
                str(defi_metrics.get("signal", "neutral")).lower(), 0.50
            )
            components["defi"] = defi_score
            rationale.append(f"DeFi TVL 30일 변화: {defi_metrics.get('tvl_30d_change_pct', 0):.2f}%")

        # ── 고래 ────────────────────────────────────────────────────────────
        if whale_activity is not None:
            whale_score = self._score_whales(whale_activity)
            # mock 데이터일 때 영향 최소화 (모든 source가 mock이면 skip)
            all_mock = all(tx.get("source", "") == "mock" for tx in whale_activity) if whale_activity else True
            if not all_mock:
                components["whale"] = whale_score
                rationale.append(self._describe_score("고래 트랜잭션", whale_score))

        # ── 스마트머니 ───────────────────────────────────────────────────────
        if smart_money and smart_money.get("source") != "mock":
            sm_score = {"accumulation": 0.72, "distribution": 0.28, "neutral": 0.50}.get(
                str(smart_money.get("bias", "neutral")).lower(), 0.50
            )
            components["smart_money"] = sm_score
            rationale.append(f"스마트머니: {smart_money.get('bias', 'neutral')}")

        if not components:
            components["fear_greed"] = 0.5
            risk_flags.append("데이터 부족: 중립 처리")

        # ── 가중 평균 ────────────────────────────────────────────────────────
        weighted_sum = weight_total = 0.0
        for name, val in components.items():
            w = float(self.weights.get(name, 0.05))
            weighted_sum += val * w
            weight_total += w
        score = weighted_sum / weight_total if weight_total else 0.5

        # ── 충돌 패널티 ──────────────────────────────────────────────────────
        bullish_n = sum(1 for v in components.values() if v >= 0.60)
        bearish_n = sum(1 for v in components.values() if v <= 0.40)
        conflict  = bullish_n > 0 and bearish_n > 0
        if conflict:
            risk_flags.append("시그널 충돌: 지표들 방향이 엇갈림")
            score = 0.5 + (score - 0.5) * (1 - self.conflict_penalty)

        # ── 판단 ─────────────────────────────────────────────────────────────
        action = "BUY" if score >= self.buy_threshold else "SELL" if score <= self.sell_threshold else "HOLD"
        dispersion = statistics.pstdev(list(components.values())) if len(components) > 1 else 0.0
        confidence = max(0.0, min(1.0,
            abs(score - 0.5) * 2 + 0.45 - dispersion - (0.12 if conflict else 0.0)
        ))

        if action == "BUY" and conflict:
            rationale.append("종합 매수 우세 — 충돌로 분할 접근 권장")
        elif action == "SELL" and conflict:
            rationale.append("종합 매도 우세 — 손절/헤지 기준 명확화 권장")

        return SignalResult(
            action, float(score), float(confidence), rationale, risk_flags, components
        ).__dict__

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _score_whales(txs: list[dict]) -> float:
        if not txs:
            return 0.5
        score = 0.5
        for tx in txs:
            text   = " ".join(str(tx.get(k, "")).lower() for k in ["from", "to", "interpretation"])
            usd    = float(tx.get("amount_usd") or 0)
            impact = min(0.08, usd / 50_000_000)
            if "exchange" in text and ("inflow" in text or str(tx.get("to","")).lower() == "exchange"):
                score -= impact
            if "cold" in text or "outflow" in text or "accumulation" in text:
                score += impact
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_exchange_flow(flow: dict) -> float:
        signal = str(flow.get("signal", "neutral")).lower()
        return {"bullish": 0.66, "bearish": 0.34}.get(signal, 0.50)

    @staticmethod
    def _describe_score(name: str, score: float) -> str:
        label = "강세" if score >= 0.60 else "약세" if score <= 0.40 else "중립"
        return f"{name}: {label} ({score:.2f})"
