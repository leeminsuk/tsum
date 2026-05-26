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

# 라벨 순서: bearish=0, neutral=1, bullish=2  (config.yaml 기준)
_DEFAULT_ID2LABEL = {0: "bearish", 1: "neutral", 2: "bullish"}


@dataclass
class Prediction:
    label: str
    score: float  # 0 bearish → 1 bullish
    confidence: float
    probabilities: dict[str, float]


class CryptoSentimentModel:
    """감성 분류 모델 래퍼.

    우선순위:
      1. PEFT(LoRA) 모델 — adapter_config.json 이 있는 경로 (Llama-3 등)
      2. 표준 파이프라인 모델 — ElKulako/cryptobert 등
      3. 룰 기반 fallback — API 키·모델 없이도 동작

    사용법:
      model = CryptoSentimentModel(model_path='./models/finetuned_llama3')
      pred  = model.predict('Bitcoin ETF inflows surge ...')
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        base_model: str = "ElKulako/cryptobert",
    ) -> None:
        self.model_path = str(model_path) if model_path else None
        self.base_model = base_model
        self._pipeline = None          # 표준 pipeline 모델
        self._peft_model = None        # PEFT/LoRA 모델
        self._peft_tokenizer = None    # PEFT 전용 토크나이저
        self._peft_device: str = "cpu"
        self._id2label: dict[int, str] = dict(_DEFAULT_ID2LABEL)
        self._load_error: str | None = None
        self._try_load_transformer()

    # ── 공개 프로퍼티 ──────────────────────────────────────────────────────

    @property
    def using_transformer(self) -> bool:
        return self._pipeline is not None or self._peft_model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ── 로딩 ──────────────────────────────────────────────────────────────

    def _try_load_transformer(self) -> None:
        candidates: list[str] = []
        if self.model_path and Path(self.model_path).exists() and any(Path(self.model_path).iterdir()):
            candidates.append(self.model_path)
        if os.getenv("LOAD_BASE_MODEL", "false").lower() == "true":
            candidates.append(self.base_model)
        if not candidates:
            self._load_error = "No local fine-tuned model found; using rule-based fallback."
            return

        for candidate in candidates:
            # ── PEFT(LoRA) 모델인지 먼저 확인 ──────────────────────────────
            if Path(candidate).is_dir() and (Path(candidate) / "adapter_config.json").exists():
                if self._try_load_peft(candidate):
                    return
            else:
                # ── 표준 pipeline 모델 ──────────────────────────────────────
                if self._try_load_pipeline(candidate):
                    return

    def _try_load_peft(self, model_path: str) -> bool:
        """PEFT/LoRA 어댑터 모델 로드 (Llama-3-8B 등).

        bitsandbytes 설치 시 4-bit QLoRA로 로드해 메모리를 절약합니다.
        GPU 없는 환경(Render CPU)에서도 동작하지만 추론이 느립니다.
        """
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            peft_cfg = PeftConfig.from_pretrained(model_path)
            base = peft_cfg.base_model_name_or_path

            # 토크나이저
            tok = AutoTokenizer.from_pretrained(base)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id
            tok.padding_side = "right"

            # 디바이스 결정
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 4-bit 양자화 시도 (bitsandbytes 없으면 fp16/fp32 로드)
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model_obj = AutoModelForSequenceClassification.from_pretrained(
                    base,
                    num_labels=3,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                )
            except Exception:
                # bitsandbytes 없거나 CPU 환경 → 일반 로드
                dtype = torch.float16 if device == "cuda" else torch.float32
                base_model_obj = AutoModelForSequenceClassification.from_pretrained(
                    base,
                    num_labels=3,
                    torch_dtype=dtype,
                ).to(device)

            # LoRA 어댑터 적용
            self._peft_model = PeftModel.from_pretrained(base_model_obj, model_path)
            self._peft_model.eval()
            self._peft_tokenizer = tok
            self._peft_device = device

            # 모델 config 에서 id2label 읽기 (있으면 우선 사용)
            cfg_id2label = getattr(base_model_obj.config, "id2label", None)
            if cfg_id2label:
                self._id2label = {int(k): v.lower() for k, v in cfg_id2label.items()}

            self._load_error = None
            print(f"[info] PEFT 모델 로드 완료: {model_path}  (device={device})")
            return True

        except Exception as exc:
            self._load_error = f"PEFT load failed for {model_path}: {exc}"
            self._peft_model = None
            self._peft_tokenizer = None
            return False

    def _try_load_pipeline(self, candidate: str) -> bool:
        """표준 transformers pipeline 로드 (CryptoBERT 등)."""
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
            return True
        except Exception as exc:
            self._load_error = f"Could not load {candidate}: {exc}"
            self._pipeline = None
            return False

    # ── 추론 ──────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Prediction:
        if self._peft_model is not None:
            return self._predict_peft(text)
        if self._pipeline is not None:
            return self._predict_pipeline(text)
        return self._predict_rules(text)

    def predict_many(self, texts: Iterable[str]) -> list[Prediction]:
        texts = list(texts)
        if self._peft_model is not None:
            return self._predict_peft_batch(texts)
        return [self.predict(t) for t in texts]

    def _predict_peft(self, text: str) -> Prediction:
        import torch
        tok = self._peft_tokenizer
        inputs = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=192,
            padding=True,
        ).to(self._peft_device)

        with torch.no_grad():
            logits = self._peft_model(**inputs).logits[0]

        probs_t = torch.softmax(logits, dim=-1).cpu().float()
        probs = probs_t.tolist()
        return self._probs_to_prediction(probs)

    def _predict_peft_batch(self, texts: list[str]) -> list[Prediction]:
        """배치 추론 (PEFT 모델, 동일 장치)."""
        import torch
        tok = self._peft_tokenizer
        inputs = tok(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=192,
            padding=True,
        ).to(self._peft_device)

        with torch.no_grad():
            logits = self._peft_model(**inputs).logits  # (B, 3)

        probs_batch = torch.softmax(logits, dim=-1).cpu().float().tolist()
        return [self._probs_to_prediction(p) for p in probs_batch]

    def _probs_to_prediction(self, probs: list[float]) -> Prediction:
        """[bearish_p, neutral_p, bullish_p] → Prediction."""
        # id2label 순서 보정
        bearish = probs[self._label_id("bearish")]
        neutral = probs[self._label_id("neutral")]
        bullish = probs[self._label_id("bullish")]

        total = bearish + neutral + bullish
        if total <= 0:
            bearish = neutral = bullish = 1 / 3

        bearish, neutral, bullish = bearish / total, neutral / total, bullish / total
        score = bullish + 0.5 * neutral
        label = "bullish" if score >= 0.58 else "bearish" if score <= 0.42 else "neutral"
        confidence = max(bearish, neutral, bullish)
        return Prediction(
            label, float(score), float(confidence),
            {"bearish": float(bearish), "neutral": float(neutral), "bullish": float(bullish)},
        )

    def _label_id(self, label_name: str) -> int:
        for k, v in self._id2label.items():
            if v == label_name:
                return k
        # fallback: 기본값
        return {"bearish": 0, "neutral": 1, "bullish": 2}[label_name]

    def _predict_pipeline(self, text: str) -> Prediction:
        raw = self._pipeline(text)
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
        return Prediction(
            label, float(score), float(confidence),
            {"bearish": bearish, "neutral": neutral, "bullish": bullish},
        )

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
