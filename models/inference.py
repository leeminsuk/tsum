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

_DEFAULT_ID2LABEL = {0: "bearish", 1: "neutral", 2: "bullish"}
_LABELS = ["bearish", "neutral", "bullish"]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a crypto market sentiment analyzer. "
    "Classify the sentiment of the given text as exactly one of: "
    "bearish, neutral, bullish. Reply with only the label word."
)


@dataclass
class Prediction:
    label: str
    score: float        # 0 bearish -> 1 bullish
    confidence: float
    probabilities: dict[str, float]


class CryptoSentimentModel:
    """감성 분류 모델 래퍼.

    우선순위:
      1. Generation 모델 - Qwen2.5-14B unsloth LoRA (Qwen 계열 adapter_config.json)
      2. PEFT 분류 모델  - 구형 Llama-3 / CryptoBERT LoRA (SeqCls 헤드)
      3. 표준 pipeline   - ElKulako/cryptobert 등
      4. 룰 기반 fallback

    사용법:
      model = CryptoSentimentModel(model_path="./models/finetuned_qwen25")
      pred  = model.predict("Bitcoin ETF inflows surge ...")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        base_model: str = "unsloth/Qwen2.5-14B-bnb-4bit",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 5,
    ) -> None:
        self.model_path = str(model_path) if model_path else None
        self.base_model = base_model
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

        self._pipeline = None
        self._peft_model = None
        self._peft_tokenizer = None
        self._peft_device: str = "cpu"
        self._gen_model = None
        self._gen_tokenizer = None
        self._gen_device: str = "cpu"
        self._id2label: dict[int, str] = dict(_DEFAULT_ID2LABEL)
        self._load_error: str | None = None

        self._try_load_transformer()

    @property
    def using_transformer(self) -> bool:
        return any([self._pipeline, self._peft_model, self._gen_model])

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
            is_peft = Path(candidate).is_dir() and (Path(candidate) / "adapter_config.json").exists()
            if is_peft:
                if self._try_load_generation(candidate):
                    return
                if self._try_load_peft(candidate):
                    return
            else:
                if self._try_load_pipeline(candidate):
                    return

    def _try_load_generation(self, model_path: str) -> bool:
        """Qwen2.5-14B unsloth LoRA - generation 방식 추론 모델 로드.
        adapter_config 의 base_model 이 Qwen 계열인 경우에만 사용합니다.
        """
        try:
            import json as _json
            adapter_cfg = _json.loads((Path(model_path) / "adapter_config.json").read_text())
            base_name = adapter_cfg.get("base_model_name_or_path", "")
            if "qwen" not in base_name.lower() and "qwen" not in str(model_path).lower():
                return False
        except Exception:
            return False

        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            peft_cfg = PeftConfig.from_pretrained(model_path)
            base = peft_cfg.base_model_name_or_path
            device = "cuda" if torch.cuda.is_available() else "cpu"

            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                )
                base_m = AutoModelForCausalLM.from_pretrained(
                    base, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True)
            except Exception:
                dtype = torch.float16 if device == "cuda" else torch.float32
                base_m = AutoModelForCausalLM.from_pretrained(
                    base, torch_dtype=dtype, trust_remote_code=True).to(device)

            self._gen_model = PeftModel.from_pretrained(base_m, model_path)
            self._gen_model.eval()
            self._gen_tokenizer = tok
            self._gen_device = device
            self._load_error = None
            print(f"[info] Generation 모델 로드 완료 (Qwen2.5): {model_path}  (device={device})")
            return True

        except Exception as exc:
            self._load_error = f"Generation load failed for {model_path}: {exc}"
            self._gen_model = None
            self._gen_tokenizer = None
            return False

    def _try_load_peft(self, model_path: str) -> bool:
        """구형 PEFT 분류 헤드 모델 로드 (SeqCls 헤드)."""
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            peft_cfg = PeftConfig.from_pretrained(model_path)
            base = peft_cfg.base_model_name_or_path
            device = "cuda" if torch.cuda.is_available() else "cpu"

            tok = AutoTokenizer.from_pretrained(base)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id
            tok.padding_side = "right"

            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                )
                base_m = AutoModelForSequenceClassification.from_pretrained(
                    base, num_labels=3, quantization_config=bnb_cfg, device_map="auto")
            except Exception:
                dtype = torch.float16 if device == "cuda" else torch.float32
                base_m = AutoModelForSequenceClassification.from_pretrained(
                    base, num_labels=3, torch_dtype=dtype).to(device)

            self._peft_model = PeftModel.from_pretrained(base_m, model_path)
            self._peft_model.eval()
            self._peft_tokenizer = tok
            self._peft_device = device

            cfg_id2label = getattr(base_m.config, "id2label", None)
            if cfg_id2label:
                self._id2label = {int(k): v.lower() for k, v in cfg_id2label.items()}
            self._load_error = None
            print(f"[info] PEFT 분류 모델 로드 완료: {model_path}  (device={device})")
            return True

        except Exception as exc:
            self._load_error = f"PEFT load failed for {model_path}: {exc}"
            self._peft_model = None
            self._peft_tokenizer = None
            return False

    def _try_load_pipeline(self, candidate: str) -> bool:
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification", model=candidate, tokenizer=candidate,
                top_k=None, truncation=True, max_length=256)
            self._load_error = None
            return True
        except Exception as exc:
            self._load_error = f"Could not load {candidate}: {exc}"
            self._pipeline = None
            return False

    # ── 추론 ──────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Prediction:
        if self._gen_model is not None:
            return self._predict_generation(text)
        if self._peft_model is not None:
            return self._predict_peft(text)
        if self._pipeline is not None:
            return self._predict_pipeline(text)
        return self._predict_rules(text)

    def predict_many(self, texts: Iterable[str]) -> list[Prediction]:
        texts = list(texts)
        if self._gen_model is not None:
            return self._predict_generation_batch(texts)
        if self._peft_model is not None:
            return self._predict_peft_batch(texts)
        return [self.predict(t) for t in texts]

    # ── Generation 추론 (Qwen2.5-14B) ────────────────────────────────────

    def _make_prompt(self, text: str) -> str:
        return (
            f"<|im_start|>system
{self.system_prompt}<|im_end|>
"
            f"<|im_start|>user
{text}<|im_end|>
"
            f"<|im_start|>assistant
"
        )

    def _parse_label(self, response: str) -> str:
        word = response.strip().lower().split()[0] if response.strip() else ""
        return word if word in _LABELS else "neutral"

    def _label_to_prediction(self, label: str) -> Prediction:
        probs = {"bearish": 0.1, "neutral": 0.1, "bullish": 0.1}
        probs[label] = 0.80
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        score = probs["bullish"] + 0.5 * probs["neutral"]
        return Prediction(label, float(score), 0.80, probs)

    def _predict_generation(self, text: str) -> Prediction:
        import torch
        inputs = self._gen_tokenizer(
            self._make_prompt(text), return_tensors="pt", truncation=True, max_length=256
        ).to(self._gen_device)
        with torch.no_grad():
            out = self._gen_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self._gen_tokenizer.eos_token_id)
        resp = self._gen_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._label_to_prediction(self._parse_label(resp))

    def _predict_generation_batch(self, texts: list[str], batch_size: int = 8) -> list[Prediction]:
        import torch
        results: list[Prediction] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._gen_tokenizer(
                [self._make_prompt(t) for t in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self._gen_device)
            with torch.no_grad():
                out = self._gen_model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self._gen_tokenizer.eos_token_id)
            new_tok = out[:, enc["input_ids"].shape[1]:]
            for resp in self._gen_tokenizer.batch_decode(new_tok, skip_special_tokens=True):
                results.append(self._label_to_prediction(self._parse_label(resp)))
        return results

    # ── PEFT 분류 추론 (구형) ──────────────────────────────────────────────

    def _predict_peft(self, text: str) -> Prediction:
        import torch
        inputs = self._peft_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=192, padding=True
        ).to(self._peft_device)
        with torch.no_grad():
            logits = self._peft_model(**inputs).logits[0]
        return self._probs_to_prediction(torch.softmax(logits, dim=-1).cpu().float().tolist())

    def _predict_peft_batch(self, texts: list[str]) -> list[Prediction]:
        import torch
        inputs = self._peft_tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=192, padding=True
        ).to(self._peft_device)
        with torch.no_grad():
            logits = self._peft_model(**inputs).logits
        return [self._probs_to_prediction(p) for p in torch.softmax(logits, dim=-1).cpu().float().tolist()]

    def _probs_to_prediction(self, probs: list[float]) -> Prediction:
        b = probs[self._label_id("bearish")]
        n = probs[self._label_id("neutral")]
        u = probs[self._label_id("bullish")]
        t = b + n + u or 1.0
        b, n, u = b / t, n / t, u / t
        score = u + 0.5 * n
        label = "bullish" if score >= 0.58 else "bearish" if score <= 0.42 else "neutral"
        return Prediction(label, float(score), float(max(b, n, u)),
                          {"bearish": float(b), "neutral": float(n), "bullish": float(u)})

    def _label_id(self, label_name: str) -> int:
        for k, v in self._id2label.items():
            if v == label_name:
                return k
        return {"bearish": 0, "neutral": 1, "bullish": 2}[label_name]

    # ── Pipeline 추론 ──────────────────────────────────────────────────────

    def _predict_pipeline(self, text: str) -> Prediction:
        raw = self._pipeline(text)
        items = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        probs = {str(x["label"]).lower(): float(x["score"]) for x in items}
        b = self._pick_prob(probs, ["bearish", "label_0", "negative", "neg"])
        n = self._pick_prob(probs, ["neutral", "label_1", "neu"])
        u = self._pick_prob(probs, ["bullish", "label_2", "positive", "pos"])
        t = b + n + u
        if t <= 0:
            return self._predict_rules(text)
        b, n, u = b / t, n / t, u / t
        score = u + 0.5 * n
        label = "bullish" if score >= 0.58 else "bearish" if score <= 0.42 else "neutral"
        return Prediction(label, float(score), float(max(b, n, u)),
                          {"bearish": b, "neutral": n, "bullish": u})

    @staticmethod
    def _pick_prob(probs: dict[str, float], keys: list[str]) -> float:
        for key in keys:
            if key in probs:
                return probs[key]
        return 0.0

    # ── 룰 기반 fallback ──────────────────────────────────────────────────

    def _predict_rules(self, text: str) -> Prediction:
        tokens = set(re.findall(r"[A-Za-z가-힣]+", text.lower()))
        bull = len(tokens & BULLISH_WORDS)
        bear = len(tokens & BEARISH_WORDS)
        score = 1.0 / (1.0 + math.exp(-(bull - bear) / 2.0))
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
        return Prediction(label, float(score), float(confidence), {k: v / s for k, v in probs.items()})


def aggregate_predictions(preds: list[Prediction]) -> dict:
    if not preds:
        return {"avg_score": 0.5, "bullish_ratio": 0.0, "bearish_ratio": 0.0,
                "neutral_ratio": 1.0, "avg_confidence": 0.0, "n": 0}
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
