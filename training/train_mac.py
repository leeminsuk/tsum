"""
training/train_mac.py
=====================
Mac (Apple Silicon / MPS) 전용 CryptoBERT LoRA 파인튜닝 스크립트.

사용법:
  cd /Users/chchou/tsum
  .venv/bin/python training/train_mac.py

특징:
  - HuggingFace 공개 데이터셋 자동 다운로드
  - MPS(M1/M2/M3/M4/M5) 자동 감지, CPU fallback
  - W&B 실험 추적
  - 학습 완료 후 models/finetuned_cryptobert/ 저장 + HF Hub 업로드
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ── 설정 ──────────────────────────────────────────────────────────────────────

BASE_MODEL   = "ElKulako/cryptobert"
OUTPUT_DIR   = ROOT / "models" / "finetuned_cryptobert"
HF_REPO      = f"{os.getenv('HF_USERNAME', 'space1637')}/tsum-cryptobert-sentiment"
MAX_LENGTH   = 192
TEST_SIZE    = 0.1
EPOCHS       = 3
BATCH_SIZE   = 16    # MPS 기준. OOM 시 8로 낮추세요
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1

LABELS   = ["bearish", "neutral", "bullish"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ── 디바이스 감지 ────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Apple Silicon MPS 가속 사용")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ CUDA GPU 사용")
else:
    DEVICE = "cpu"
    print("⚠️  CPU 사용 (느림)")


# ── HuggingFace 로그인 ────────────────────────────────────────────────────────

def setup_hf():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print(f"✅ HuggingFace 로그인 완료 (repo: {HF_REPO})")
        except Exception as e:
            print(f"⚠️  HF 로그인 실패: {e}")
    else:
        print("⚠️  HF_TOKEN 없음 — HF Hub 업로드 건너뜀")


# ── W&B 초기화 ───────────────────────────────────────────────────────────────

def setup_wandb():
    wb_key = os.getenv("WANDB_API_KEY")
    if not wb_key:
        print("⚠️  WANDB_API_KEY 없음 — W&B 추적 건너뜀")
        return False
    try:
        import wandb
        wandb.login(key=wb_key, relogin=True)
        print("✅ W&B 로그인 완료")
        return True
    except Exception as e:
        print(f"⚠️  W&B 로그인 실패: {e}")
        return False


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_datasets() -> pd.DataFrame:
    from datasets import load_dataset

    dfs: list[pd.DataFrame] = []

    # ── A: StephanAkkerman/financial-tweets-crypto (~57K) ──────────────────
    # description 컬럼 = 트윗 본문
    # sentiment 컬럼 = 'Bullish', 'Bearish', 'Neutral', 'Bullish (99.9%)' 등
    print("\n[1/2] financial-tweets-crypto 로딩 중...")
    try:
        ds = load_dataset("StephanAkkerman/financial-tweets-crypto", split="train")
        df_a = ds.to_pandas()
        print(f"    컬럼: {list(df_a.columns)}")

        df_a = df_a[["description", "sentiment"]].rename(
            columns={"description": "text", "sentiment": "label"}
        )

        # 'Bullish (99.9%)' → 'bullish' 등 정규화
        import re as _re
        def _norm_label(s):
            if not isinstance(s, str):
                return None
            m = _re.match(r"^(Bullish|Bearish|Neutral)", s, _re.I)
            return m.group(1).lower() if m else None

        df_a["label"] = df_a["label"].apply(_norm_label)
        df_a = df_a.dropna(subset=["label"])
        print(f"    → {len(df_a):,}개\n{df_a['label'].value_counts().to_string()}")
        dfs.append(df_a)
    except Exception as e:
        print(f"    ⚠️  로드 실패: {e}")

    # ── B: zeroshot/twitter-financial-news-sentiment (~9.5K) ───────────────
    # text 컬럼 / label: 0=bearish, 1=neutral, 2=bullish
    print("\n[2/2] twitter-financial-news-sentiment 로딩 중...")
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        df_b = ds.to_pandas()
        print(f"    컬럼: {list(df_b.columns)}")

        label_map = {0: "bearish", 1: "neutral", 2: "bullish"}
        df_b["label"] = df_b["label"].map(label_map)
        df_b = df_b[["text", "label"]].dropna(subset=["label"])
        print(f"    → {len(df_b):,}개\n{df_b['label'].value_counts().to_string()}")
        dfs.append(df_b)
    except Exception as e:
        print(f"    ⚠️  로드 실패: {e}")

    # ── 로컬 CSV ─────────────────────────────────────────────────────────────
    local_csv = ROOT / "data" / "crypto_news_labeled.csv"
    if local_csv.exists():
        print(f"\n[+] 로컬 CSV: {local_csv}")
        try:
            df_local = pd.read_csv(local_csv)[["text", "label"]].dropna()
            print(f"    → {len(df_local):,}개 추가")
            dfs.append(df_local)
        except Exception as e:
            print(f"    ⚠️  실패: {e}")

    if not dfs:
        raise RuntimeError("데이터셋 로드 실패. 인터넷 연결 확인 후 재시도.")

    return pd.concat(dfs, ignore_index=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = (
        df["text"].astype(str)
        .str.replace(r"http\S+", "[URL]", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text"].str.len() >= 10].drop_duplicates("text").reset_index(drop=True)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"\n전처리 완료: {len(df):,}개")
    print(df["label"].value_counts().to_string())
    return df


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, use_wandb: bool = False) -> None:
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    set_seed(42)

    if use_wandb:
        import wandb
        run = wandb.init(
            project="tsum-cryptobert-sentiment",
            name=f"cryptobert-r{LORA_R}-lr{LR}-ep{EPOCHS}",
            config={
                "base_model": BASE_MODEL,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": ["query", "value"],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "lr_scheduler": "linear",
                "weight_decay": 0.01,
                "max_length": MAX_LENGTH,
                "test_size": TEST_SIZE,
                "labels": LABELS,
                "training_style": "LoRA+WeightedCE",
                "device": DEVICE,
                "dataset_size": len(df),
            },
        )
        os.environ["WANDB_PROJECT"] = "tsum-cryptobert-sentiment"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_WATCH"] = "gradients"

    # 토크나이저 + 데이터셋
    print(f"\n모델 로드: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # scikit-learn으로 stratified split (datasets ClassLabel 제약 우회)
    from sklearn.model_selection import train_test_split as sk_split
    df_train, df_test = sk_split(
        df[["text", "label_id"]], test_size=TEST_SIZE, random_state=42,
        stratify=df["label_id"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    train_ds = Dataset.from_pandas(df_train)
    test_ds  = Dataset.from_pandas(df_test)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tokenize,  batched=True, remove_columns=["text"])

    train_ds = train_ds.rename_column("label_id", "labels")
    test_ds  = test_ds.rename_column("label_id",  "labels")

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    tokenized = {"train": train_ds, "test": test_ds}
    print(f"Train: {len(train_ds):,}개  Test: {len(test_ds):,}개")

    # 클래스 가중치
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=df["label_id"].values)
    print(f"클래스 가중치: bearish={class_weights[0]:.3f}, neutral={class_weights[1]:.3f}, bullish={class_weights[2]:.3f}")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # 모델 + LoRA
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights_tensor)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,
        bf16=False,
        use_cpu=(DEVICE == "cpu"),
        dataloader_num_workers=0,
        report_to="wandb" if use_wandb else "none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n=== 학습 시작 (device={DEVICE}) ===")
    trainer.train()

    # 최종 평가
    print("\n=== 최종 평가 ===")
    metrics = trainer.evaluate()
    acc  = metrics["eval_accuracy"]
    f1   = metrics["eval_macro_f1"]
    print(f"  accuracy : {acc:.4f}")
    print(f"  macro_f1 : {f1:.4f}")

    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true = preds_out.label_ids
    report = classification_report(
        [ID2LABEL[i] for i in y_true],
        [ID2LABEL[i] for i in y_pred],
        labels=LABELS,
    )
    print("\n" + report)

    # W&B 최종 기록
    if use_wandb:
        import wandb
        wandb.summary.update({
            "best_accuracy": acc,
            "best_macro_f1": f1,
        })

    # 로컬 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n모델 저장 중: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    meta = {
        "base_model": BASE_MODEL,
        "hf_repo": HF_REPO,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "eval_accuracy": acc,
        "eval_macro_f1": f1,
        "labels": LABELS,
        "device": DEVICE,
        "dataset_size": len(df),
    }
    with open(OUTPUT_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ 로컬 저장 완료: {OUTPUT_DIR}")

    # HuggingFace Hub 업로드
    _push_to_hub(model, tokenizer, acc, f1, use_wandb)


def _push_to_hub(model, tokenizer, acc: float, f1: float, use_wandb: bool):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        print("⚠️  HF_TOKEN 없음 — Hub 업로드 건너뜀")
        return

    try:
        from huggingface_hub import HfApi

        print(f"\nHuggingFace Hub 업로드 중: {HF_REPO}")
        # 모델 카드 생성
        model_card = f"""---
language: en
tags:
  - crypto
  - sentiment-analysis
  - text-classification
  - lora
  - cryptobert
license: mit
---

# TSUM CryptoBERT Sentiment Classifier

암호화폐 뉴스 헤드라인 → bearish / neutral / bullish 분류 모델.

## 학습 결과
- **Accuracy**: {acc:.4f}
- **Macro F1**: {f1:.4f}

## 사용법
```python
from transformers import pipeline
clf = pipeline("text-classification", model="{HF_REPO}")
clf("Bitcoin ETF inflows surge as institutional adoption accelerates")
```

## 학습 데이터
- StephanAkkerman/financial-tweets-crypto
- SahandNZ/cryptonews-with-price-momentum-labels

## 베이스 모델
- ElKulako/cryptobert + LoRA (r={LORA_R}, alpha={LORA_ALPHA})
"""
        with open(OUTPUT_DIR / "README.md", "w") as f:
            f.write(model_card)

        model.push_to_hub(HF_REPO, token=token)
        tokenizer.push_to_hub(HF_REPO, token=token)

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(OUTPUT_DIR / "README.md"),
            path_in_repo="README.md",
            repo_id=HF_REPO,
        )
        api.upload_file(
            path_or_fileobj=str(OUTPUT_DIR / "training_meta.json"),
            path_in_repo="training_meta.json",
            repo_id=HF_REPO,
        )

        print(f"✅ HF Hub 업로드 완료: https://huggingface.co/{HF_REPO}")

        if use_wandb:
            import wandb
            wandb.summary["hf_repo"] = f"https://huggingface.co/{HF_REPO}"
            wandb.finish()

    except Exception as e:
        print(f"⚠️  HF Hub 업로드 실패: {e}")
        if use_wandb:
            import wandb
            wandb.finish()


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TSUM CryptoBERT LoRA 파인튜닝 — Mac MPS 버전")
    print(f"  device  : {DEVICE}")
    print(f"  model   : {BASE_MODEL}")
    print(f"  output  : {OUTPUT_DIR}")
    print(f"  hf_repo : {HF_REPO}")
    print("=" * 60)

    setup_hf()
    use_wandb = setup_wandb()

    # wandb 패키지 설치 확인
    if use_wandb:
        try:
            import wandb  # noqa
        except ImportError:
            print("⚠️  wandb 패키지 없음 — pip install wandb")
            use_wandb = False

    df = load_datasets()
    df = preprocess(df)

    if len(df) < 100:
        raise RuntimeError(f"데이터 부족: {len(df)}개 (최소 100개 필요)")

    train(df, use_wandb=use_wandb)


if __name__ == "__main__":
    main()
