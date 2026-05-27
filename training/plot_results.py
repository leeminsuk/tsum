"""
training/plot_results.py
========================
학습 완료 후 W&B에서 메트릭을 가져와 results/ 에 그래프 + 요약 저장.

사용법:
  cd /Users/chchou/tsum
  .venv/bin/python training/plot_results.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WANDB_PROJECT = "tsum-cryptobert-sentiment"
WANDB_ENTITY  = os.getenv("WANDB_ENTITY", "lms040608-")


def fetch_wandb_metrics():
    """W&B API로 최신 run의 메트릭 가져오기."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
        if not runs:
            print("⚠️  W&B run 없음")
            return None

        run = sorted(runs, key=lambda r: r.created_at, reverse=True)[0]
        print(f"Run: {run.name} ({run.id})")

        # 히스토리
        history = run.history(pandas=True)
        return run, history

    except Exception as e:
        print(f"⚠️  W&B 메트릭 로드 실패: {e}")
        return None


def plot_from_training_meta():
    """training_meta.json에서 최종 메트릭 읽어 간단 요약 생성."""
    meta_path = ROOT / "models" / "finetuned_cryptobert" / "training_meta.json"
    if not meta_path.exists():
        print(f"⚠️  {meta_path} 없음 — 학습 완료 후 실행하세요")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    print("\n=== 학습 결과 요약 ===")
    print(f"  모델   : {meta['base_model']}")
    print(f"  데이터 : {meta.get('dataset_size', '?')}개")
    print(f"  디바이스: {meta.get('device', '?')}")
    print(f"  Accuracy : {meta['eval_accuracy']:.4f}")
    print(f"  Macro F1 : {meta['eval_macro_f1']:.4f}")
    print(f"  HF Hub   : https://huggingface.co/{meta.get('hf_repo', '?')}")
    print(f"  W&B      : https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")

    # 결과 마크다운 생성
    md = f"""# TSUM CryptoBERT 학습 결과

## 모델 정보

| 항목 | 값 |
|------|----|
| 베이스 모델 | `{meta['base_model']}` |
| LoRA r / alpha | {meta['lora_r']} / {meta['lora_alpha']} |
| LoRA dropout | {meta['lora_dropout']} |
| Max length | {meta['max_length']} |
| Epochs | {meta['epochs']} |
| Batch size | {meta['batch_size']} |
| Learning rate | {meta['lr']} |
| 디바이스 | {meta.get('device', '?')} |
| 데이터셋 크기 | {meta.get('dataset_size', '?')}개 |

## 최종 평가 메트릭

| 메트릭 | 값 |
|--------|----|
| **Accuracy** | **{meta['eval_accuracy']:.4f}** |
| **Macro F1** | **{meta['eval_macro_f1']:.4f}** |

## 링크

- **HuggingFace Hub**: https://huggingface.co/{meta.get('hf_repo', '?')}
- **W&B 대시보드**: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}
"""

    out_path = RESULTS_DIR / "training_results.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\n✅ 결과 저장: {out_path}")

    # JSON 복사
    import shutil
    shutil.copy(meta_path, RESULTS_DIR / "training_meta.json")
    print(f"✅ 메타 저장: {RESULTS_DIR / 'training_meta.json'}")


def plot_loss_curve():
    """W&B 히스토리로 loss 커브 그래프 생성."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 없음 — pip install matplotlib")
        return

    result = fetch_wandb_metrics()
    if result is None:
        return

    run, history = result
    if history is None or history.empty:
        print("⚠️  히스토리 비어있음")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"TSUM CryptoBERT 학습 결과\n{run.name}", fontsize=13)

    # Train loss
    if "loss" in history.columns:
        ax = axes[0]
        train_hist = history.dropna(subset=["loss"]).copy()
        x_col = "_step" if "_step" in train_hist.columns else train_hist.index
        if "_step" in train_hist.columns:
            ax.plot(train_hist["_step"], train_hist["loss"], label="Train Loss", color="#1f77b4")
            ax.set_xlabel("Step")
        else:
            ax.plot(train_hist["loss"].values, label="Train Loss", color="#1f77b4")
            ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Eval metrics
    eval_score_cols = [c for c in history.columns
                       if "eval" in c.lower()
                       and "runtime" not in c.lower()
                       and "second" not in c.lower()
                       and "step" not in c.lower()]
    if eval_score_cols:
        ax = axes[1]
        eval_hist = history.dropna(subset=[eval_score_cols[0]]).copy()
        x_vals = eval_hist["_step"].values if "_step" in eval_hist.columns else np.arange(len(eval_hist))
        for col in eval_score_cols[:3]:
            label = col.replace("eval_", "")
            ax.plot(x_vals, eval_hist[col].values, marker="o", label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out = RESULTS_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 그래프 저장: {out}")


def main():
    print("=== TSUM 학습 결과 시각화 ===")

    # training_meta.json 기반 요약 (필수)
    plot_from_training_meta()

    # W&B 기반 그래프 (선택)
    wb_key = os.getenv("WANDB_API_KEY")
    if wb_key:
        try:
            import wandb  # noqa
            print("\nW&B 그래프 생성 중...")
            plot_loss_curve()
        except ImportError:
            print("⚠️  wandb 패키지 없음")
    else:
        print("\n⚠️  WANDB_API_KEY 없음 — W&B 그래프 건너뜀")


if __name__ == "__main__":
    main()
