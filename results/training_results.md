# TSUM CryptoBERT 학습 결과

## 모델 정보

| 항목 | 값 |
|------|----|
| 베이스 모델 | `ElKulako/cryptobert` |
| LoRA r / alpha | 16 / 32 |
| LoRA dropout | 0.1 |
| Max length | 192 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 0.0002 |
| 디바이스 | mps |
| 데이터셋 크기 | 54262개 |

## 최종 평가 메트릭

| 메트릭 | 값 |
|--------|----|
| **Accuracy** | **0.6659** |
| **Macro F1** | **0.6447** |

## 링크

- **HuggingFace Hub**: https://huggingface.co/space1637/tsum-cryptobert-sentiment
- **W&B 대시보드**: https://wandb.ai/lms040608-/tsum-cryptobert-sentiment
