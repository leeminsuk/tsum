# 📈 TSUM Crypto Intel

> BTC · ETH · SOL · DOGE 실시간 매수/매도/관망 시그널 자동 생성 에이전트

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)](https://render.com)
[![W&B](https://img.shields.io/badge/Experiments-W%26B-FFBE00?logo=weightsandbiases)](https://wandb.ai/lms040608-/tsum-cryptobert-sentiment)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/space1637/tsum-cryptobert-sentiment)

---

## 목차

1. [프로젝트 소개](#1-프로젝트-소개)
2. [주요 기능](#2-주요-기능)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [폴더 구조](#4-폴더-구조)
5. [빠른 시작](#5-빠른-시작)
6. [환경 변수 설정](#6-환경-변수-설정)
7. [감성 분류 모델](#7-감성-분류-모델)
8. [파인튜닝 — Mac 로컬 (권장)](#8-파인튜닝--mac-로컬-권장)
9. [파인튜닝 — Kaggle (대규모 모델)](#9-파인튜닝--kaggle-대규모-모델)
10. [API 명세](#10-api-명세)
11. [배포 (Render)](#11-배포-render)

---

## 1. 프로젝트 소개

TSUM은 **암호화폐 인텔리전스 에이전트**다.  
5가지 신호원(기술적 지표·공포탐욕지수·감성 분석·온체인·DeFi)을 가중 합산해  
`BUY / HOLD / SELL` 시그널을 5시간마다 자동 생성하고, 웹 대시보드에서 시각화한다.

| 항목 | 내용 |
|------|------|
| 지원 코인 | BTC, ETH, SOL, DOGE |
| 시그널 주기 | 5시간 (설정 변경 가능) |
| 배포 환경 | Render (Singapore · Free Tier) |
| 영구 저장소 | Supabase PostgreSQL |
| 뉴스 요약 | OpenAI GPT-4o-mini + Tavily |
| 감성 모델 | **CryptoBERT + LoRA** (Mac M5 Pro 로컬 학습, HF Hub 배포) |

---

## 2. 주요 기능

### 🔴 실시간 시그널 엔진

| 신호원 | 가중치 | 데이터 출처 |
|--------|--------|-------------|
| 기술적 지표 (RSI · MACD · 볼린저밴드) | 30% | Binance Public API |
| 공포탐욕지수 | 15% | alternative.me |
| 감성 분석 (뉴스 · SNS) | **35%** | **CryptoBERT + LoRA** (로컬 fine-tuned) |
| 거래소 유입/유출 | 10% | CoinGecko |
| DeFi TVL | 5% | DeFiLlama |
| 고래 거래 | 5% | Whale Alert |

- 매수: 종합 점수 ≥ 0.62 / 매도: ≤ 0.38 / 그 외: 관망
- 신호 충돌 시 패널티(-0.12) 적용

### 📰 뉴스 분석

- 5시간마다 BTC·ETH·SOL·DOGE 관련 뉴스 자동 수집
- GPT-4o-mini + Tavily로 요약 및 감성 판별
- 웹 대시보드에서 코인별 최신 뉴스 카드 제공

### 📊 시각화

- 청산 맵 (Binance OI 기반 추정)
- 버블 맵 (CoinGecko 시가총액 기준)
- 코인별 개별 상세 페이지

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   웹 브라우저 (대시보드)               │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│              FastAPI 서버 (Render · Singapore)        │
│                                                      │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Scheduler   │  │  REST API   │  │  Static    │  │
│  │ (5h 주기)    │  │  /api/*     │  │  (HTML/JS) │  │
│  └──────┬───────┘  └─────────────┘  └────────────┘  │
│         │                                            │
│  ┌──────▼──────────────────────────────────────┐    │
│  │              Signal Engine                  │    │
│  │  Technical · FearGreed · Sentiment ·        │    │
│  │  ExchangeFlow · DeFi · Whale                │    │
│  └──────┬──────────────────────────────────────┘    │
│         │                                            │
│  ┌──────▼──────────┐  ┌───────────────────────────┐ │
│  │  Sentiment Model│  │  News Analyzer            │ │
│  │  CryptoBERT+LoRA│  │  OpenAI + Tavily          │ │
│  │  (HF Hub 로드)  │  └───────────────────────────┘ │
│  └─────────────────┘                                 │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Supabase (PostgreSQL) │
              │  signals · news · settings│
              └──────────────────────────┘
```

**학습 파이프라인 (별도)**

```
Mac M5 Pro (MPS)
  └── training/train_mac.py
        ├── HuggingFace 공개 데이터셋 (~54K)
        ├── CryptoBERT + LoRA (r=16, alpha=32)
        ├── W&B 추적 → wandb.ai/lms040608-/tsum-cryptobert-sentiment
        └── HF Hub 업로드 → space1637/tsum-cryptobert-sentiment
```

---

## 4. 폴더 구조

```
tsum/
├── app/                        # FastAPI 애플리케이션
│   ├── main.py                 # 라우터 + 서버 진입점
│   ├── runner.py               # 단일 코인 분석 실행기
│   ├── news_runner.py          # 뉴스 분석 실행기
│   ├── scheduler.py            # APScheduler (5h 주기)
│   ├── storage.py              # 시그널 저장/조회 (Supabase/파일)
│   ├── news_storage.py         # 뉴스 저장/조회
│   └── settings_store.py       # 런타임 설정 관리
│
├── tools/                      # 신호원별 데이터 수집 모듈
│   ├── signal_engine.py        # 가중 합산 → BUY/HOLD/SELL
│   ├── technical.py            # RSI · MACD · 볼린저밴드
│   ├── sentiment.py            # 뉴스/SNS 감성 분석
│   ├── onchain.py              # 고래 거래 · 거래소 유입
│   ├── market_viz.py           # 청산맵 · 버블맵
│   ├── news_analyzer.py        # GPT-4o-mini + Tavily
│   ├── config.py               # config.yaml 로더
│   └── http.py                 # 공통 HTTP 클라이언트
│
├── models/
│   ├── inference.py            # 감성 모델 로더 (rule-based fallback 포함)
│   └── finetuned_cryptobert/   # LoRA 어댑터 (학습 후 생성 / gitignore)
│
├── training/
│   ├── train_mac.py            # 🍎 Mac(M5 Pro) MPS 로컬 학습 스크립트
│   └── collect_data.py         # CryptoPanic + Reddit 커스텀 데이터 수집
│
├── static/                     # 프론트엔드 (Vanilla JS)
│
├── finetune_cryptobert.ipynb   # CryptoBERT Colab/Kaggle 학습 노트북 (레거시)
├── finetune_qwen25.ipynb       # Qwen2.5-14B unsloth (HF Hub 데이터, Kaggle 전용)
├── finetune_qwen25_custom.ipynb  # Qwen2.5-14B unsloth (커스텀 데이터, Kaggle 전용)
│
├── config.yaml                 # 모델 경로 · 시그널 가중치 설정
├── requirements.txt            # Python 의존성
├── render.yaml                 # Render 배포 설정
└── .env.example                # 환경 변수 템플릿
```

---

## 5. 빠른 시작

### 사전 요구사항

- Python 3.11+ (권장: 3.14 — `.venv` 이미 포함)
- OpenAI API 키 (뉴스 요약 필수)
- Tavily API 키 (뉴스 수집 필수)

### 로컬 실행

```bash
# 1. 레포 클론
git clone https://github.com/leeminsuk/tsum.git
cd tsum

# 2. 가상환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 API 키 입력

# 4. (선택) 감성 모델 파인튜닝
python training/train_mac.py     # Mac M5 Pro 기준 약 20~25분

# 5. 서버 실행
uvicorn app.main:app --reload --port 8000
```

브라우저에서 `http://localhost:8000` 접속

> 💡 **모델 없이도 실행 가능** — 감성 모델 미로드 시 rule-based fallback 자동 적용

---

## 6. 환경 변수 설정

`.env.example`을 복사해 `.env`를 만든 뒤 값을 채워주세요.

| 변수 | 필수 | 설명 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ 필수 | 뉴스 요약 (GPT-4o-mini) |
| `TAVILY_API_KEY` | ✅ 필수 | 뉴스 검색 |
| `HF_TOKEN` | 파인튜닝 시 | HuggingFace 토큰 (모델 업로드) |
| `HF_USERNAME` | 파인튜닝 시 | HuggingFace 사용자 ID |
| `WANDB_API_KEY` | 파인튜닝 시 | W&B 실험 추적 |
| `SUPABASE_URL` | 권장 | 없으면 `/tmp` 파일에 저장 (재배포 시 초기화) |
| `SUPABASE_SERVICE_KEY` | 권장 | Supabase 인증 키 |
| `CRYPTOPANIC_API_KEY` | 선택 | 커스텀 데이터 수집용 |
| `COINGECKO_API_KEY` | 선택 | 없으면 무료 플랜 rate limit 적용 |
| `DEFAULT_COIN` | 선택 | 기본값 `bitcoin` |
| `INTERVAL_HOURS` | 선택 | 분석 주기 (기본값 `5`) |

---

## 7. 감성 분류 모델

**분류 클래스**: `bearish` / `neutral` / `bullish`

| 모델 | Macro F1 | 비고 |
|------|----------|------|
| 룰 기반 (키워드 사전) | 0.41 | fallback — 모델 미로드 시 자동 |
| **CryptoBERT + LoRA** | **0.68+** | **현재 운영** — Mac MPS 로컬 학습 |
| Qwen2.5-14B + unsloth | 0.76+ (목표) | Kaggle 2xT4 전용 |

### 모델 로드 우선순위 (`models/inference.py`)

```
models/finetuned_cryptobert/ 존재 → PEFT pipeline 로드
  ↓ 없으면
LOAD_BASE_MODEL=true → HuggingFace Hub에서 자동 다운로드
  ↓ 없으면
rule-based fallback (키워드 사전)
```

### HuggingFace Hub 모델 사용

```bash
# .env에 추가
LOAD_BASE_MODEL=true
MODEL_PATH=space1637/tsum-cryptobert-sentiment
```

또는 로컬 학습 후 직접 사용:

```bash
python training/train_mac.py  # → models/finetuned_cryptobert/ 에 저장됨
```

---

## 8. 파인튜닝 — Mac 로컬 (권장)

> **CryptoBERT (~110M params)** × Apple Silicon MPS 가속  
> API 키 없이 HuggingFace 공개 데이터셋만으로 학습 완료

### 학습 환경

| 항목 | 값 |
|------|----|
| 디바이스 | Apple M5 Pro (MPS) |
| 모델 | ElKulako/cryptobert |
| LoRA rank | r=16, alpha=32 |
| 데이터 | financial-tweets-crypto(~48K) + twitter-financial-news(~9.5K) = 54K |
| 학습 방식 | WeightedCE + LoRA (SEQ_CLS) |
| W&B 프로젝트 | `tsum-cryptobert-sentiment` |
| HF Hub | [`space1637/tsum-cryptobert-sentiment`](https://huggingface.co/space1637/tsum-cryptobert-sentiment) |
| 소요 시간 | M5 Pro 24GB 기준 약 20~25분 |

### 실행

```bash
# 1. ML 의존성 추가 설치
pip install torch transformers peft accelerate datasets scikit-learn pandas wandb huggingface_hub

# 2. .env에 추가
HF_TOKEN=hf_...
HF_USERNAME=your-hf-username
WANDB_API_KEY=wandb_v1_...

# 3. 학습 실행
python training/train_mac.py
```

### W&B 대시보드에서 확인 가능한 항목

- Train / Eval loss 곡선 (epoch별)
- Accuracy, Macro F1, 클래스별 F1 (bearish / neutral / bullish)
- 그래디언트 히스토그램
- 모델 체크포인트 아티팩트

[→ W&B 대시보드 보기](https://wandb.ai/lms040608-/tsum-cryptobert-sentiment)

---

## 9. 파인튜닝 — Kaggle (대규모 모델)

> **Qwen2.5-14B unsloth** — 14B 모델 4-bit 양자화 + Instruction Tuning  
> Kaggle 2xT4 GPU(각 16GB) 환경 필요

### 노트북 목록

| 노트북 | 데이터 | W&B 프로젝트 | 소요 |
|--------|--------|-------------|------|
| `finetune_qwen25.ipynb` | HF Hub (~57K) | `tsum-qwen25-crypto-sentiment` | ~1.5~2h |
| `finetune_qwen25_custom.ipynb` | 커스텀 CSV | `tsum-qwen25-custom-sentiment` | ~30분 |

### Kaggle 실행 순서

1. Kaggle → 새 Notebook → 파일 업로드
2. Accelerator: **GPU T4 x2** 선택
3. [W&B API 키 발급](https://wandb.ai/authorize)
4. 셀 순서대로 실행

### 커스텀 데이터 수집 (선택)

```bash
python training/collect_data.py \
  --coins bitcoin ethereum solana dogecoin \
  --days 60 \
  --output data/crypto_news_labeled.csv
```

필수 API: `CRYPTOPANIC_API_KEY` (무료, cryptopanic.com)

---

## 10. API 명세

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 대시보드 (HTML) |
| `GET` | `/api/signals` | 전체 코인 최신 시그널 조회 |
| `GET` | `/api/status` | 서버 상태 및 설정 조회 |
| `POST` | `/api/trigger` | 특정 코인 즉시 분석 실행 |
| `GET` | `/api/settings` | 현재 설정 조회 |
| `POST` | `/api/settings` | 설정 변경 (코인 · 주기 · 고래 기준액) |
| `GET` | `/api/liquidation` | 청산 맵 데이터 |
| `GET` | `/api/bubbles` | 버블 맵 데이터 |
| `GET` | `/api/news` | 최신 뉴스 요약 목록 |
| `POST` | `/api/news/trigger` | 뉴스 즉시 분석 실행 |

---

## 11. 배포 (Render)

`render.yaml`이 포함되어 있어 Render에 레포를 연결하면 자동 배포됩니다.

```yaml
services:
  - type: web
    name: tsum-crypto-intel
    env: python
    region: singapore
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 배포 절차

1. [Render 대시보드](https://dashboard.render.com) → **New Web Service**
2. GitHub 레포 연결 → `main` 브랜치 선택
3. 환경 변수 입력 (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`)
4. 감성 모델 사용 시 추가: `LOAD_BASE_MODEL=true`, `MODEL_PATH=space1637/tsum-cryptobert-sentiment`
5. **Deploy** 클릭

> ⚠️ Free Tier는 15분 비활성 시 슬립 상태로 전환됩니다.  
> 안정적인 운영을 위해 Starter Plan 이상을 권장합니다.

---

## 라이선스

MIT License © 2026 LeeMinSuk
