# 📈 TSUM Crypto Intel

> BTC · ETH · SOL · DOGE 실시간 매수/매도/관망 시그널 자동 생성 에이전트

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)](https://render.com)
[![W&B](https://img.shields.io/badge/Experiments-W%26B-FFBE00?logo=weightsandbiases)](https://wandb.ai)

---

## 목차

1. [프로젝트 소개](#1-프로젝트-소개)
2. [주요 기능](#2-주요-기능)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [폴더 구조](#4-폴더-구조)
5. [빠른 시작](#5-빠른-시작)
6. [환경 변수 설정](#6-환경-변수-설정)
7. [감성 분류 모델](#7-감성-분류-모델)
8. [파인튜닝 (W&B 연동)](#8-파인튜닝-wb-연동)
9. [API 명세](#9-api-명세)
10. [배포 (Render)](#10-배포-render)

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

---

## 2. 주요 기능

### 🔴 실시간 시그널 엔진

| 신호원 | 가중치 | 데이터 출처 |
|--------|--------|-------------|
| 기술적 지표 (RSI · MACD · 볼린저밴드) | 30% | Binance Public API |
| 공포탐욕지수 | 20% | alternative.me |
| 감성 분석 (뉴스 · SNS) | 20% | CryptoPanic · Reddit · CryptoBERT/Llama-3 |
| 거래소 유입/유출 | 15% | CoinGecko |
| DeFi TVL | 10% | DeFiLlama |
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
│  │  CryptoBERT LoRA│  │  OpenAI + Tavily          │ │
│  │  Llama-3 QLoRA  │  └───────────────────────────┘ │
│  └─────────────────┘                                 │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Supabase (PostgreSQL) │
              │  signals · news · settings│
              └──────────────────────────┘
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
│   └── inference.py            # 감성 모델 로더 (PEFT 자동 감지)
│
├── training/                   # 데이터 수집 스크립트
│   └── collect_data.py         # CryptoPanic + Reddit 수집
│
├── static/                     # 프론트엔드 (Vanilla JS)
│
├── finetune_llama3.ipynb       # 🦙 Llama-3-8B QLoRA 파인튜닝 (W&B 연동)
├── finetune_cryptobert.ipynb   # 🤖 CryptoBERT LoRA 파인튜닝 (W&B 연동)
│
├── config.yaml                 # 모델 경로 · 시그널 가중치 설정
├── requirements.txt            # Python 의존성
├── render.yaml                 # Render 배포 설정
└── .env.example                # 환경 변수 템플릿
```

---

## 5. 빠른 시작

### 사전 요구사항

- Python 3.11+
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

# 4. 서버 실행
uvicorn app.main:app --reload --port 8000
```

브라우저에서 `http://localhost:8000` 접속

---

## 6. 환경 변수 설정

`.env.example`을 복사해 `.env`를 만든 뒤 값을 채워주세요.

| 변수 | 필수 | 설명 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ 필수 | 뉴스 요약 (GPT-4o-mini) |
| `TAVILY_API_KEY` | ✅ 필수 | 뉴스 검색 |
| `SUPABASE_URL` | 권장 | 없으면 `/tmp` 파일에 저장 (재배포 시 초기화) |
| `SUPABASE_SERVICE_KEY` | 권장 | Supabase 인증 키 |
| `CRYPTOPANIC_API_KEY` | 선택 | 없으면 rule-based fallback |
| `NEWSAPI_KEY` | 선택 | 없으면 rule-based fallback |
| `ETHERSCAN_API_KEY` | 선택 | 이더리움 온체인 데이터 |
| `WHALE_ALERT_API_KEY` | 선택 | 고래 거래 탐지 |
| `COINGECKO_API_KEY` | 선택 | 없으면 무료 플랜 rate limit 적용 |
| `DEFAULT_COIN` | 선택 | 기본값 `bitcoin` |
| `INTERVAL_HOURS` | 선택 | 분석 주기 (기본값 `5`) |

---

## 7. 감성 분류 모델

시그널 엔진의 감성 분석 컴포넌트(가중치 20%)에서 사용하는 모델.  
`models/inference.py`가 `adapter_config.json` 존재 여부로 **PEFT 모델 자동 감지**한다.

| 모델 | Macro F1 | 비고 |
|------|----------|------|
| 룰 기반 (키워드 사전) | 0.41 | 모델 없을 때 자동 fallback |
| CryptoBERT + LoRA | 0.68 | `finetune_cryptobert.ipynb` 결과물 |
| **Llama-3-8B + QLoRA** | **0.74 (목표)** | `finetune_llama3.ipynb` 결과물 |

### 파인튜닝 모델 적용 방법

```bash
# 1. Colab에서 노트북 실행 후 zip 다운로드
# 2. 압축 해제
unzip finetuned_llama3_crypto.zip -d models/finetuned_llama3/

# 3. config.yaml 확인
# sentiment.model_path: ./models/finetuned_llama3
```

> Render(CPU) 환경에서 8B 모델 직접 로드는 RAM 부족으로 어렵습니다.  
> HuggingFace Hub에 업로드하거나 별도 GPU 서버로 분리하는 방식을 권장합니다.

---

## 8. 파인튜닝 (W&B 연동)

두 노트북 모두 **Weights & Biases** 실험 추적이 연동되어 있습니다.  
학습 중 loss 곡선, 클래스별 F1, confusion matrix를 대시보드에서 실시간으로 확인할 수 있습니다.

### `finetune_llama3.ipynb` — Llama-3-8B QLoRA

| 항목 | 값 |
|------|----|
| 베이스 모델 | meta-llama/Meta-Llama-3-8B-Instruct |
| 양자화 | 4-bit NF4 (QLoRA, bitsandbytes) |
| LoRA rank | r=16, alpha=32 |
| 학습 가능 파라미터 | ~41M / 8.03B (0.51%) |
| GPU | Google Colab T4 (16GB) |
| W&B 프로젝트 | `tsum-llama3-crypto-sentiment` |

### `finetune_cryptobert.ipynb` — CryptoBERT LoRA

| 항목 | 값 |
|------|----|
| 베이스 모델 | ElKulako/cryptobert (125M) |
| LoRA rank | r=16, alpha=32 |
| 배치 사이즈 | 16 (T4 기준) |
| GPU | Google Colab T4 (16GB) |
| W&B 프로젝트 | `tsum-cryptobert-crypto-sentiment` |

### W&B 대시보드에서 확인 가능한 항목

- 학습/검증 loss 곡선
- Accuracy, Macro F1, 클래스별 F1 (bearish / neutral / bullish)
- Learning rate cosine decay 커브
- 그래디언트 히스토그램
- Per-class F1 bar chart
- Confusion matrix (test set)
- 에폭별 모델 체크포인트 아티팩트

### Colab 실행 순서

1. `런타임 → 런타임 유형 변경 → T4 GPU` 선택
2. (Llama-3만) [HuggingFace 약관 동의](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) + [토큰 발급](https://huggingface.co/settings/tokens)
3. [W&B API 키 발급](https://wandb.ai/authorize)
4. 셀 순서대로 실행

---

## 9. API 명세

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

## 10. 배포 (Render)

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
4. **Deploy** 클릭

> ⚠️ Free Tier는 15분 비활성 시 슬립 상태로 전환됩니다.  
> 안정적인 운영을 위해 Starter Plan 이상을 권장합니다.

---

## 라이선스

MIT License © 2026 LeeMinSuk
