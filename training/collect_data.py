"""
training/collect_data.py
========================
CryptoPanic 뉴스 헤드라인 + CoinGecko 24h 가격 변동으로 자동 라벨링 데이터셋 생성.

사용법:
  python training/collect_data.py
  python training/collect_data.py --coins bitcoin ethereum solana --days 90 --output data/raw.csv

필수 API 키 (.env):
  CRYPTOPANIC_API_KEY  — https://cryptopanic.com/developers/api/  (무료)
  COINGECKO_API_KEY    — https://www.coingecko.com/en/api         (무료, 선택)

API 키 없이도 Reddit 공개 검색 + mock fallback으로 소량 수집 가능.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# repo root → tools/ 임포트 가능하게
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import requests

# ── 설정 ──────────────────────────────────────────────────────────────────────

COINGECKO_IDS = {
    "bitcoin":  "bitcoin",
    "ethereum": "ethereum",
    "solana":   "solana",
    "dogecoin": "dogecoin",
}
SYMBOLS = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "dogecoin": "DOGE"}
BULLISH_CUTOFF =  1.5   # 24h 가격 변동 % 기준
BEARISH_CUTOFF = -1.5

# CoinGecko 무료 제한 대응 (30 req/min)
COINGECKO_DELAY = 2.5   # 요청 사이 대기 (초)
CRYPTOPANIC_DELAY = 1.0

REDDIT_UA = os.getenv("REDDIT_USER_AGENT", "tsum-collector/0.1")


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, headers: dict | None = None,
         timeout: int = 15, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def _ts_to_date(ts: float | int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def label_from_change(change: float) -> str:
    if change >= BULLISH_CUTOFF:
        return "bullish"
    if change <= BEARISH_CUTOFF:
        return "bearish"
    return "neutral"


# ── CoinGecko 가격 조회 ────────────────────────────────────────────────────────

_price_cache: dict[str, float] = {}


def get_price_on_date(coin: str, date_str: str) -> float | None:
    """date_str: 'YYYY-MM-DD' → USD 종가."""
    cache_key = f"{coin}:{date_str}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    cg_id = COINGECKO_IDS.get(coin)
    if not cg_id:
        return None

    dd, mm, yyyy = date_str[8:10], date_str[5:7], date_str[:4]
    params: dict = {"date": f"{dd}-{mm}-{yyyy}", "localization": "false"}
    headers = {}
    key = os.getenv("COINGECKO_API_KEY", "")
    if key:
        headers["x-cg-demo-api-key"] = key

    try:
        data = _get(f"https://api.coingecko.com/api/v3/coins/{cg_id}/history",
                    params=params, headers=headers or None)
        price = data["market_data"]["current_price"]["usd"]
        _price_cache[cache_key] = float(price)
        time.sleep(COINGECKO_DELAY)
        return float(price)
    except Exception as e:
        print(f"  [warn] CoinGecko {coin} {date_str}: {e}")
        return None


def get_price_change_24h(coin: str, article_date: str) -> float | None:
    """기사 날짜 → +1일 가격 대비 % 변동."""
    d0 = datetime.strptime(article_date, "%Y-%m-%d")
    d1 = (d0 + timedelta(days=1)).strftime("%Y-%m-%d")

    p0 = get_price_on_date(coin, article_date)
    p1 = get_price_on_date(coin, d1)

    if p0 and p1 and p0 > 0:
        return round((p1 - p0) / p0 * 100, 4)
    return None


# ── CryptoPanic 뉴스 수집 ──────────────────────────────────────────────────────

def collect_cryptopanic(coin: str, days: int, max_pages: int = 10) -> list[dict]:
    key = os.getenv("CRYPTOPANIC_API_KEY", "")
    if not key:
        print(f"  [skip] CRYPTOPANIC_API_KEY 없음 — {coin} CryptoPanic 건너뜀")
        return []

    symbol = SYMBOLS.get(coin, coin[:3].upper())
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    articles = []
    url = "https://cryptopanic.com/api/v1/posts/"
    next_url = None

    for page in range(1, max_pages + 1):
        params = {
            "auth_token": key,
            "currencies": symbol,
            "kind": "news",
            "public": "true",
        }
        try:
            data = _get(next_url or url, params=params if not next_url else None)
        except Exception as e:
            print(f"  [warn] CryptoPanic page {page}: {e}")
            break

        for item in data.get("results", []):
            pub = item.get("published_at", "")
            try:
                ts = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except Exception:
                continue
            if ts < cutoff:
                break
            title = (item.get("title") or "").strip()
            if title and len(title) >= 15:
                articles.append({
                    "text": title,
                    "source": "cryptopanic",
                    "coin": coin,
                    "date": ts.strftime("%Y-%m-%d"),
                })
        else:
            next_url = data.get("next")
            if not next_url:
                break
            time.sleep(CRYPTOPANIC_DELAY)
            continue
        break  # cutoff 도달

    print(f"  CryptoPanic {coin}: {len(articles)}개 수집")
    return articles


# ── Reddit 공개 검색 수집 ──────────────────────────────────────────────────────

def collect_reddit(coin: str, days: int, limit: int = 100) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    posts = []
    after = None
    headers = {"User-Agent": REDDIT_UA}

    for _ in range(5):
        params: dict = {"q": f"{coin} crypto", "sort": "new", "limit": 100, "type": "link"}
        if after:
            params["after"] = after
        try:
            data = _get("https://www.reddit.com/search.json", params=params, headers=headers)
            children = data.get("data", {}).get("children", [])
        except Exception as e:
            print(f"  [warn] Reddit {coin}: {e}")
            break

        for child in children:
            d = child.get("data", {})
            ts = float(d.get("created_utc", 0))
            if ts and datetime.fromtimestamp(ts, tz=timezone.utc) < cutoff:
                break
            title = (d.get("title") or "").strip()
            if title and len(title) >= 15:
                posts.append({
                    "text": title,
                    "source": "reddit",
                    "coin": coin,
                    "date": _ts_to_date(ts) if ts else "",
                })
        else:
            after = data.get("data", {}).get("after")
            if not after:
                break
            time.sleep(1.5)
            continue
        break

    print(f"  Reddit {coin}: {len(posts)}개 수집")
    return posts


# ── 라벨링 ────────────────────────────────────────────────────────────────────

def label_articles(articles: list[dict]) -> list[dict]:
    labeled = []
    total = len(articles)
    for i, art in enumerate(articles):
        date = art.get("date", "")
        coin = art.get("coin", "bitcoin")
        if not date:
            continue
        change = get_price_change_24h(coin, date)
        if change is None:
            continue
        labeled.append({
            "text":             art["text"],
            "coin":             coin,
            "source":           art.get("source", ""),
            "date":             date,
            "price_change_24h": change,
            "label":            label_from_change(change),
            "label_id":         {"bearish": 0, "neutral": 1, "bullish": 2}[label_from_change(change)],
        })
        if (i + 1) % 20 == 0:
            print(f"  라벨링 진행: {i+1}/{total}")
    return labeled


# ── 저장 ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "coin", "source", "date", "price_change_24h", "label", "label_id"]
    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    # 중복 제거 (text 기준)
    seen = {r["text"] for r in existing}
    new_rows = [r for r in rows if r["text"] not in seen]
    all_rows = existing + new_rows

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n저장 완료: {path}  (기존 {len(existing)} + 신규 {len(new_rows)} = 총 {len(all_rows)}개)")
    labels = [r["label"] for r in all_rows]
    for lbl in ["bullish", "neutral", "bearish"]:
        print(f"  {lbl}: {labels.count(lbl)}개 ({labels.count(lbl)/max(len(labels),1)*100:.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="크립토 뉴스 데이터 수집 및 가격 변동 라벨링")
    parser.add_argument("--coins", nargs="+", default=["bitcoin"],
                        choices=list(COINGECKO_IDS.keys()),
                        help="수집할 코인 목록 (기본: bitcoin)")
    parser.add_argument("--days", type=int, default=60,
                        help="최근 N일치 데이터 수집 (기본: 60)")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "crypto_news_labeled.csv",
                        help="출력 CSV 경로")
    parser.add_argument("--skip-reddit", action="store_true",
                        help="Reddit 수집 건너뜀 (rate limit 걱정 시)")
    args = parser.parse_args()

    print(f"=== 데이터 수집 시작 ===")
    print(f"코인: {args.coins}  기간: {args.days}일  출력: {args.output}\n")

    all_articles: list[dict] = []
    for coin in args.coins:
        print(f"[{coin.upper()}] 뉴스 수집 중...")
        all_articles += collect_cryptopanic(coin, args.days)
        if not args.skip_reddit:
            all_articles += collect_reddit(coin, args.days)

    print(f"\n총 {len(all_articles)}개 기사 수집 → 가격 변동 라벨링 중...")
    labeled = label_articles(all_articles)

    if not labeled:
        print("라벨링된 데이터가 없습니다. API 키 확인 또는 --days 늘려보세요.")
        return

    save_csv(labeled, args.output)


if __name__ == "__main__":
    main()
