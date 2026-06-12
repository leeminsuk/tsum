from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# 새로고침마다 다른 각도로 검색해 같은 기사 반복을 피한다 (이전 요약 개수로 로테이션)
KR_QUERIES = [
    "한국 증시 코스피 코스닥 주식 오늘 뉴스",
    "코스피 외국인 기관 수급 동향 오늘",
    "한국 주식 업종 등락 반도체 2차전지 오늘",
    "한국 증시 환율 금리 영향 전망 오늘",
    "코스닥 특징주 급등 급락 오늘",
]
US_QUERIES = [
    "US stock market S&P500 NASDAQ news today",
    "US stocks sector movers tech earnings today",
    "Federal Reserve rates US market reaction today",
    "Dow Jones S&P500 winners losers today",
    "US premarket futures market outlook today",
]


def _openai_client():
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)


def _tavily_search(query: str, max_results: int = 8, exclude_urls: set[str] | None = None) -> list[dict]:
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")
    client = TavilyClient(api_key=api_key)
    result = client.search(
        query=query,
        search_depth="basic",
        topic="news",
        days=2,
        max_results=max_results,
        include_answer=False,
    )
    exclude = exclude_urls or set()
    items = []
    for r in result.get("results", []):
        if not r.get("content") or r.get("url") in exclude:
            continue
        items.append({"url": r.get("url", ""), "title": r.get("title", ""), "content": r["content"]})
    return items[:5]


def fetch_stock_news(prev_items: list[dict] | None = None) -> dict:
    """Tavily로 한국/미국 주식 뉴스 검색 후 OpenAI로 요약.

    prev_items(기존 요약 스택)를 주면 이미 인용한 기사 URL은 제외하고
    검색 쿼리도 로테이션해 매번 새로운 내용을 가져온다."""
    prev_items = prev_items or []
    seen_urls: set[str] = set()
    for it in prev_items:
        seen_urls.update(it.get("source_urls", []))
    rot = len(prev_items) % len(KR_QUERIES)

    try:
        kr_items = _tavily_search(KR_QUERIES[rot], exclude_urls=seen_urls)
        us_items = _tavily_search(US_QUERIES[rot], exclude_urls=seen_urls)
    except Exception as e:
        logger.error(f"Tavily 검색 실패: {e}")
        return _fallback_news(str(e))

    kr_text = "\n\n".join(f"[{i['title']}]\n{i['content']}" for i in kr_items) if kr_items else "데이터 없음"
    us_text = "\n\n".join(f"[{i['title']}]\n{i['content']}" for i in us_items) if us_items else "No data"

    banned_block = ""
    if prev_items:
        banned = []
        for it in prev_items[:2]:
            banned.append(it.get("kr_summary", ""))
            banned.append(it.get("us_summary", ""))
            banned.extend(it.get("kr_points", []))
            banned.extend(it.get("us_points", []))
        banned_lines = "\n".join(f"- {b}" for b in banned if b)
        banned_block = f"""
[반복 금지 목록 — 아래는 이미 사용자에게 보여준 내용입니다. 같은 사실·수치·문장을 다시 쓰지 마세요.
이번 기사들에서 아직 다루지 않은 종목·이슈·수치를 골라 새로운 요약을 만드세요.]
{banned_lines}
"""

    def _summarize(temperature: float) -> dict:
        prompt = f"""다음은 한국과 미국 주식 시장의 최신 뉴스입니다.
{banned_block}
[한국 증시 뉴스]
{kr_text}

[미국 증시 뉴스]
{us_text}

위 뉴스를 바탕으로 다음 JSON 형식으로 요약하세요. 반드시 한국어로 작성하세요:
{{
  "kr_summary": "한국 증시 현황 2-3문장 요약",
  "us_summary": "미국 증시 현황 2-3문장 요약",
  "kr_points": ["주요 포인트1", "주요 포인트2", "주요 포인트3"],
  "us_points": ["Key point 1", "Key point 2", "Key point 3"],
  "kr_sentiment": "bullish 또는 neutral 또는 bearish",
  "us_sentiment": "bullish or neutral or bearish"
}}"""
        client = _openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content)

    try:
        result = _summarize(temperature=0.7)
        # 발행 게이트(결정적): 직전 요약과 단어 집합이 과반 겹치면 한 번 더 강제 재생성
        if prev_items and _too_similar(result.get("kr_summary", ""), prev_items[0].get("kr_summary", "")):
            logger.info("직전 요약과 유사 — 재생성 1회")
            result = _summarize(temperature=1.0)
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["source"] = "tavily+openai"
        result["source_urls"] = [i["url"] for i in kr_items + us_items if i.get("url")]
        return result
    except Exception as e:
        logger.error(f"OpenAI 요약 실패: {e}")
        return _fallback_news(str(e))


def _too_similar(a: str, b: str, threshold: float = 0.5) -> bool:
    """단어 집합 자카드 유사도 — LLM 표현이 아닌 결정적 코드로 중복 판정."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) > threshold


def _fallback_news(error: str = "") -> dict:
    return {
        "kr_summary": "뉴스 데이터를 가져오지 못했습니다.",
        "us_summary": "Failed to fetch news data.",
        "kr_points": [],
        "us_points": [],
        "kr_sentiment": "neutral",
        "us_sentiment": "neutral",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "fallback",
        "error": error,
    }
