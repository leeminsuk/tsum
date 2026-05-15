from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _openai_client():
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)


def _tavily_search(query: str, max_results: int = 5) -> list[str]:
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")
    client = TavilyClient(api_key=api_key)
    result = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
    )
    return [r.get("content", "") for r in result.get("results", []) if r.get("content")]


def fetch_stock_news() -> dict:
    """Tavily로 한국/미국 주식 뉴스 검색 후 OpenAI로 요약."""
    try:
        kr_articles = _tavily_search("한국 증시 코스피 코스닥 주식 오늘 뉴스", max_results=5)
        us_articles = _tavily_search("US stock market S&P500 NASDAQ news today", max_results=5)
    except Exception as e:
        logger.error(f"Tavily 검색 실패: {e}")
        return _fallback_news(str(e))

    kr_text = "\n\n".join(kr_articles) if kr_articles else "데이터 없음"
    us_text = "\n\n".join(us_articles) if us_articles else "No data"

    prompt = f"""다음은 한국과 미국 주식 시장의 최신 뉴스입니다.

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

    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.3,
        )
        result = json.loads(response.choices[0].message.content)
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["source"] = "tavily+openai"
        return result
    except Exception as e:
        logger.error(f"OpenAI 요약 실패: {e}")
        return _fallback_news(str(e))


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
