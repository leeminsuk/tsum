from __future__ import annotations

import logging
from tools.news_analyzer import fetch_stock_news
from app.news_storage import push_news

logger = logging.getLogger(__name__)


def run_news_analysis() -> dict:
    logger.info("뉴스 분석 시작")
    result = fetch_stock_news()
    push_news(result)
    logger.info(f"뉴스 분석 완료: KR={result.get('kr_sentiment')} US={result.get('us_sentiment')} source={result.get('source')}")
    return result
