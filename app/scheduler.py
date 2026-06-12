from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]

_scheduler: BackgroundScheduler | None = None


def _all_coins_job() -> None:
    from app.runner import run_analysis
    for coin in COINS:
        try:
            run_analysis(coin=coin)
            time.sleep(5)  # CoinGecko rate limit 간격 (배치 캐시 도입 후에도 여유 확보)
        except Exception as exc:
            logger.error(f"Scheduled analysis failed for {coin}: {exc}")


def _news_job() -> None:
    from app.news_runner import run_news_analysis
    try:
        run_news_analysis()
    except Exception as exc:
        logger.error(f"Scheduled news analysis failed: {exc}")


def start(interval_hours: int = 5) -> None:
    global _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    # coalesce + grace 무제한: 맥 슬립 등으로 실행 시각을 놓쳐도 깨어나면 1회만 몰아서 실행
    _scheduler.add_job(
        _all_coins_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="analysis",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=None,
        max_instances=1,
    )
    _scheduler.add_job(
        _news_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="news_analysis",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=None,
        max_instances=1,
    )
    _scheduler.start()
    job = _scheduler.get_job("analysis")
    logger.info(f"Scheduler started — interval={interval_hours}h, next={job.next_run_time if job else 'unknown'}")


def get_next_run() -> str | None:
    if _scheduler:
        job = _scheduler.get_job("analysis")
        if job and job.next_run_time:
            # 시스템 슬립 중 monotonic 클럭이 멈춰 대기 스레드가 제때 못 깨어나면
            # next_run_time이 과거에 고정된다 — 감지 시 스레드를 깨워 밀린 잡 처리·재스케줄
            if job.next_run_time <= datetime.now(timezone.utc):
                _scheduler.wakeup()
            return job.next_run_time.isoformat()
    return None


def reschedule(interval_hours: int) -> None:
    if _scheduler:
        _scheduler.reschedule_job("analysis", trigger=IntervalTrigger(hours=interval_hours))
        _scheduler.reschedule_job("news_analysis", trigger=IntervalTrigger(hours=interval_hours))


def stop() -> None:
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
