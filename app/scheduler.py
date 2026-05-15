from __future__ import annotations

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _crypto_job(coin: str) -> None:
    from app.runner import run_analysis
    try:
        run_analysis(coin=coin)
    except Exception as exc:
        logger.error(f"Scheduled crypto analysis failed: {exc}")


def _news_job() -> None:
    from app.news_runner import run_news_analysis
    try:
        run_news_analysis()
    except Exception as exc:
        logger.error(f"Scheduled news analysis failed: {exc}")


def start(interval_hours: int = 5, coin: str = "bitcoin") -> None:
    global _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _crypto_job,
        trigger=IntervalTrigger(hours=interval_hours),
        args=[coin],
        id="analysis",
        replace_existing=True,
    )
    _scheduler.add_job(
        _news_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="news_analysis",
        replace_existing=True,
    )
    _scheduler.start()
    job = _scheduler.get_job("analysis")
    logger.info(f"Scheduler started — interval={interval_hours}h, next={job.next_run_time if job else 'unknown'}")


def get_next_run() -> str | None:
    if _scheduler:
        job = _scheduler.get_job("analysis")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
    return None


def reschedule(interval_hours: int, coin: str) -> None:
    if _scheduler:
        _scheduler.reschedule_job(
            "analysis",
            trigger=IntervalTrigger(hours=interval_hours),
            args=[coin],
        )
        _scheduler.modify_job("analysis", args=[coin])
        _scheduler.reschedule_job(
            "news_analysis",
            trigger=IntervalTrigger(hours=interval_hours),
        )


def stop() -> None:
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
