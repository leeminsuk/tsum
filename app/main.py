from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import scheduler, storage
from app import news_storage
from app.settings_store import SUPPORTED_COINS, load as load_settings, save as save_settings
from app.scheduler import COINS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

# Vercel 서버리스: 인스턴스가 요청 단위로 뜨고 사라지므로 상주 스케줄러·기동 시
# 일괄 분석이 불가능하다 — vercel.json crons가 /api/cron을 대신 호출한다
IS_SERVERLESS = bool(os.getenv("VERCEL"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if IS_SERVERLESS:
        yield
        return

    cfg = load_settings()
    scheduler.start(interval_hours=cfg["interval_hours"])

    # 시작 시 데이터 없는 코인만 분석 (중복 방지)
    try:
        from app.runner import run_analysis
        import time
        for coin in COINS:
            if not storage.has_recent_signal(coin, within_hours=1.0):
                run_analysis(coin=coin)
                time.sleep(2)
    except Exception as exc:
        logger.warning(f"Initial crypto analysis error: {exc}")

    # 뉴스도 없을 때만 분석
    try:
        from app.news_runner import run_news_analysis
        from app.news_storage import get_news
        if not get_news():
            run_news_analysis()
    except Exception as exc:
        logger.warning(f"Initial news analysis skipped: {exc}")

    yield
    scheduler.stop()


app = FastAPI(title="TSUM Crypto Intel", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── Crypto API ────────────────────────────────────────────────────────────────

def _serverless_next_run() -> str:
    """vercel.json crons("0 0 * * *") 기준 다음 실행 시각 — 다음 자정 UTC."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.isoformat()


def _ensure_signal(coin: str) -> None:
    """서버리스에서 로컬 lifespan의 기동 시 분석을 대신함 — 데이터 없거나 낡으면 즉석 분석."""
    cfg = load_settings()
    if storage.has_recent_signal(coin, within_hours=float(cfg["interval_hours"])):
        return
    try:
        from app.runner import run_analysis
        run_analysis(coin=coin)
    except Exception as exc:
        logger.warning(f"Lazy analysis failed for {coin}: {exc}")


@app.get("/api/signals")
async def get_signals(coin: str | None = None):
    if IS_SERVERLESS and coin and coin.lower() in SUPPORTED_COINS:
        _ensure_signal(coin.lower())
    return storage.get_signals(coin=coin)


@app.get("/api/status")
async def get_status(coin: str | None = None):
    cfg = load_settings()
    signals = storage.get_signals(coin=coin)
    latest = signals[0] if signals else {}
    return {
        "next_run": _serverless_next_run() if IS_SERVERLESS else scheduler.get_next_run(),
        "signal_count": len(signals),
        "max_stack": storage.MAX_STACK,
        "settings": cfg,
        "latest_action": latest.get("signal", {}).get("action"),
        "latest_score": latest.get("signal", {}).get("score"),
        "coin": coin or "all",
    }


class TriggerRequest(BaseModel):
    coin: str | None = None


@app.post("/api/trigger")
async def trigger(req: TriggerRequest):
    from app.runner import run_analysis
    cfg = load_settings()
    coin = (req.coin or cfg["coin"]).lower()
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail=f"Unsupported coin: {coin}")
    try:
        result = run_analysis(coin=coin)
        return {"ok": True, "action": result["signal"]["action"], "score": result["signal"]["score"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SettingsRequest(BaseModel):
    coin: str | None = Field(None)
    interval_hours: int | None = Field(None, ge=1, le=168)
    min_whale_usd: int | None = Field(None, ge=100_000)
    lookback_hours: int | None = Field(None, ge=1, le=168)


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if "coin" in updates and updates["coin"] not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail=f"Unsupported coin. Choose from: {SUPPORTED_COINS}")
    new_cfg = save_settings(updates)
    if "interval_hours" in updates:
        scheduler.reschedule(new_cfg["interval_hours"])
    return {"ok": True, "settings": new_cfg}


@app.get("/api/settings")
async def get_settings():
    return load_settings()


# ── Market Viz API ───────────────────────────────────────────────────────────

@app.get("/api/liquidation")
async def get_liquidation(symbol: str = "BTC"):
    from tools.market_viz import fetch_liquidation_map
    return fetch_liquidation_map(symbol.upper())


@app.get("/api/bubbles")
async def get_bubbles():
    from tools.market_viz import fetch_bubble_coins
    return fetch_bubble_coins()


# ── Cron (Vercel) ─────────────────────────────────────────────────────────────

@app.get("/api/cron")
async def cron(request: Request):
    """Vercel Cron이 호출 — 전 코인 분석 + 뉴스 분석을 1회 수행."""
    secret = os.getenv("CRON_SECRET")
    if secret and request.headers.get("authorization") != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    from app.runner import run_analysis
    from app.news_runner import run_news_analysis

    results: dict[str, str] = {}
    for c in COINS:
        try:
            r = run_analysis(coin=c)
            results[c] = r["signal"]["action"]
        except Exception as exc:
            logger.error(f"Cron analysis failed for {c}: {exc}")
            results[c] = f"error: {exc}"
    try:
        run_news_analysis()
        results["news"] = "ok"
    except Exception as exc:
        logger.error(f"Cron news analysis failed: {exc}")
        results["news"] = f"error: {exc}"
    return {"ok": True, "results": results}


# ── News API ──────────────────────────────────────────────────────────────────

@app.get("/api/news")
async def get_news():
    items = news_storage.get_news()
    if IS_SERVERLESS and not items:
        # 로컬 lifespan의 "뉴스 없으면 1회 분석"과 동일 동작
        try:
            from app.news_runner import run_news_analysis
            run_news_analysis()
            items = news_storage.get_news()
        except Exception as exc:
            logger.warning(f"Lazy news analysis failed: {exc}")
    return items


@app.post("/api/news/trigger")
async def trigger_news():
    from app.news_runner import run_news_analysis
    try:
        result = run_news_analysis()
        return {"ok": True, "kr_sentiment": result.get("kr_sentiment"), "us_sentiment": result.get("us_sentiment")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
