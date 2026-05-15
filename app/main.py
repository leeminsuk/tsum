from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
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


@asynccontextmanager
async def lifespan(app: FastAPI):
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

@app.get("/api/signals")
async def get_signals(coin: str | None = None):
    return storage.get_signals(coin=coin)


@app.get("/api/status")
async def get_status(coin: str | None = None):
    cfg = load_settings()
    signals = storage.get_signals(coin=coin)
    latest = signals[0] if signals else {}
    return {
        "next_run": scheduler.get_next_run(),
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


# ── News API ──────────────────────────────────────────────────────────────────

@app.get("/api/news")
async def get_news():
    return news_storage.get_news()


@app.post("/api/news/trigger")
async def trigger_news():
    from app.news_runner import run_news_analysis
    try:
        result = run_news_analysis()
        return {"ok": True, "kr_sentiment": result.get("kr_sentiment"), "us_sentiment": result.get("us_sentiment")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
