from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import scheduler, storage
from app.settings_store import SUPPORTED_COINS, load as load_settings, save as save_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_settings()
    scheduler.start(interval_hours=cfg["interval_hours"], coin=cfg["coin"])

    # Run initial analysis on cold start so the dashboard is never empty
    try:
        from app.runner import run_analysis
        run_analysis(coin=cfg["coin"])
    except Exception as exc:
        logger.warning(f"Initial analysis skipped: {exc}")

    yield
    scheduler.stop()


app = FastAPI(title="TSUM Crypto Intel", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/signals")
async def get_signals():
    return storage.get_signals()


@app.get("/api/status")
async def get_status():
    cfg = load_settings()
    return {
        "next_run": scheduler.get_next_run(),
        "signal_count": len(storage.get_signals()),
        "max_stack": storage.MAX_STACK,
        "settings": cfg,
    }


class TriggerRequest(BaseModel):
    coin: str | None = None


@app.post("/api/trigger")
async def trigger(req: TriggerRequest):
    from app.runner import run_analysis
    cfg = load_settings()
    coin = req.coin or cfg["coin"]
    try:
        result = run_analysis(coin=coin)
        return {"ok": True, "action": result["signal"]["action"], "score": result["signal"]["score"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SettingsRequest(BaseModel):
    coin: str | None = Field(None, description="bitcoin | ethereum | solana | dogecoin")
    interval_hours: int | None = Field(None, ge=1, le=168)
    min_whale_usd: int | None = Field(None, ge=100_000)
    lookback_hours: int | None = Field(None, ge=1, le=168)


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}

    if "coin" in updates and updates["coin"] not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail=f"Unsupported coin. Choose from: {SUPPORTED_COINS}")

    new_cfg = save_settings(updates)

    # Reschedule if interval or coin changed
    if "interval_hours" in updates or "coin" in updates:
        scheduler.reschedule(new_cfg["interval_hours"], new_cfg["coin"])

    return {"ok": True, "settings": new_cfg}


@app.get("/api/settings")
async def get_settings():
    return load_settings()
