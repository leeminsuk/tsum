"""Vercel serverless entrypoint — 전체 FastAPI 앱을 단일 함수로 노출."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402, F401
