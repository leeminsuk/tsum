"""Vercel Blob 기반 JSON 상태 저장 — Supabase 미설정 시 서버리스 영속 백엔드.

공개 스토어의 작은 JSON 문서(시그널/뉴스/설정 스택)만 다룬다.
BLOB_READ_WRITE_TOKEN이 없으면 비활성 — 호출부가 파일 폴백으로 내려간다.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_API = "https://blob.vercel-storage.com"
_TIMEOUT = 10

# 같은 웜 인스턴스 내 60초 폴링이 매번 블롭을 왕복하지 않도록 짧게 캐시
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 20.0


def _token() -> str:
    return os.getenv("BLOB_READ_WRITE_TOKEN", "").strip()


def available() -> bool:
    return bool(_token())


def put_json(pathname: str, data: Any) -> bool:
    try:
        r = requests.put(
            f"{_API}/{pathname}",
            data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_token()}",
                "x-content-type": "application/json",
                "x-allow-overwrite": "1",
                "x-add-random-suffix": "0",
                "x-cache-control-max-age": "60",
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        _cache[pathname] = (time.monotonic(), data)
        return True
    except Exception as exc:
        logger.warning(f"Blob put failed for {pathname}: {exc}")
        return False


def get_json(pathname: str, default: Any = None) -> Any:
    hit = _cache.get(pathname)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL:
        return hit[1]
    try:
        r = requests.get(
            _API,
            params={"prefix": pathname, "limit": "10"},
            headers={"Authorization": f"Bearer {_token()}"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        blobs = [b for b in r.json().get("blobs", []) if b.get("pathname") == pathname]
        if not blobs:
            return default
        # 공개 URL은 CDN 캐시를 타므로 쿼리로 우회
        r2 = requests.get(blobs[0]["url"], params={"_": str(int(time.time()))}, timeout=_TIMEOUT)
        r2.raise_for_status()
        data = r2.json()
        _cache[pathname] = (time.monotonic(), data)
        return data
    except Exception as exc:
        logger.warning(f"Blob get failed for {pathname}: {exc}")
        return default
