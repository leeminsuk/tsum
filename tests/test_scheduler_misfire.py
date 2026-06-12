# 맥 슬립 복귀 후 카운트다운 멈춤(next_run_time 과거 고정) 재현·복구 테스트
# 실행: .venv/bin/python tests/test_scheduler_misfire.py
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.scheduler as sched

ran = []
sched._all_coins_job = lambda: ran.append("analysis")
sched._news_job = lambda: ran.append("news")

sched.start(interval_hours=5)
try:
    past = datetime.now(timezone.utc) - timedelta(hours=6)
    # 슬립 복귀 직후 상태 재현: 대기 스레드는 자고 있고 next_run_time만 과거
    # (job.modify는 자체적으로 wakeup을 부르므로 잡스토어를 직접 갱신해 우회)
    for job_id in ("analysis", "news_analysis"):
        job = sched._scheduler.get_job(job_id)
        job.next_run_time = past
        sched._scheduler._jobstores["default"].update_job(job)

    time.sleep(1.5)
    stale = sched._scheduler.get_job("analysis").next_run_time
    assert stale == past and not ran, f"멈춘 상태 재현 실패: next={stale}, ran={ran}"
    print(f"[1/3] 멈춘 상태 재현 OK — next_run_time {stale.isoformat()} (과거), 잡 미실행")

    returned = sched.get_next_run()  # 과거 감지 → wakeup
    assert returned is not None
    print(f"[2/3] get_next_run() 호출 (반환: {returned})")

    time.sleep(2)
    now = datetime.now(timezone.utc)
    next_rt = sched._scheduler.get_job("analysis").next_run_time
    assert ran.count("analysis") == 1, f"coalesce 1회 실행 기대, got {ran}"
    assert next_rt > now, f"next_run_time 미래 갱신 기대, got {next_rt.isoformat()}"
    print(f"[3/3] 복구 OK — 밀린 잡 1회 실행 {ran}, next_run_time {next_rt.isoformat()} (미래)")
    print("PASS")
finally:
    sched.stop()
