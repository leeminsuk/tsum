-- ============================================================
-- TSUM Crypto Intel — Supabase 테이블 초기화 SQL
-- Supabase 대시보드 > SQL Editor 에서 실행하세요
-- ============================================================

-- 1. 신호 스택 테이블
CREATE TABLE IF NOT EXISTS signals (
  id            TEXT        PRIMARY KEY,
  generated_at  TIMESTAMPTZ NOT NULL,
  coin          TEXT        NOT NULL,
  price_usd     NUMERIC,
  price_change_24h NUMERIC,
  signal        JSONB       NOT NULL DEFAULT '{}',
  summary       JSONB       NOT NULL DEFAULT '{}',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 최신순 조회 인덱스
CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals (generated_at DESC);

-- 2. 앱 설정 테이블 (단일 행)
CREATE TABLE IF NOT EXISTS app_settings (
  id    INT     PRIMARY KEY DEFAULT 1,
  data  JSONB   NOT NULL DEFAULT '{
    "coin": "bitcoin",
    "interval_hours": 5,
    "min_whale_usd": 1000000,
    "lookback_hours": 24
  }'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 기본 설정 행 삽입
INSERT INTO app_settings (id, data)
VALUES (1, '{
  "coin": "bitcoin",
  "interval_hours": 5,
  "min_whale_usd": 1000000,
  "lookback_hours": 24
}'::jsonb)
ON CONFLICT (id) DO NOTHING;

-- 3. Row Level Security (공개 읽기 / 서비스 키로만 쓰기)
ALTER TABLE signals     ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_settings ENABLE ROW LEVEL SECURITY;

-- 누구나 읽기 가능 (대시보드 API용)
CREATE POLICY "public read signals"
  ON signals FOR SELECT USING (true);

CREATE POLICY "public read settings"
  ON app_settings FOR SELECT USING (true);

-- service_role 키로만 INSERT/UPDATE/DELETE 가능 (백엔드 전용)
CREATE POLICY "service insert signals"
  ON signals FOR INSERT
  WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "service update signals"
  ON signals FOR UPDATE
  USING (auth.role() = 'service_role');

CREATE POLICY "service delete signals"
  ON signals FOR DELETE
  USING (auth.role() = 'service_role');

CREATE POLICY "service update settings"
  ON app_settings FOR ALL
  USING (auth.role() = 'service_role');
