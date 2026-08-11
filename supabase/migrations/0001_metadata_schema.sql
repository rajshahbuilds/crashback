-- STU-46 — Supabase metadata & experiment schema (crashback).
--
-- Durable RELATIONAL project state only: dataset/feature/model versions, pipeline runs,
-- metrics, predictions, and references to Parquet artifacts. Bulk research data (daily
-- bars, wide feature/training matrices) stays in Parquet on disk — Postgres holds only
-- the provenance and pointers, never the bulk market data.
--
-- Stable linkage to Parquet: the `artifacts` table is the single registry of files;
-- every versioned entity references an artifact by id, so any dataset/model run can be
-- reconstructed from metadata + the referenced artifact + git_commit + config.
--
-- Security: RLS is enabled on every table with no policies, so the PostgREST API denies
-- anon/authenticated access; the pipeline writes via the service role (or a direct
-- Postgres connection, which bypasses RLS). No public data exposure.

-- ---------------------------------------------------------------------------
-- updated_at trigger helper
-- ---------------------------------------------------------------------------
create or replace function set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- ---------------------------------------------------------------------------
-- artifacts — registry of files produced/consumed by the pipeline (pointers only)
-- ---------------------------------------------------------------------------
create table artifacts (
  id           uuid primary key default gen_random_uuid(),
  kind         text not null check (kind in ('parquet','model','report','config','other')),
  storage      text not null default 'local'
                 check (storage in ('local','supabase_storage','s3','other')),
  path         text not null,                 -- filesystem path or object key
  uri          text,                          -- optional fully-qualified URI
  sha256       text,                          -- content hash for integrity/dedup
  size_bytes   bigint,
  row_count    bigint,
  column_schema jsonb,                        -- e.g. canonical Polars schema snapshot
  meta         jsonb not null default '{}'::jsonb,
  created_at   timestamptz not null default now()
);
comment on table artifacts is
  'Registry of Parquet/model/report files. Postgres stores the pointer + hash, not the data.';
create unique index artifacts_sha256_key on artifacts (sha256) where sha256 is not null;
create index artifacts_kind_idx on artifacts (kind);

-- ---------------------------------------------------------------------------
-- dataset_versions — versioned research datasets (e.g. events_v1)
-- ---------------------------------------------------------------------------
create table dataset_versions (
  id            uuid primary key default gen_random_uuid(),
  name          text not null,                -- e.g. 'events_v1'
  version       text not null,                -- e.g. 'v1', '2026-08-11a'
  git_commit    text,
  config        jsonb not null default '{}'::jsonb,  -- resolved config snapshot
  config_hash   text,
  row_count     bigint,
  date_start    date,                         -- event-date coverage
  date_end      date,
  artifact_id   uuid references artifacts (id) on delete set null,
  description   text,
  created_at    timestamptz not null default now(),
  unique (name, version)
);
comment on table dataset_versions is
  'One row per built research dataset version; reproducible from config + git_commit + artifact.';

-- ---------------------------------------------------------------------------
-- feature_versions — versioned feature sets / definitions
-- ---------------------------------------------------------------------------
create table feature_versions (
  id                 uuid primary key default gen_random_uuid(),
  name               text not null,           -- e.g. 'crash_day', 'recent_crash', 'market_sector'
  version            text not null,
  spec               jsonb not null default '{}'::jsonb,  -- feature list / definitions
  git_commit         text,
  dataset_version_id uuid references dataset_versions (id) on delete set null,
  artifact_id        uuid references artifacts (id) on delete set null,
  description        text,
  created_at         timestamptz not null default now(),
  unique (name, version)
);
comment on table feature_versions is 'Versioned feature families and their definitions.';

-- ---------------------------------------------------------------------------
-- pipeline_runs — a single execution of a pipeline stage
-- ---------------------------------------------------------------------------
create table pipeline_runs (
  id                 uuid primary key default gen_random_uuid(),
  stage              text not null,           -- 'ingestion','events','labels','features','dataset',...
  status             text not null default 'running'
                       check (status in ('running','succeeded','failed','cancelled')),
  git_commit         text,
  config             jsonb not null default '{}'::jsonb,
  params             jsonb not null default '{}'::jsonb,
  dataset_version_id uuid references dataset_versions (id) on delete set null,
  started_at         timestamptz not null default now(),
  finished_at        timestamptz,
  error              text,
  created_at         timestamptz not null default now()
);
comment on table pipeline_runs is 'Execution log for pipeline stages; links to the dataset it produced.';
create index pipeline_runs_stage_idx on pipeline_runs (stage, started_at desc);

-- ---------------------------------------------------------------------------
-- model_runs — a trained model (M0 base-rate .. M3 fundamentals, and beyond)
-- ---------------------------------------------------------------------------
create table model_runs (
  id                 uuid primary key default gen_random_uuid(),
  name               text not null,           -- e.g. 'model_3_lgbm'
  model_family       text not null            -- 'baseline','logistic','lightgbm'
                       check (model_family in ('baseline','logistic','lightgbm','other')),
  stage              text,                    -- 'M0','M1','M2','M3','M4'
  target             text not null,           -- e.g. 'hit_10pct_20d'
  dataset_version_id uuid references dataset_versions (id) on delete set null,
  feature_version_id uuid references feature_versions (id) on delete set null,
  hyperparams        jsonb not null default '{}'::jsonb,
  splits             jsonb not null default '{}'::jsonb,  -- train/val/test date ranges + embargo
  git_commit         text,
  config             jsonb not null default '{}'::jsonb,
  artifact_id        uuid references artifacts (id) on delete set null,  -- serialized model
  status             text not null default 'succeeded'
                       check (status in ('running','succeeded','failed','cancelled')),
  created_at         timestamptz not null default now()
);
comment on table model_runs is
  'One row per trained model; full provenance via dataset/feature version + config + git_commit.';
create index model_runs_dataset_idx on model_runs (dataset_version_id);
create index model_runs_created_idx on model_runs (created_at desc);

-- ---------------------------------------------------------------------------
-- metrics — evaluation metrics per model_run (long form; supports the target grid)
-- ---------------------------------------------------------------------------
create table metrics (
  id            uuid primary key default gen_random_uuid(),
  model_run_id  uuid not null references model_runs (id) on delete cascade,
  split         text not null                 -- 'train','validation','test','all'
                  check (split in ('train','validation','test','all')),
  metric_name   text not null,                -- 'brier','log_loss','roc_auc','pr_auc',
                                              -- 'calibration_error','top_decile_lift',...
  metric_value  double precision not null,
  horizon_days  integer,                      -- target-grid slicing (nullable)
  threshold     numeric,                      -- target-grid slicing (nullable)
  meta          jsonb not null default '{}'::jsonb,
  created_at    timestamptz not null default now()
);
comment on table metrics is 'Long-form evaluation metrics; one row per (model_run, split, metric, grid cell).';
create index metrics_model_run_idx on metrics (model_run_id, split);

-- ---------------------------------------------------------------------------
-- securities_ref — LIGHT security metadata references (NOT bulk prices)
-- ---------------------------------------------------------------------------
create table securities_ref (
  security_id    bigint primary key,          -- CRSP permno
  company_id     bigint,                      -- CRSP permco
  ticker         text,                        -- representative/latest ticker
  company_name   text,
  exchange       text,
  security_type  text,
  sic_code       integer,
  listing_date   date,
  delisting_date date,
  meta           jsonb not null default '{}'::jsonb,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);
comment on table securities_ref is
  'Light per-security reference for joins/queries. Canonical master + bars live in Parquet.';
create trigger securities_ref_set_updated_at
  before update on securities_ref
  for each row execute function set_updated_at();

-- ---------------------------------------------------------------------------
-- predictions — per-event model predictions (held-out and later live)
-- ---------------------------------------------------------------------------
create table predictions (
  id                    uuid primary key default gen_random_uuid(),
  model_run_id          uuid not null references model_runs (id) on delete cascade,
  event_id              text not null,        -- crash-event id (from the events dataset)
  security_id           bigint references securities_ref (security_id) on delete set null,
  crash_date            date,
  target                text not null,        -- e.g. 'hit_10pct_20d'
  split                 text check (split in ('train','validation','test','all')),
  predicted_probability double precision not null,
  actual_outcome        integer,              -- realized label if known (nullable)
  created_at            timestamptz not null default now(),
  unique (model_run_id, event_id, target)
);
comment on table predictions is 'Per-event predicted probabilities; keyed to a model_run and crash event.';
create index predictions_model_run_idx on predictions (model_run_id);
create index predictions_security_idx on predictions (security_id);
create index predictions_crash_date_idx on predictions (crash_date);

-- ---------------------------------------------------------------------------
-- Row-level security: enable on all tables, no policies (deny via API; service role
-- and direct Postgres connections bypass RLS). Keeps the metadata private by default.
-- ---------------------------------------------------------------------------
alter table artifacts        enable row level security;
alter table dataset_versions enable row level security;
alter table feature_versions enable row level security;
alter table pipeline_runs    enable row level security;
alter table model_runs       enable row level security;
alter table metrics          enable row level security;
alter table securities_ref   enable row level security;
alter table predictions      enable row level security;
