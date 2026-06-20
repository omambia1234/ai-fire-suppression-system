-- Reference schema for Phase 2 event logging.
-- You don't need to run this manually — db.init_db() creates it
-- automatically the first time main.py connects. This file exists for
-- reference, or if you'd rather set it up by hand with psql.

CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    camera_id       TEXT NOT NULL,
    confidence      REAL NOT NULL,
    fire_detected   BOOLEAN NOT NULL,
    snapshot_path   TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_fire_detected ON events (fire_detected);
