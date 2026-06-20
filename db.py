"""
Phase 2 — persistence layer. Logs fire detection events to PostgreSQL.

Schema (auto-created on first connect if missing):
    events(
        id              SERIAL PRIMARY KEY,
        timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
        camera_id       TEXT NOT NULL,
        confidence      REAL NOT NULL,
        fire_detected   BOOLEAN NOT NULL,
        snapshot_path   TEXT
    )

Connection failures are handled, not raised: every function here prints
a clear [db] warning and returns False/[] instead of crashing, so the
detection loop in main.py keeps running even if the database is
temporarily unreachable.
"""

from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import OperationalError

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    camera_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    fire_detected BOOLEAN NOT NULL,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_fire_detected ON events (fire_detected);
"""


def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        connect_timeout=5,
    )


def init_db() -> bool:
    """Creates the events table (and indexes) if they don't exist yet.

    Returns True if the DB is reachable and ready, False otherwise —
    callers use this to decide whether to enable logging at all.
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA)
        conn.close()
        return True
    except OperationalError as e:
        print(f"[db] Could not connect to PostgreSQL ({e.__class__.__name__}: {e}).")
        print("[db] Detection will continue, but events won't be logged.")
        return False


def log_event(
    camera_id: str,
    confidence: float,
    fire_detected: bool,
    snapshot_path: Optional[str] = None,
) -> bool:
    """Insert one event row. Returns True on success."""
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (camera_id, confidence, fire_detected, snapshot_path)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (camera_id, confidence, fire_detected, snapshot_path),
                )
        conn.close()
        return True
    except OperationalError as e:
        print(f"[db] Could not log event ({e}). Continuing without logging this one.")
        return False


def fetch_recent(limit: int = 20, fire_only: bool = False) -> List[Dict[str, Any]]:
    """Returns the most recent events as a list of dicts, newest first."""
    query = "SELECT id, timestamp, camera_id, confidence, fire_detected, snapshot_path FROM events"
    if fire_only:
        query += " WHERE fire_detected = TRUE"
    query += " ORDER BY timestamp DESC LIMIT %s"

    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        conn.close()
        return [dict(zip(cols, row)) for row in rows]
    except OperationalError as e:
        print(f"[db] Could not query events ({e}).")
        return []


def fetch_stats() -> Dict[str, Any]:
    """Returns summary counts/timestamps across all logged events."""
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (WHERE fire_detected),
                        MAX(confidence),
                        MIN(timestamp),
                        MAX(timestamp)
                    FROM events
                    """
                )
                total, fire_count, max_conf, first_ts, last_ts = cur.fetchone()
        conn.close()
        return {
            "total_events": total or 0,
            "fire_events": fire_count or 0,
            "max_confidence": max_conf,
            "first_event": first_ts,
            "last_event": last_ts,
        }
    except OperationalError as e:
        print(f"[db] Could not fetch stats ({e}).")
        return {"total_events": 0, "fire_events": 0, "max_confidence": None, "first_event": None, "last_event": None}
