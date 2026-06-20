# Phase 1 + 2 — Core Detection + Event Logging

Real-time camera feed → fire detection → bounding boxes + confidence score
→ alert (console + beep) → **logged to PostgreSQL with a snapshot** of the
moment it fired. Detection keeps running even if the database is down.

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Create the database (one-time)

You said Postgres is already installed and running. Create a dedicated
database for this project:

```bash
createdb fire_detection
```

If that errors with something like `role "postgres" does not exist` or
asks for a password you don't know, your local Postgres is probably set
up to use your Mac username as the default superuser instead. Try:

```bash
createdb fire_detection
psql -d fire_detection -c '\conninfo'
```

The `\conninfo` output tells you exactly which user/host/port you're
actually connected as — use those values in step 3.

### 3. Configure connection details

```bash
cp .env.example .env
```

Open `.env` and fill in real values for `DB_USER` / `DB_PASSWORD` if
needed. Many local Postgres setups (Postgres.app, Homebrew) use trust
auth with no password and your Mac username as the user — if so, set:

```
DB_USER=your_mac_username
DB_PASSWORD=
```

### 4. Run it

```bash
python main.py
```

You should see `[db] Connected. Logging events for camera_id='cam_0'.`
near the top of the output. If you instead see a `[db] Could not
connect...` warning, detection still runs fine — it just means
something in `.env` doesn't match your local Postgres setup yet (see
troubleshooting below).

Press `q` in the video window to quit.

## What gets logged

Every time an alert fires (after the 3-second cooldown), one row is
written to the `events` table:

| column | meaning |
|---|---|
| `id` | auto-incrementing primary key |
| `timestamp` | when the event was logged |
| `camera_id` | from `--camera-id` (default `cam_0`) |
| `confidence` | the detector's confidence score, 0.0–1.0 |
| `fire_detected` | always `TRUE` for now — every logged row is a fire alert |
| `snapshot_path` | path to the annotated frame saved in `snapshots/` |

The table is created automatically on first connect — you don't need to
run `schema.sql` by hand unless you'd rather set it up manually with
`psql`.

## Viewing the data

No SQL required:

```bash
python query_events.py                 # last 20 events
python query_events.py --limit 50       # last 50
python query_events.py --fire-only      # only fire_detected=True rows (currently: all of them)
python query_events.py --stats          # summary counts instead of a row listing
```

Or with `psql` directly:

```bash
psql -d fire_detection -c "SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;"
```

## Useful flags

| Flag | What it does |
|---|---|
| `--source 1` | Use a different camera index, or a path to a video file |
| `--conf 0.65` | Raise the confidence threshold (fewer, more confident alerts) |
| `--no-sound` | Disable the terminal beep |
| `--no-db` | Skip DB logging entirely (Phase 1 behavior) |
| `--camera-id porch_cam` | Tag events from this run with a specific camera id |
| `--model path/to/weights.pt` | Use a YOLO model instead of the heuristic detector |
| `--headless --save out.mp4` | No GUI window — write the annotated video to disk |

## Detection approach (unchanged from Phase 1)

By default this uses a **color + motion heuristic**: HSV thresholding
for fire-like hues, ANDed with a frame-differencing motion mask so
static red/orange objects don't trigger it. It's a heuristic, not a
certified detector — expect occasional false positives on things like
warm-lit reflective objects, skin tones, or sunsets through a window.

To use a trained model instead, once you have one:

```bash
pip install ultralytics
python main.py --model path/to/fire_weights.pt
```

If the weights are missing or `ultralytics` isn't installed, it warns
and automatically falls back to the heuristic detector.

## Troubleshooting the DB connection

- **"password authentication failed"** — your `.env` password doesn't
  match. Either set the right password, or switch your local Postgres
  to trust auth for local connections (varies by how you installed it).
- **"role ... does not exist"** — `DB_USER` in `.env` doesn't match any
  Postgres role. Run `psql -d fire_detection -c '\du'` to list existing
  roles and use one of those.
- **"database ... does not exist"** — you haven't run `createdb
  fire_detection` yet, or `DB_NAME` in `.env` doesn't match what you
  created.
- **Detection runs but logging is silently off** — check the `[db]`
  line near the top of the output; it always tells you whether logging
  is active and why if it isn't.

## What's intentionally NOT here yet

- No suppression hardware / alarm integration
- No multi-camera dashboard or UI
- No automatic cleanup of old snapshots (they'll accumulate in
  `snapshots/` — fine for testing, worth adding before any real
  deployment)

Those are later phases.

## Files

- `detector.py` — `HeuristicFireDetector` (color+motion) and
  `YoloFireDetector` (model hook), behind a shared `Detection` interface.
- `main.py` — capture loop, drawing, alert cooldown, DB logging, CLI.
- `config.py` — reads DB/camera settings from environment variables / `.env`.
- `db.py` — PostgreSQL connection, schema creation, insert/query helpers.
- `query_events.py` — CLI to view logged events without writing SQL.
- `schema.sql` — reference schema (auto-applied by `db.py`, not required to run by hand).
- `.env.example` — template for your local DB connection settings.

