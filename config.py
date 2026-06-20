"""
Phase 2 configuration — reads DB connection settings and camera identity
from environment variables, with a .env file loaded automatically if
present (via python-dotenv, optional — env vars still work without it).

Nothing here is hardcoded credentials; copy .env.example to .env and
fill in your real values, or export the variables in your shell.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; plain env vars still work

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fire_detection")
DB_USER = os.getenv("DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CAMERA_ID = os.getenv("CAMERA_ID", "cam_0")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
