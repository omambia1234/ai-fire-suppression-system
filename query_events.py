"""
Phase 2 — quick CLI to inspect logged fire events without needing to
know SQL. This is what makes the "queryable data" output tangible.

Usage:
    python query_events.py                 # last 20 events, any kind
    python query_events.py --limit 50      # last 50 events
    python query_events.py --fire-only     # only fire_detected=True rows
    python query_events.py --stats         # summary counts instead of a row listing
"""

import argparse

import db


def print_table(rows):
    if not rows:
        print("No events found.")
        return

    headers = ["id", "timestamp", "camera_id", "confidence", "fire_detected", "snapshot_path"]
    display_rows = []
    for r in rows:
        conf = r["confidence"]
        display_rows.append([
            str(r["id"]),
            str(r["timestamp"]),
            str(r["camera_id"]),
            f"{conf * 100:.0f}%" if conf is not None else "-",
            str(r["fire_detected"]),
            r["snapshot_path"] or "-",
        ])

    widths = [len(h) for h in headers]
    for row in display_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt(values):
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in display_rows:
        print(fmt(row))


def main():
    parser = argparse.ArgumentParser(description="Inspect logged fire events")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to show (default 20)")
    parser.add_argument("--fire-only", action="store_true", help="Only show fire_detected=True rows")
    parser.add_argument("--stats", action="store_true", help="Show summary stats instead of a row listing")
    args = parser.parse_args()

    if args.stats:
        stats = db.fetch_stats()
        max_conf = stats["max_confidence"]
        print(f"Total events logged : {stats['total_events']}")
        print(f"Fire events          : {stats['fire_events']}")
        print(f"Highest confidence   : {max_conf * 100:.0f}%" if max_conf is not None else "Highest confidence   : -")
        print(f"First event          : {stats['first_event']}")
        print(f"Last event           : {stats['last_event']}")
        return

    rows = db.fetch_recent(limit=args.limit, fire_only=args.fire_only)
    print_table(rows)


if __name__ == "__main__":
    main()
