"""
Phase 1 + 2 — Core Detection + Event Logging for the AI Fire Suppression
System.

Real-time camera feed -> fire detection -> bounding boxes + confidence
score -> alert (console + beep) -> logged to PostgreSQL with a snapshot
of the frame (Phase 2). Detection keeps running even if the DB is down.

Usage:
    python main.py                          # webcam 0, heuristic detector, DB logging on
    python main.py --source 1               # a different webcam index
    python main.py --source clip.mp4        # run on a video file instead
    python main.py --model weights/fire.pt  # use a YOLO model if/when you have one
    python main.py --conf 0.65              # raise/lower alert sensitivity
    python main.py --no-sound               # disable the terminal beep
    python main.py --no-db                  # disable DB logging (Phase 1 behavior)
    python main.py --camera-id porch_cam    # tag events with a specific camera id
    python main.py --headless --source clip.mp4 --save out.mp4
                                             # no GUI window, write annotated video to disk

Press 'q' in the video window to quit.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2

import config
import db
from detector import HeuristicFireDetector, YoloFireDetector


def build_detector(model_path, conf_threshold, sat_min):
    if model_path:
        try:
            print(f"[info] Loading YOLO model from {model_path} ...")
            return YoloFireDetector(model_path, conf_threshold=conf_threshold)
        except Exception as e:
            print(f"[warn] Could not load YOLO model ({e}). Falling back to heuristic detector.")
    print(f"[info] Using color+motion heuristic detector (sat_min={sat_min}, no model file required).")
    return HeuristicFireDetector(sat_min=sat_min)


def draw_detections(frame, detections, conf_threshold):
    """Draws boxes for detections above threshold. Returns True if any fire is present."""
    fire_present = False
    for d in detections:
        if d.confidence < conf_threshold:
            continue
        fire_present = True
        color = (0, 0, 255)  # red, BGR
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.label.upper()} {d.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(d.y1 - th - 8, 0)
        cv2.rectangle(frame, (d.x1, text_y), (d.x1 + tw + 6, text_y + th + 8), color, -1)
        cv2.putText(
            frame, label, (d.x1 + 3, text_y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    return fire_present


class AlertManager:
    """Prints + beeps on fire detection, with a cooldown so it doesn't spam every frame."""

    def __init__(self, cooldown_seconds: float = 3.0, sound_enabled: bool = True):
        self.cooldown_seconds = cooldown_seconds
        self.sound_enabled = sound_enabled
        self._last_alert_time = 0.0
        self.alert_count = 0

    def trigger(self, max_confidence: float) -> bool:
        """Prints/beeps if outside the cooldown window. Returns True if it fired."""
        now = time.time()
        if now - self._last_alert_time < self.cooldown_seconds:
            return False
        self._last_alert_time = now
        self.alert_count += 1
        print(f"\U0001F525 [ALERT #{self.alert_count}] Fire detected — confidence {max_confidence * 100:.0f}%")
        if self.sound_enabled:
            sys.stdout.write("\a")
            sys.stdout.flush()
        return True


def main():
    parser = argparse.ArgumentParser(description="Phase 1 fire detection MVP")
    parser.add_argument("--source", default="0", help="Camera index or video file path (default: 0)")
    parser.add_argument("--model", default=None, help="Path to YOLO .pt weights (optional)")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold for drawing/alerting")
    parser.add_argument("--no-sound", action="store_true", help="Disable terminal beep on alert")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window (e.g. on a server)")
    parser.add_argument("--save", default=None, help="Optional path to write annotated output video")
    parser.add_argument("--no-db", action="store_true", help="Disable PostgreSQL event logging")
    parser.add_argument("--camera-id", default=config.CAMERA_ID, help=f"Identifier stored with each event (default: {config.CAMERA_ID})")
    parser.add_argument("--sat-min", type=int, default=130, help="Minimum HSV saturation for the heuristic detector's fire-color mask. Raise this if shiny/reflective objects cause false alerts; lower it if real flame isn't being picked up (default 130)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] Could not open video source: {source}")
        sys.exit(1)

    detector = build_detector(args.model, args.conf, args.sat_min)
    alerts = AlertManager(sound_enabled=not args.no_sound)

    db_enabled = not args.no_db
    if db_enabled:
        db_enabled = db.init_db()
        if db_enabled:
            os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
            print(f"[db] Connected. Logging events for camera_id='{args.camera_id}'.")
        else:
            print("[db] Logging disabled for this run.")
    else:
        print("[db] Logging disabled (--no-db).")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    prev_time = time.time()
    print(
        "[info] Starting detection loop. Press 'q' in the video window to quit."
        if not args.headless else
        "[info] Running headless."
    )

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[info] End of stream or camera read failed.")
                break
            frame_count += 1

            detections = detector.detect(frame)
            fire_present = draw_detections(frame, detections, args.conf)

            if fire_present:
                max_conf = max(d.confidence for d in detections if d.confidence >= args.conf)
                fired = alerts.trigger(max_conf)
                if fired and db_enabled:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    snapshot_path = os.path.join(config.SNAPSHOT_DIR, f"{args.camera_id}_{ts}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    db.log_event(
                        camera_id=args.camera_id,
                        confidence=max_conf,
                        fire_detected=True,
                        snapshot_path=snapshot_path,
                    )

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            if writer is not None:
                writer.write(frame)

            if not args.headless:
                cv2.imshow("Fire Detection - Phase 1 MVP", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[info] Processed {frame_count} frames. Total alerts: {alerts.alert_count}.")


if __name__ == "__main__":
    main()
