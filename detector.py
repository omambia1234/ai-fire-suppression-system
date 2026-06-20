"""
Fire detection backends for the AI Fire Suppression System — Phase 1.

Two detectors implementing a shared interface:
  - HeuristicFireDetector: color (HSV) + motion based, no model required.
    Works the moment you plug in a webcam.
  - YoloFireDetector: drop-in hook for a YOLO model (e.g. an Ultralytics
    .pt file trained on fire/smoke). Falls back gracefully in main.py if
    ultralytics isn't installed or the weights file can't be loaded.

Both return a list of Detection objects: (x1, y1, x2, y2, confidence, label).
This shared shape is what lets main.py treat either backend identically —
swapping detectors later doesn't require touching the capture loop.
"""

from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "fire"


class FireDetector:
    """Base interface. Subclasses implement detect()."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class HeuristicFireDetector(FireDetector):
    """
    Color + motion based fire detector.

    Logic:
      1. Threshold the frame in HSV for fire-like hues (red/orange/yellow,
         high saturation, high brightness).
      2. AND that color mask with a motion mask (frame differencing) to
         reject static red/orange objects — clothing, signs, brake lights,
         a red wall — since real flame flickers and shifts shape.
      3. Find contours in the combined mask above a minimum area.
      4. Confidence is a blend of color saturation strength, brightness,
         and contour area — a heuristic score, not a calibrated
         probability. It's tunable via the weights below if it's too
         trigger-happy or too quiet for your room's lighting.

    This is the MVP signal: zero setup, but it WILL false-positive on
    things like sunsets through a window or someone in an orange jacket
    moving around. That's expected and fine for Phase 1 — Phase 2+ is
    where you'd swap in or blend a trained model.
    """

    def __init__(self, min_area: int = 400, sat_min: int = 130):
        self.min_area = min_area
        self.sat_min = sat_min
        self._prev_gray: Optional[np.ndarray] = None

    def _fire_color_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Fire spans roughly red -> orange -> yellow in hue (0-35),
        # with high saturation and high value (bright, vivid colors).
        # sat_min is the key knob for telling rich flame color apart from
        # pale warm-tinted specular highlights (shiny metal, glass, etc) —
        # those tend to sit much lower on saturation than real flame.
        lower1 = np.array([0, self.sat_min, 140])
        upper1 = np.array([35, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        # Hue wraps around near 180 for deep red flame edges.
        lower2 = np.array([170, self.sat_min, 140])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return mask

    def _motion_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            # No motion history yet on the very first frame. Returning an
            # all-active mask here would false-alert on any static
            # red/orange object the camera happens to be pointed at on
            # startup, so instead we report "no motion" until frame 2 —
            # one frame of latency, but no startup false positive.
            return np.zeros_like(gray)

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, np.ones((9, 9), np.uint8), iterations=2)
        return motion

    def detect(self, frame: np.ndarray) -> List[Detection]:
        color_mask = self._fire_color_mask(frame)
        motion_mask = self._motion_mask(frame)
        combined = cv2.bitwise_and(color_mask, motion_mask)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            roi_sat = hsv[y:y + h, x:x + w, 1]
            roi_val = hsv[y:y + h, x:x + w, 2]
            sat_strength = float(np.mean(roi_sat)) / 255.0 if roi_sat.size else 0.0
            val_strength = float(np.mean(roi_val)) / 255.0 if roi_val.size else 0.0
            area_strength = min(area / 8000.0, 1.0)

            confidence = 0.45 * sat_strength + 0.25 * val_strength + 0.30 * area_strength
            confidence = max(0.0, min(confidence, 0.99))

            detections.append(Detection(x, y, x + w, y + h, confidence, "fire"))

        return detections


class YoloFireDetector(FireDetector):
    """
    Hook for a real trained model. Pass weights_path to a YOLO .pt file
    trained on fire/smoke classes (Ultralytics format).

    If ultralytics isn't installed or the weights fail to load, this
    raises at construction time (in __init__) rather than at detect()
    time — so main.py can catch it once at startup and fall back to the
    heuristic detector instead of crashing mid-stream.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.4, device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )
        detections: List[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names.get(cls_id, "fire") if isinstance(names, dict) else str(cls_id)
                detections.append(Detection(x1, y1, x2, y2, conf, label))
        return detections
