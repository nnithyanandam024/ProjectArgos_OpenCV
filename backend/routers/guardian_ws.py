"""
Project Argos – Core Guardian WebSocket  (v3 — ORT + Pose + Fire/Smoke)
Endpoint: WS /ws/guardian

Message in  (JSON): { frame: <base64 jpg>, params: { zone: [[x,y],...], conf: 0.35 } }
Message out (JSON): { frame: <base64 jpg>, alerts: [...], risk_score: 0-100,
                      heatmap_pts: [[cx,cy],...], detections: [...],
                      fire_smoke: [...] }

Improvements over v2:
  + Fire & Smoke detection via YOLOv8n (luminous0219, 150 epochs, 2 classes)
  + Graceful fire/smoke degradation if model not yet downloaded
"""

from __future__ import annotations

import cv2
import numpy as np
import base64
import json
import math
import time
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from utils import (
    decode_image, encode_image,
    run_yolo, run_pose,
    # run_fire_smoke,   # PAUSED — uncomment to re-enable fire/smoke detection
    BAG_CLASSES, CLASS_PERSON, CLASS_CELL_PHONE,
    KP_L_SHOULDER, KP_R_SHOULDER,
    KP_L_HIP, KP_R_HIP,
    KP_L_KNEE, KP_R_KNEE,
    KP_NOSE,
    SKELETON, KP_COLOR, SKELETON_COLOR,
)
from routers.incidents_router import add_incident

router = APIRouter()

# ── Alert colours (BGR) ───────────────────────────────────────────────────────
COLOURS = {
    "phone":         (0, 255, 255),    # yellow
    "fall":          (0, 0, 255),      # red
    "abandoned_bag": (0, 140, 255),    # orange
    "zone_breach":   (255, 0, 255),    # magenta
    "person":        (120, 120, 120),  # gray
    "bag":           (150, 200, 150),  # light green
    "pose_person":   (0, 230, 100),    # bright green (normal standing)
    "fire":          (0, 60, 255),     # red-orange
    "smoke":         (160, 160, 160),  # light gray
}

# ── Risk weights per alert type ───────────────────────────────────────────────
RISK_WEIGHTS = {
    "phone":         15,
    "fall":          40,
    "abandoned_bag": 35,
    "zone_breach":   30,
    "fire":          70,   # highest threat — immediate evacuation
    "smoke":         50,   # serious but may be early warning
}

# ── Frame-change gating threshold ─────────────────────────────────────────────
FRAME_DIFF_THRESH = 3.0    # mean pixel diff below this → skip inference
FORCE_RUN_EVERY   = 10     # force inference every N frames regardless

# ── Fall detection parameters ─────────────────────────────────────────────────
FALL_ANGLE_THRESH   = 35.0   # degrees — torso angle below this → possible fall
FALL_DROP_THRESH    = 35     # pixels — hip drop over 5 frames to confirm fall
FALL_CONFIRM_FRAMES = 3      # consecutive frames flagged before alerting
VIS_THRESH          = 0.3    # keypoint visibility threshold


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def point_in_polygon(px: int, py: int, polygon: list) -> bool:
    """Ray-casting algorithm to test if point is inside polygon."""
    if len(polygon) < 3:
        return False
    n      = len(polygon)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def draw_zone(frame: np.ndarray, zone: list, breach: bool):
    """Draw the restricted zone polygon on the frame."""
    if len(zone) < 2:
        return
    pts    = np.array(zone, dtype=np.int32)
    colour = (0, 0, 255) if breach else (80, 80, 255)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
    overlay = frame.copy()
    if len(zone) >= 3:
        cv2.fillPoly(overlay, [pts], color=(*colour[:2], 60))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    label = "!! RESTRICTED !!" if breach else "RESTRICTED ZONE"
    if len(zone) > 0:
        cv2.putText(frame, label, (zone[0][0], zone[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint-based fall detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _visible(kp: np.ndarray, idx: int) -> bool:
    return kp[idx, 2] >= VIS_THRESH


def compute_torso_angle(kp: np.ndarray) -> float | None:
    """
    Compute angle of the torso vector (mid-hip → mid-shoulder) from vertical.
    Returns angle in degrees, or None if keypoints are not visible.
    0° = upright, 90° = fully horizontal (fallen).
    """
    if not (_visible(kp, KP_L_SHOULDER) and _visible(kp, KP_R_SHOULDER) and
            _visible(kp, KP_L_HIP)      and _visible(kp, KP_R_HIP)):
        return None

    sx = (kp[KP_L_SHOULDER, 0] + kp[KP_R_SHOULDER, 0]) / 2
    sy = (kp[KP_L_SHOULDER, 1] + kp[KP_R_SHOULDER, 1]) / 2
    hx = (kp[KP_L_HIP, 0] + kp[KP_R_HIP, 0]) / 2
    hy = (kp[KP_L_HIP, 1] + kp[KP_R_HIP, 1]) / 2

    dx = sx - hx
    dy = sy - hy
    length = math.hypot(dx, dy)
    if length < 5:
        return None

    # Angle with vertical (dy axis) — 0° is upright, 90° is lying flat
    angle_from_vertical = math.degrees(math.acos(abs(dy) / length))
    return angle_from_vertical


def compute_mid_hip_y(kp: np.ndarray) -> float | None:
    """Return Y coordinate of the midpoint between hips, or None."""
    if _visible(kp, KP_L_HIP) and _visible(kp, KP_R_HIP):
        return (kp[KP_L_HIP, 1] + kp[KP_R_HIP, 1]) / 2
    return None


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, is_fallen: bool):
    """Draw keypoints and skeleton lines on the frame."""
    colour = (0, 0, 220) if is_fallen else SKELETON_COLOR

    # Draw skeleton limbs
    for a, b in SKELETON:
        if _visible(kp, a) and _visible(kp, b):
            pt1 = (int(kp[a, 0]), int(kp[a, 1]))
            pt2 = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(frame, pt1, pt2, colour, 2)

    # Draw keypoint circles
    for i in range(17):
        if _visible(kp, i):
            pt = (int(kp[i, 0]), int(kp[i, 1]))
            kp_c = (0, 0, 220) if is_fallen else KP_COLOR
            cv2.circle(frame, pt, 4, kp_c, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-connection state
# ─────────────────────────────────────────────────────────────────────────────

class GuardianState:
    def __init__(self):
        # ── Abandoned bag tracking ─────────────────────────────────────────
        self.bag_first_seen: dict[int, float] = {}
        self.BAG_TIMEOUT = 5.0   # seconds before abandoned alert

        # ── Frame-change gating ────────────────────────────────────────────
        self.prev_gray:       np.ndarray | None = None
        self.frame_counter:   int               = 0
        self._last_det:       list[dict]        = []
        self._last_pose:      list[dict]        = []
        self._last_fire_smoke: list[dict]       = []

        # ── Keypoint-based fall detection state ────────────────────────────
        # Keyed by detection index (int) within the current frame.
        # history: deque of (torso_angle, mid_hip_y) from last 5 frames
        self.person_kp_history: dict[int, deque]    = {}
        self.fall_counter:      dict[int, int]      = {}  # consecutive fall-flagged frames

    # ── Frame gating ──────────────────────────────────────────────────────────

    def should_run_inference(self, frame: np.ndarray) -> bool:
        """
        Return True if enough has changed to warrant full YOLO inference.
        Always runs every FORCE_RUN_EVERY frames regardless.
        """
        self.frame_counter += 1
        if self.frame_counter % FORCE_RUN_EVERY == 0:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))   # downsample for speed

        if self.prev_gray is None:
            self.prev_gray = gray
            return True

        diff = cv2.absdiff(gray, self.prev_gray).mean()
        self.prev_gray = gray

        return diff >= FRAME_DIFF_THRESH

    # ── Abandoned bag detection ───────────────────────────────────────────────

    def check_abandoned(self, bag_dets: list, person_dets: list, now: float) -> list:
        """
        Compare each bag against all person bboxes.
        If no person is within proximity for > BAG_TIMEOUT seconds → alert.
        """
        alerts      = []
        current_ids = set()
        for idx, bag in enumerate(bag_dets):
            bx = (bag["bbox"]["x1"] + bag["bbox"]["x2"]) / 2
            by = (bag["bbox"]["y1"] + bag["bbox"]["y2"]) / 2
            near_person = False
            for p in person_dets:
                px = (p["bbox"]["x1"] + p["bbox"]["x2"]) / 2
                py = (p["bbox"]["y1"] + p["bbox"]["y2"]) / 2
                if abs(bx - px) < 150 and abs(by - py) < 200:
                    near_person = True
                    break
            if not near_person:
                current_ids.add(idx)
                if idx not in self.bag_first_seen:
                    self.bag_first_seen[idx] = now
                elif now - self.bag_first_seen[idx] >= self.BAG_TIMEOUT:
                    alerts.append(bag)
            else:
                self.bag_first_seen.pop(idx, None)

        # Prune stale keys
        for k in list(self.bag_first_seen):
            if k not in current_ids:
                del self.bag_first_seen[k]
        return alerts

    # ── Keypoint fall detection ───────────────────────────────────────────────

    def check_fall_keypoints(self, pose_results: list[dict], frame_h: int) -> list[bool]:
        """
        Analyse each person's pose history to determine fall status.
        Returns list of bool (len == len(pose_results)).

        Algorithm:
          1. Compute torso angle (0° upright → 90° horizontal).
          2. Track mid-hip Y across last 5 frames.
          3. A person is 'falling' if:
               angle > FALL_ANGLE_THRESH  (torso tilted)
               OR  recent hip Y change > FALL_DROP_THRESH (dropped fast)
          4. Must be flagged for FALL_CONFIRM_FRAMES consecutive frames.
        """
        fall_flags = []
        active_ids = set(range(len(pose_results)))

        for idx, person in enumerate(pose_results):
            kp = person["keypoints"]   # (17, 3)

            # Initialise history for new tracklets
            if idx not in self.person_kp_history:
                self.person_kp_history[idx] = deque(maxlen=5)
                self.fall_counter[idx]      = 0

            angle   = compute_torso_angle(kp)
            hip_y   = compute_mid_hip_y(kp)

            self.person_kp_history[idx].append((angle, hip_y))

            is_falling = False

            # ── Primary: torso angle ────────────────────────────────────────
            if angle is not None and angle > FALL_ANGLE_THRESH:
                is_falling = True

            # ── Secondary: rapid hip drop ───────────────────────────────────
            if not is_falling and len(self.person_kp_history[idx]) >= 3:
                ys = [entry[1] for entry in self.person_kp_history[idx]
                      if entry[1] is not None]
                if len(ys) >= 3:
                    drop = max(ys) - min(ys[:2])  # drop from early frames
                    if drop > FALL_DROP_THRESH:
                        is_falling = True

            # ── Temporal debounce ───────────────────────────────────────────
            if is_falling:
                self.fall_counter[idx] += 1
            else:
                self.fall_counter[idx] = max(0, self.fall_counter[idx] - 1)

            fall_confirmed = self.fall_counter[idx] >= FALL_CONFIRM_FRAMES
            fall_flags.append(fall_confirmed)

        # ── Fallback: aspect-ratio heuristic when pose model unavailable ────
        # (handled in guardian_ws — pose_results will be empty)

        # Prune stale person IDs (track IDs that weren't seen this frame)
        for k in list(self.person_kp_history):
            if k not in active_ids:
                del self.person_kp_history[k]
                del self.fall_counter[k]

        return fall_flags


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Handler
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/guardian")
async def guardian_ws(websocket: WebSocket):
    await websocket.accept()
    state = GuardianState()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            frame_b64: str = msg.get("frame", "")
            params: dict   = msg.get("params", {})
            if not frame_b64:
                continue

            img_bytes = base64.b64decode(frame_b64)
            frame     = decode_image(img_bytes)
            if frame is None:
                continue

            conf      = float(params.get("conf", 0.35))
            zone_pts  = params.get("zone", [])
            now       = time.time()

            # ── Frame-change gating ──────────────────────────────────────────
            run_inference = state.should_run_inference(frame)

            if run_inference:
                # ── Run YOLOv8n detection ────────────────────────────────────
                _, detections    = run_yolo(frame, conf)
                # ── Run YOLOv8n-pose ─────────────────────────────────────────
                pose_results     = run_pose(frame, conf_thresh=0.30)
                # ── Fire/Smoke (PAUSED) ───────────────────────────────────────
                # fire_smoke_dets  = run_fire_smoke(frame)    # uncomment to enable
                fire_smoke_dets  = []                         # disabled
                state._last_det        = detections
                state._last_pose       = pose_results
                state._last_fire_smoke = fire_smoke_dets
            else:
                detections      = state._last_det
                pose_results    = state._last_pose
                fire_smoke_dets = state._last_fire_smoke

            person_dets = [d for d in detections if d["class_id"] == CLASS_PERSON]
            bag_dets    = [d for d in detections if d["class_id"] in BAG_CLASSES]
            phone_dets  = [d for d in detections if d["class_id"] == CLASS_CELL_PHONE]

            alerts      = []
            heatmap_pts = []

            # ── 1. Phone Detection ────────────────────────────────────────────
            for d in phone_dets:
                b = d["bbox"]
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["phone"], 2)
                cv2.putText(frame, f"PHONE {d['confidence']:.2f}", (b["x1"], b["y1"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOURS["phone"], 2)
                alerts.append({"type": "phone", "confidence": d["confidence"], "bbox": b})

            # ── 2. Fall Detection — Keypoint-based (primary) ──────────────────
            pose_available = len(pose_results) > 0
            fall_flags     = state.check_fall_keypoints(pose_results, frame.shape[0])

            for idx, person in enumerate(pose_results):
                b   = person["bbox"]
                kp  = person["keypoints"]
                cx  = (b["x1"] + b["x2"]) // 2
                cy  = (b["y1"] + b["y2"]) // 2
                heatmap_pts.append([cx, cy])

                is_fallen = fall_flags[idx]
                draw_skeleton(frame, kp, is_fallen)

                if is_fallen:
                    cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["fall"], 3)
                    cv2.putText(frame, "FALL DETECTED", (b["x1"], b["y1"] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS["fall"], 2)
                    alerts.append({"type": "fall", "confidence": person["confidence"], "bbox": b})
                else:
                    cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                                  COLOURS["pose_person"], 1)

                # ── 3a. Zone Breach (pose-derived person) ─────────────────────
                if zone_pts and point_in_polygon(cx, cy, zone_pts):
                    alerts.append({"type": "zone_breach", "confidence": person["confidence"], "bbox": b})

            # ── 2b. Fallback: aspect-ratio for persons from det model ─────────
            # Used when pose model is not available (ONNX not downloaded yet)
            if not pose_available:
                for d in person_dets:
                    b  = d["bbox"]
                    pw = b["x2"] - b["x1"]
                    ph = b["y2"] - b["y1"]
                    cx = (b["x1"] + b["x2"]) // 2
                    cy = (b["y1"] + b["y2"]) // 2
                    heatmap_pts.append([cx, cy])

                    if ph > 0 and pw / ph > 1.4:
                        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["fall"], 3)
                        cv2.putText(frame, "FALL DETECTED (heuristic)", (b["x1"], b["y1"] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOURS["fall"], 2)
                        alerts.append({"type": "fall", "confidence": d["confidence"], "bbox": b})
                    else:
                        cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["person"], 2)
                        cv2.putText(frame, f"Person {d['confidence']:.2f}", (b["x1"], b["y1"] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS["person"], 1)

                    # ── 3b. Zone Breach (fallback path) ───────────────────────
                    if zone_pts and point_in_polygon(cx, cy, zone_pts):
                        alerts.append({"type": "zone_breach", "confidence": d["confidence"], "bbox": b})

            # ── 4. Abandoned Bag ──────────────────────────────────────────────
            for d in bag_dets:
                b = d["bbox"]
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["bag"], 2)
                cv2.putText(frame, d["label"], (b["x1"], b["y1"] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS["bag"], 1)

            abandoned_bags = state.check_abandoned(bag_dets, person_dets, now)
            for d in abandoned_bags:
                b = d["bbox"]
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["abandoned_bag"], 3)
                cv2.putText(frame, "ABANDONED BAG!", (b["x1"], b["y1"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS["abandoned_bag"], 2)
                alerts.append({"type": "abandoned_bag", "confidence": d["confidence"], "bbox": b})

            # ── 5. Fire & Smoke Detection (PAUSED) ───────────────────────────
            # To re-enable: uncomment run_fire_smoke above and this block
            # for d in fire_smoke_dets:
            #     b     = d["bbox"]
            #     label = d["label"]          # "fire" or "smoke"
            #     col   = COLOURS.get(label, (0, 100, 255))
            #     thickness = 3 if label == "fire" else 2
            #     cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), col, thickness)
            #     cv2.putText(frame,
            #                 f"{label.upper()} {d['confidence']:.2f}",
            #                 (b["x1"], b["y1"] - 8),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            #     alerts.append({"type": label, "confidence": d["confidence"], "bbox": b})

            # ── Zone overlay ──────────────────────────────────────────────────
            zone_breach = any(a["type"] == "zone_breach" for a in alerts)
            draw_zone(frame, zone_pts, zone_breach)

            # ── Risk Score (0–100) ────────────────────────────────────────────
            seen_types = {a["type"] for a in alerts}
            risk_score = min(100, sum(RISK_WEIGHTS.get(t, 0) for t in seen_types))

            # Colour-coded risk indicator
            if risk_score >= 70:
                risk_colour = (0, 0, 255)      # red
            elif risk_score >= 40:
                risk_colour = (0, 140, 255)    # orange
            else:
                risk_colour = (0, 200, 50)     # green

            cv2.putText(frame, f"RISK: {risk_score}", (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_colour, 2)

            # Pose / fire-smoke mode indicator
            fire_available = len(fire_smoke_dets) >= 0   # model loaded = always True after first run
            parts = []
            if pose_available:               parts.append("POSE")
            from inference import FIRE_SMOKE_MODEL
            import os
            if os.path.exists(FIRE_SMOKE_MODEL): parts.append("FIRE")
            mode_label = "MODE: " + "+".join(parts) + "+DET" if parts else "MODE: DET-ONLY"
            cv2.putText(frame, mode_label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # ── Timestamp overlay ─────────────────────────────────────────────
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (10, frame.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # ── Persist new incidents ─────────────────────────────────────────
            snapshot_b64 = encode_image(frame)
            for alert in alerts:
                add_incident({
                    "type":       alert["type"],
                    "confidence": alert["confidence"],
                    "timestamp":  ts,
                    "snapshot":   snapshot_b64,
                    "bbox":       alert.get("bbox"),
                })

            # ── Send response ─────────────────────────────────────────────────
            await websocket.send_text(json.dumps({
                "frame":       snapshot_b64,
                "alerts":      alerts,
                "risk_score":  risk_score,
                "heatmap_pts": heatmap_pts,
                "detections":  detections,
                "fire_smoke":  fire_smoke_dets,
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
