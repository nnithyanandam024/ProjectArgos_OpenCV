"""
Project Argos – Core Guardian WebSocket
Endpoint: WS /ws/guardian

Message in  (JSON): { frame: <base64 jpg>, params: { zone: [[x,y],...], conf: 0.35 } }
Message out (JSON): { frame: <base64 jpg>, alerts: [...], risk_score: 0-100,
                      heatmap_pts: [[cx,cy],...], detections: [...] }
"""
import cv2
import numpy as np
import base64
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from utils import decode_image, encode_image, run_yolo, BAG_CLASSES, CLASS_PERSON, CLASS_CELL_PHONE
from routers.incidents_router import add_incident

router = APIRouter()

# ── Alert colours ─────────────────────────────────────────────────────────────
COLOURS = {
    "phone":        (255, 255, 255),   # white
    "fall":         (255, 255, 255),   # white
    "abandoned_bag":(255, 255, 255),   # white
    "zone_breach":  (255, 255, 255),   # white
    "person":       (120, 120, 120),   # gray
    "bag":          (150, 150, 150),   # gray
}

# ── Risk weights per alert type ───────────────────────────────────────────────
RISK_WEIGHTS = {
    "phone":         15,
    "fall":          40,
    "abandoned_bag": 35,
    "zone_breach":   30,
}





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
    pts = np.array(zone, dtype=np.int32)
    colour = (255, 255, 255) if breach else (120, 120, 120)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
    overlay = frame.copy()
    if len(zone) >= 3:
        cv2.fillPoly(overlay, [pts], color=(*colour[:2], 80))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    label = "⚠ RESTRICTED" if breach else "RESTRICTED ZONE"
    if len(zone) > 0:
        cv2.putText(frame, label, (zone[0][0], zone[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


# ── Per-connection state ──────────────────────────────────────────────────────
class GuardianState:
    def __init__(self):
        self.bag_first_seen: dict[int, float] = {}   # bag_detection_idx → timestamp
        self.BAG_TIMEOUT = 5.0   # seconds before abandoned alert

    def check_abandoned(self, bag_dets: list, person_dets: list, now: float):
        """
        Compare each bag against all person bboxes.
        If no person is within 150px of a bag for >BAG_TIMEOUT seconds → alert.
        """
        alerts = []
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


@router.websocket("/ws/guardian")
async def guardian_ws(websocket: WebSocket):
    await websocket.accept()
    state = GuardianState()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            frame_b64: str  = msg.get("frame", "")
            params: dict    = msg.get("params", {})
            if not frame_b64:
                continue

            img_bytes = base64.b64decode(frame_b64)
            frame     = decode_image(img_bytes)
            if frame is None:
                continue

            conf     = float(params.get("conf", 0.35))
            zone_pts = params.get("zone", [])   # [[x,y], ...]
            now      = time.time()

            # ── Run YOLOv8 ───────────────────────────────────────────────────
            _, detections = run_yolo(frame, conf)

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

            # ── 2. Fall Detection (person bbox aspect ratio) ──────────────────
            for d in person_dets:
                b  = d["bbox"]
                pw = b["x2"] - b["x1"]
                ph = b["y2"] - b["y1"]
                cx = (b["x1"] + b["x2"]) // 2
                cy = (b["y1"] + b["y2"]) // 2
                heatmap_pts.append([cx, cy])

                if ph > 0 and pw / ph > 1.4:   # wider than tall → fallen
                    cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["fall"], 3)
                    cv2.putText(frame, f"FALL DETECTED", (b["x1"], b["y1"] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS["fall"], 2)
                    alerts.append({"type": "fall", "confidence": d["confidence"], "bbox": b})
                else:
                    cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["person"], 2)
                    cv2.putText(frame, f"Person {d['confidence']:.2f}", (b["x1"], b["y1"] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS["person"], 1)

                # ── 3. Zone Breach ────────────────────────────────────────────
                if zone_pts and point_in_polygon(cx, cy, zone_pts):
                    alerts.append({"type": "zone_breach", "confidence": d["confidence"], "bbox": b})

            # ── 4. Abandoned Bag ──────────────────────────────────────────────
            for d in bag_dets:
                b = d["bbox"]
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["bag"], 2)
                cv2.putText(frame, d["label"], (b["x1"], b["y1"] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS["bag"], 1)

            abandoned = state.check_abandoned(bag_dets, person_dets, now)
            for d in abandoned:
                b = d["bbox"]
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), COLOURS["abandoned_bag"], 3)
                cv2.putText(frame, "ABANDONED BAG!", (b["x1"], b["y1"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS["abandoned_bag"], 2)
                alerts.append({"type": "abandoned_bag", "confidence": d["confidence"], "bbox": b})



            # ── Zone overlay ──────────────────────────────────────────────────
            zone_breach = any(a["type"] == "zone_breach" for a in alerts)
            draw_zone(frame, zone_pts, zone_breach)

            # ── Risk Score (0–100) ────────────────────────────────────────────
            seen_types = {a["type"] for a in alerts}
            risk_score = min(100, sum(RISK_WEIGHTS.get(t, 0) for t in seen_types))

            # Overlay risk score on frame
            risk_colour = (255, 255, 255)
            cv2.putText(frame, f"RISK: {risk_score}", (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_colour, 2)

            # ── Timestamp overlay ─────────────────────────────────────────────
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (10, frame.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # ── Persist new incidents ─────────────────────────────────────────
            snapshot_b64 = encode_image(frame)   # snapshot of annotated frame
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
                "frame":        snapshot_b64,
                "alerts":       alerts,
                "risk_score":   risk_score,
                "heatmap_pts":  heatmap_pts,
                "detections":   detections,
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
