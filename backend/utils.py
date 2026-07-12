"""
Project Argos – Shared utility helpers
Handles image encode/decode and YOLOv8n ONNX inference via OpenCV DNN.
Tuned for: person, cell phone, bags – no smoke.
"""
import cv2
import numpy as np
import base64
import os

# ── Image helpers ────────────────────────────────────────────────────────────

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw file bytes into an OpenCV BGR image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_image(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode an OpenCV BGR image to a base64 string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt == ".jpg" else []
    _, buffer = cv2.imencode(fmt, img, params)
    return base64.b64encode(buffer).decode("utf-8")


def b64_to_image(b64_str: str) -> np.ndarray:
    """Decode a base64 string back into an OpenCV BGR image."""
    return decode_image(base64.b64decode(b64_str))


# ── YOLOv8n ONNX Engine ──────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8n.onnx")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Indices for classes we care about
CLASS_PERSON      = 0
CLASS_BACKPACK    = 24
CLASS_HANDBAG     = 26
CLASS_SUITCASE    = 28
CLASS_CELL_PHONE  = 67
BAG_CLASSES       = {CLASS_BACKPACK, CLASS_HANDBAG, CLASS_SUITCASE}

# Pre-compute the set once at module level for fast filter
TARGET_CLASSES = {CLASS_PERSON, CLASS_BACKPACK, CLASS_HANDBAG, CLASS_SUITCASE, CLASS_CELL_PHONE}

# ── Fine-tuned inference parameters ─────────────────────────────────────────
# Lower conf → catches more; NMS 0.4 → tighter deduplication
DEFAULT_CONF   = 0.25   # was 0.35 — more sensitive
DEFAULT_NMS    = 0.40   # was 0.45 — tighter boxes

_net = None

def get_net():
    global _net
    if _net is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        # Use CPU backend (CUDA not assumed)
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return _net


def run_yolo(img: np.ndarray, conf_thresh: float = DEFAULT_CONF):
    """
    Run YOLOv8n inference — tuned for campus guardian target classes.
    Returns (annotated_img, detections) where detections is a list of dicts:
      { label, class_id, confidence, bbox: {x1,y1,x2,y2} }
    """
    net = get_net()
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()                   # (1, 84, 8400)
    preds = np.transpose(preds[0], (1, 0))  # (8400, 84)

    boxes, confs, class_ids = [], [], []
    for pred in preds:
        scores = pred[4:]
        cid = int(np.argmax(scores))
        # Skip irrelevant classes immediately — much faster
        if cid not in TARGET_CLASSES:
            continue
        conf = float(scores[cid])
        if conf >= conf_thresh:
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) / 640 * w)
            y1 = int((cy - bh / 2) / 640 * h)
            x2 = int((cx + bw / 2) / 640 * w)
            y2 = int((cy + bh / 2) / 640 * h)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confs.append(conf)
            class_ids.append(cid)

    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, DEFAULT_NMS)
    detections = []
    out = img.copy()

    if len(indices) > 0:
        for i in (indices.flatten() if isinstance(indices, np.ndarray) else indices):
            x, y, bw, bh = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            cid   = class_ids[i]
            conf  = confs[i]
            label = COCO_CLASSES[cid]
            detections.append({
                "label": label,
                "class_id": cid,
                "confidence": round(conf, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    return out, detections
