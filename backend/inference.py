"""
Project Argos – Unified ONNX Runtime Inference Engine
Replaces cv2.dnn with onnxruntime for:
  - YOLOv8n detection (person / bag / phone)
  - YOLOv8n-pose (skeleton keypoints → accurate fall detection)

Key improvements over the old cv2.dnn approach:
  • Letterbox preprocessing  – no distortion on wide/tall webcam frames
  • Per-class confidence thresholds
  • Vectorised NMS via numpy (no cv2.dnn.NMSBoxes dependency)
  • Frame-change gating (skip inference on static scenes)
  • Multi-output support for pose model (outputs: [detection, keypoints])
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import onnxruntime as ort

# ── Model Paths ───────────────────────────────────────────────────────────────
_MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
DET_MODEL       = os.path.join(_MODELS_DIR, "yolov8n.onnx")
POSE_MODEL      = os.path.join(_MODELS_DIR, "yolov8n-pose.onnx")
FIRE_SMOKE_MODEL = os.path.join(_MODELS_DIR, "fire_smoke.onnx")

# ── COCO Class List ───────────────────────────────────────────────────────────
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

# ── Class IDs ─────────────────────────────────────────────────────────────────
CLASS_PERSON     = 0
CLASS_BACKPACK   = 24
CLASS_HANDBAG    = 26
CLASS_SUITCASE   = 28
CLASS_CELL_PHONE = 67
BAG_CLASSES      = {CLASS_BACKPACK, CLASS_HANDBAG, CLASS_SUITCASE}
TARGET_CLASSES   = frozenset({CLASS_PERSON, CLASS_BACKPACK, CLASS_HANDBAG,
                               CLASS_SUITCASE, CLASS_CELL_PHONE})

# ── Per-class confidence thresholds (tuned) ───────────────────────────────────
# Bags: low threshold -> catch backpacks in poor lighting
# Phone: higher threshold -> reduce false positives (books, remotes look similar)
CLASS_CONF = {
    CLASS_PERSON:     0.30,
    CLASS_BACKPACK:   0.25,
    CLASS_HANDBAG:    0.25,
    CLASS_SUITCASE:   0.25,
    CLASS_CELL_PHONE: 0.40,
}
DEFAULT_CONF = 0.25   # fallback for any class not listed above
DEFAULT_NMS  = 0.40

# ── Fire / Smoke classes (luminous0219 model: 2-class YOLOv8n) ───────────────
# Trained on Roboflow fire-and-smoke dataset, 150 epochs
FIRE_SMOKE_CLASSES = ["fire", "smoke"]   # class 0=fire, 1=smoke
FIRE_CLASS_ID   = 0
SMOKE_CLASS_ID  = 1
FIRE_CONF_THRESH  = 0.35   # fire: prefer precision over recall
SMOKE_CONF_THRESH = 0.30   # smoke: be a bit more sensitive

# ── COCO Pose Keypoint Labels (17 keypoints) ──────────────────────────────────
KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR      = 0, 1, 2, 3, 4
KP_L_SHOULDER, KP_R_SHOULDER                           = 5, 6
KP_L_ELBOW,    KP_R_ELBOW                              = 7, 8
KP_L_WRIST,    KP_R_WRIST                              = 9, 10
KP_L_HIP,      KP_R_HIP                               = 11, 12
KP_L_KNEE,     KP_R_KNEE                              = 13, 14
KP_L_ANKLE,    KP_R_ANKLE                             = 15, 16

SKELETON = [
    (KP_NOSE, KP_L_SHOULDER), (KP_NOSE, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP),  (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE),  (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE),  (KP_R_KNEE, KP_R_ANKLE),
]

# Keypoint drawing colours (BGR)
KP_COLOR       = (0, 230, 255)   # cyan
SKELETON_COLOR = (100, 200, 100) # green


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    target: tuple[int, int] = (640, 640),
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image to target size preserving aspect ratio, padding with pad_color.
    Returns:
        padded_img  – (H, W, 3) uint8 image ready for inference
        scale       – scalar used for bbox rescaling
        (dw, dh)    – padding added (pixels) on each side
    """
    ih, iw = img.shape[:2]
    th, tw = target
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    padded = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    dw = (tw - nw) // 2
    dh = (th - nh) // 2
    padded[dh:dh + nh, dw:dw + nw] = resized
    return padded, scale, (dw, dh)


def rescale_boxes(
    boxes: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """
    Rescale YOLO (cx,cy,w,h) 640-space bboxes back to original image space.
    Returns (N,4) array of [x1,y1,x2,y2] clipped to orig_shape.
    """
    dw, dh = pad
    oh, ow = orig_shape

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((cx - bw / 2) - dw) / scale
    y1 = ((cy - bh / 2) - dh) / scale
    x2 = ((cx + bw / 2) - dw) / scale
    y2 = ((cy + bh / 2) - dh) / scale

    x1 = np.clip(x1, 0, ow).astype(int)
    y1 = np.clip(y1, 0, oh).astype(int)
    x2 = np.clip(x2, 0, ow).astype(int)
    y2 = np.clip(y2, 0, oh).astype(int)

    return np.stack([x1, y1, x2, y2], axis=1)


def rescale_keypoints(
    kpts: np.ndarray,
    scale: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """
    Rescale keypoints from 640-space back to original image space.
    kpts: (N, 17, 3)  – (x, y, visibility)
    """
    dw, dh = pad
    out = kpts.copy()
    out[:, :, 0] = (kpts[:, :, 0] - dw) / scale  # x
    out[:, :, 1] = (kpts[:, :, 1] - dh) / scale  # y
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised NMS (pure numpy — no cv2.dnn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = DEFAULT_NMS,
) -> list[int]:
    """IoU-based NMS. boxes: (N,4) [x1,y1,x2,y2]. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Singleton inference engine backed by onnxruntime.
    Lazy-loads sessions on first use.
    Models:
      - yolov8n.onnx        -> person / bag / phone detection
      - yolov8n-pose.onnx   -> skeleton keypoints for fall detection
      - fire_smoke.onnx     -> fire & smoke detection (luminous0219, 150 epochs)
    """

    _instance: "InferenceEngine | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._det_session        = None
            cls._instance._pose_session       = None
            cls._instance._fire_smoke_session  = None
        return cls._instance

    # ── Session Loaders ───────────────────────────────────────────────────────

    def _get_det_session(self) -> ort.InferenceSession:
        if self._det_session is None:
            if not os.path.exists(DET_MODEL):
                raise FileNotFoundError(f"Detection model not found: {DET_MODEL}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self._det_session = ort.InferenceSession(
                DET_MODEL, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._det_session

    def _get_pose_session(self) -> ort.InferenceSession | None:
        if self._pose_session is None:
            if not os.path.exists(POSE_MODEL):
                return None   # gracefully degrade — pose model not downloaded yet
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self._pose_session = ort.InferenceSession(
                POSE_MODEL, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._pose_session

    # ── Detection (person / bag / phone) ──────────────────────────────────────

    def run_detection(
        self,
        img: np.ndarray,
        conf_thresh: float = DEFAULT_CONF,
    ) -> list[dict]:
        """
        Run YOLOv8n detection, filtering to campus-relevant classes only.
        Returns list of dicts: { label, class_id, confidence, bbox:{x1,y1,x2,y2} }
        """
        session = self._get_det_session()
        h, w = img.shape[:2]

        padded, scale, pad = letterbox(img)
        inp = padded.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]   # (1,3,640,640)

        input_name = session.get_inputs()[0].name
        preds = session.run(None, {input_name: inp})[0]   # (1, 84, 8400)
        preds = preds[0].T                                 # (8400, 84)

        raw_boxes, raw_confs, raw_cids = [], [], []
        for pred in preds:
            scores = pred[4:]
            cid    = int(np.argmax(scores))
            if cid not in TARGET_CLASSES:
                continue
            conf = float(scores[cid])
            thresh = CLASS_CONF.get(cid, conf_thresh)
            if conf >= thresh:
                raw_boxes.append(pred[:4])    # cx, cy, w, h  (640-space)
                raw_confs.append(conf)
                raw_cids.append(cid)

        if not raw_boxes:
            return []

        boxes_arr = np.array(raw_boxes)
        confs_arr = np.array(raw_confs)
        cids_arr  = np.array(raw_cids, dtype=int)

        # Rescale to original image space for NMS
        xyxy = rescale_boxes(boxes_arr, scale, pad, (h, w))
        keep = nms_numpy(xyxy, confs_arr, DEFAULT_NMS)

        detections = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "label":      COCO_CLASSES[cids_arr[i]],
                "class_id":   int(cids_arr[i]),
                "confidence": round(float(confs_arr[i]), 3),
                "bbox":       {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            })
        return detections

    # ── Pose (keypoints for fall detection) ──────────────────────────────────

    def run_pose(
        self,
        img: np.ndarray,
        conf_thresh: float = 0.30,
    ) -> list[dict]:
        """
        Run YOLOv8n-pose and return person detections with keypoints.
        Returns list of dicts:
          { confidence, bbox:{x1,y1,x2,y2}, keypoints: np.ndarray (17,3) [x,y,vis] }
        Falls back to empty list if pose model not available.
        """
        session = self._get_pose_session()
        if session is None:
            return []   # pose model not downloaded — caller handles gracefully

        h, w = img.shape[:2]
        padded, scale, pad = letterbox(img)
        inp = padded.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]

        input_name = session.get_inputs()[0].name
        raw = session.run(None, {input_name: inp})[0]   # (1, 56, 8400)
        preds = raw[0].T                                  # (8400, 56)

        raw_boxes, raw_confs, raw_kpts = [], [], []
        for pred in preds:
            conf = float(pred[4])
            if conf < conf_thresh:
                continue
            raw_boxes.append(pred[:4])        # cx, cy, w, h
            raw_confs.append(conf)
            raw_kpts.append(pred[5:].reshape(17, 3))   # (17, 3)

        if not raw_boxes:
            return []

        boxes_arr = np.array(raw_boxes)
        confs_arr = np.array(raw_confs)
        kpts_arr  = np.array(raw_kpts)    # (N, 17, 3)

        xyxy = rescale_boxes(boxes_arr, scale, pad, (h, w))
        keep = nms_numpy(xyxy, confs_arr, DEFAULT_NMS)

        kpts_rescaled = rescale_keypoints(kpts_arr, scale, pad)

        results = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i]
            results.append({
                "confidence": round(float(confs_arr[i]), 3),
                "bbox":       {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "keypoints":  kpts_rescaled[i],   # (17, 3)
            })
        return results

    # ── Fire / Smoke detection (luminous0219 YOLOv8n, 150 epochs) ─────────────

    def _get_fire_smoke_session(self) -> ort.InferenceSession | None:
        if self._fire_smoke_session is None:
            if not os.path.exists(FIRE_SMOKE_MODEL):
                return None   # gracefully degrade if not yet downloaded
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4
            self._fire_smoke_session = ort.InferenceSession(
                FIRE_SMOKE_MODEL, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._fire_smoke_session

    def run_fire_smoke(
        self,
        img: np.ndarray,
        fire_thresh: float = FIRE_CONF_THRESH,
        smoke_thresh: float = SMOKE_CONF_THRESH,
    ) -> list[dict]:
        """
        Run YOLOv8n fire/smoke model (luminous0219, 2 classes).
        Returns list of dicts:
          { label: 'fire'|'smoke', class_id: 0|1, confidence, bbox:{x1,y1,x2,y2} }
        Returns [] gracefully if model not downloaded yet.
        """
        session = self._get_fire_smoke_session()
        if session is None:
            return []

        h, w = img.shape[:2]
        padded, scale, pad = letterbox(img)
        inp = padded.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]   # (1,3,640,640)

        input_name = session.get_inputs()[0].name
        preds = session.run(None, {input_name: inp})[0]   # (1, 6, 8400)
        preds = preds[0].T                                  # (8400, 6)
        # Layout: [cx, cy, w, h, fire_conf, smoke_conf]

        raw_boxes, raw_confs, raw_cids = [], [], []
        for pred in preds:
            class_scores = pred[4:]   # [fire_score, smoke_score]
            cid  = int(np.argmax(class_scores))
            conf = float(class_scores[cid])
            thresh = fire_thresh if cid == FIRE_CLASS_ID else smoke_thresh
            if conf >= thresh:
                raw_boxes.append(pred[:4])
                raw_confs.append(conf)
                raw_cids.append(cid)

        if not raw_boxes:
            return []

        boxes_arr = np.array(raw_boxes)
        confs_arr = np.array(raw_confs)
        cids_arr  = np.array(raw_cids, dtype=int)

        xyxy = rescale_boxes(boxes_arr, scale, pad, (h, w))
        keep = nms_numpy(xyxy, confs_arr, DEFAULT_NMS)

        detections = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "label":      FIRE_SMOKE_CLASSES[cids_arr[i]],
                "class_id":   int(cids_arr[i]),
                "confidence": round(float(confs_arr[i]), 3),
                "bbox":       {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            })
        return detections


# ── Module-level singleton accessor ──────────────────────────────────────────

_engine: InferenceEngine | None = None

def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def run_yolo(img: np.ndarray, conf_thresh: float = DEFAULT_CONF):
    """
    Backward-compatible wrapper. Returns (img_copy, detections).
    Drop-in replacement for the old cv2.dnn-based run_yolo().
    """
    detections = get_engine().run_detection(img, conf_thresh)
    return img.copy(), detections


def run_pose(img: np.ndarray, conf_thresh: float = 0.30):
    """Run pose estimation. Returns list of person pose dicts."""
    return get_engine().run_pose(img, conf_thresh)


def run_fire_smoke(img: np.ndarray,
                   fire_thresh: float = FIRE_CONF_THRESH,
                   smoke_thresh: float = SMOKE_CONF_THRESH):
    """Run fire/smoke detection. Returns list of detection dicts."""
    return get_engine().run_fire_smoke(img, fire_thresh, smoke_thresh)
