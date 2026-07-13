"""
Project Argos – Shared utility helpers
Handles image encode/decode and re-exports the new ORT-based inference API.

Backward compatibility:
  run_yolo()  → still callable from guardian_ws and tests
  get_net()   → deprecated; kept to avoid import errors in old tests
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


# ── Re-export inference API from inference.py ─────────────────────────────────
# All detection/pose logic now lives in inference.py (ORT-backed).

from inference import (
    run_yolo,
    run_pose,
    run_fire_smoke,
    get_engine,
    letterbox,
    rescale_boxes,
    rescale_keypoints,
    nms_numpy,
    COCO_CLASSES,
    CLASS_PERSON,
    CLASS_BACKPACK,
    CLASS_HANDBAG,
    CLASS_SUITCASE,
    CLASS_CELL_PHONE,
    BAG_CLASSES,
    TARGET_CLASSES,
    DEFAULT_CONF,
    DEFAULT_NMS,
    FIRE_SMOKE_CLASSES,
    FIRE_CLASS_ID,
    SMOKE_CLASS_ID,
    FIRE_CONF_THRESH,
    SMOKE_CONF_THRESH,
    KP_L_SHOULDER, KP_R_SHOULDER,
    KP_L_HIP, KP_R_HIP,
    KP_L_KNEE, KP_R_KNEE,
    KP_L_ANKLE, KP_R_ANKLE,
    KP_NOSE,
    SKELETON,
    KP_COLOR,
    SKELETON_COLOR,
)


# ── Deprecated legacy shim ────────────────────────────────────────────────────

def get_net():
    """
    DEPRECATED — kept for backward compatibility with old tests.
    Use get_engine() from inference.py instead.
    """
    import warnings
    warnings.warn(
        "get_net() is deprecated. Use get_engine() from inference.py.",
        DeprecationWarning, stacklevel=2,
    )
    return None
