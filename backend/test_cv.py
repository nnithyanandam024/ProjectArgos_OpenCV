# -*- coding: utf-8 -*-
"""
Project Argos – CV Test Suite v2
Tests: ORT inference engine, letterbox preprocessing, keypoint fall detection,
       polygon breach, abandoned bag logic, and inference latency benchmark.
"""
import sys
import os
import time
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import guard ──────────────────────────────────────────────────────────────
try:
    import utils
    import inference
    from routers import guardian_ws
    print("[OK] Modules imported successfully.")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)


PASS = "[OK]  "
FAIL = "[FAIL]"

def result(label: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))
    return ok


# ─────────────────────────────────────────────────────────────────────────────

def test_ort_detection():
    print("\n── Test 1: ORT Detection Engine ──────────────────────────────────")
    try:
        engine = inference.get_engine()
        dummy  = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy, (100, 50), (300, 430), (180, 180, 180), -1)  # person-like shape

        t0 = time.perf_counter()
        dets = engine.run_detection(dummy, conf_thresh=0.10)
        ms  = (time.perf_counter() - t0) * 1000

        result("ORT session loads without error", True)
        result("run_detection returns a list", isinstance(dets, list))
        result(f"Inference latency < 500 ms", ms < 500, f"{ms:.1f} ms")
        print(f"      Detections on dummy frame: {len(dets)}")
    except Exception as e:
        result("ORT detection engine", False, str(e))


def test_letterbox():
    print("\n── Test 2: Letterbox Preprocessing ──────────────────────────────")
    # Wide 16:9 image
    wide_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    padded, scale, (dw, dh) = inference.letterbox(wide_img, target=(640, 640))

    result("Output shape is 640x640", padded.shape == (640, 640, 3),
           str(padded.shape))
    result("Scale < 1 for downscale", scale < 1.0, f"scale={scale:.4f}")
    # Wide 1280x720 -> scales to 640x360 -> vertical padding (top/bottom) to reach 640
    result("Vertical padding for wide 16:9 image", dh > 0 and dw == 0,
           f"dw={dw}, dh={dh}")

    # Tall portrait image 480x1080 -> scales to 284x640 -> horizontal padding (left/right)
    tall_img = np.zeros((1080, 480, 3), dtype=np.uint8)
    _, scale2, (dw2, dh2) = inference.letterbox(tall_img, target=(640, 640))
    result("Horizontal padding for tall portrait image", dw2 > 0 and dh2 == 0,
           f"dw={dw2}, dh={dh2}")

    # Square stays square
    sq_img = np.zeros((640, 640, 3), dtype=np.uint8)
    _, scale3, (dw3, dh3) = inference.letterbox(sq_img, target=(640, 640))
    result("Square image: no padding needed", dw3 == 0 and dh3 == 0 and scale3 == 1.0)


def test_nms_numpy():
    print("\n── Test 3: Vectorised NMS ────────────────────────────────────────")
    # Two heavily overlapping boxes + one separate
    boxes  = np.array([[10, 10, 100, 100],
                        [12, 12, 102, 102],
                        [200, 200, 300, 300]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    kept   = inference.nms_numpy(boxes, scores, iou_thresh=0.5)

    result("Two overlapping boxes → 1 kept + 1 separate → 2 total",
           len(kept) == 2, f"kept={kept}")
    result("Highest-score box retained first", kept[0] == 0 or boxes[kept[0], 0] == 10)


def test_polygon_breach():
    print("\n── Test 4: Polygon Breach (Ray-casting) ──────────────────────────")
    zone        = [[100, 100], [200, 100], [200, 200], [100, 200]]
    inside_pt   = (150, 150)
    outside_pt  = (50, 50)
    edge_pt     = (100, 150)   # on the border — should not raise

    is_in  = guardian_ws.point_in_polygon(*inside_pt,  zone)
    is_out = guardian_ws.point_in_polygon(*outside_pt, zone)
    try:
        guardian_ws.point_in_polygon(*edge_pt, zone)
        no_crash = True
    except Exception:
        no_crash = False

    result(f"Point {inside_pt}  is inside zone",  is_in  == True)
    result(f"Point {outside_pt} is outside zone", is_out == False)
    result("Edge point does not raise exception",  no_crash)


def test_torso_angle():
    print("\n── Test 5: Torso Angle Calculation ──────────────────────────────")
    # Upright person: shoulders above hips
    kp_upright = np.zeros((17, 3))
    kp_upright[[5, 6], 2] = 1.0   # shoulders visible
    kp_upright[[11, 12], 2] = 1.0  # hips visible
    kp_upright[5]  = [200, 100, 1]  # L shoulder (x, y, vis)
    kp_upright[6]  = [300, 100, 1]  # R shoulder
    kp_upright[11] = [200, 300, 1]  # L hip
    kp_upright[12] = [300, 300, 1]  # R hip

    angle_upright = guardian_ws.compute_torso_angle(kp_upright)
    result("Upright person angle ~0°", angle_upright is not None and angle_upright < 10,
           f"{angle_upright:.1f}°" if angle_upright else "None")

    # Fallen person: shoulders beside hips (horizontal)
    kp_fallen = np.zeros((17, 3))
    kp_fallen[[5, 6, 11, 12], 2] = 1.0
    kp_fallen[5]  = [100, 200, 1]   # L shoulder
    kp_fallen[6]  = [100, 300, 1]   # R shoulder  (same x, different y → rotated 90°)
    kp_fallen[11] = [300, 200, 1]   # L hip
    kp_fallen[12] = [300, 300, 1]   # R hip
    # torso vector: (100, 250) → (300, 250)  → horizontal

    angle_fallen = guardian_ws.compute_torso_angle(kp_fallen)
    result("Fallen person angle ~90°", angle_fallen is not None and angle_fallen > 70,
           f"{angle_fallen:.1f}°" if angle_fallen else "None")

    # Not enough visibility
    kp_invisible = np.zeros((17, 3))
    angle_none = guardian_ws.compute_torso_angle(kp_invisible)
    result("Invisible keypoints → None returned", angle_none is None)


def test_fall_debounce():
    print("\n── Test 6: Fall Debounce (3-frame confirm) ───────────────────────")
    state = guardian_ws.GuardianState()

    # Build a fallen-pose keypoint set
    kp = np.zeros((17, 3))
    kp[[5, 6, 11, 12], 2] = 1.0
    kp[5]  = [100, 200, 1]
    kp[6]  = [100, 300, 1]
    kp[11] = [300, 200, 1]
    kp[12] = [300, 300, 1]
    fake_pose = [{"confidence": 0.85, "bbox": {"x1":50,"y1":150,"x2":350,"y2":400},
                  "keypoints": kp}]

    # Frame 1 — should not yet confirm
    flags1 = state.check_fall_keypoints(fake_pose, 480)
    result("Frame 1: not yet confirmed (counter=1)", not flags1[0])

    # Frame 2 — still not confirmed
    flags2 = state.check_fall_keypoints(fake_pose, 480)
    result("Frame 2: not yet confirmed (counter=2)", not flags2[0])

    # Frame 3 — now confirmed
    flags3 = state.check_fall_keypoints(fake_pose, 480)
    result("Frame 3: fall confirmed (counter≥3)", flags3[0])


def test_abandoned_bag():
    print("\n── Test 7: Abandoned Bag Timer ───────────────────────────────────")
    state = guardian_ws.GuardianState()
    state.BAG_TIMEOUT = 0.1   # speed up test

    bag    = {"bbox": {"x1": 200, "y1": 200, "x2": 260, "y2": 260}, "confidence": 0.7, "class_id": 24, "label": "backpack"}
    person = {"bbox": {"x1": 100, "y1": 100, "x2": 160, "y2": 400}, "confidence": 0.9, "class_id": 0,  "label": "person"}

    # Bag near person -> no alert
    r1 = state.check_abandoned([bag], [person], time.time())
    result("Bag near person -> no abandoned alert", len(r1) == 0)

    # First call alone: starts the timer (bag_first_seen gets set)
    state.check_abandoned([bag], [], time.time())
    # Wait past the timeout
    time.sleep(0.15)
    # Second call alone: timer has now expired -> should produce alert
    r2 = state.check_abandoned([bag], [], time.time())
    result("Bag alone > timeout -> abandoned alert", len(r2) > 0)


def test_frame_gating():
    print("\n── Test 8: Frame-Change Gating ───────────────────────────────────")
    state = guardian_ws.GuardianState()

    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = frame1.copy()                          # identical → skip
    frame3 = frame1.copy()
    frame3[100:200, 100:200] = 200                  # changed → run

    r1 = state.should_run_inference(frame1)
    r2 = state.should_run_inference(frame2)
    r3 = state.should_run_inference(frame3)

    result("First frame always runs inference", r1)
    result("Identical frame → skips inference", not r2)
    result("Changed frame → runs inference",    r3)


def test_pose_graceful_fallback():
    print("\n── Test 9: Pose Graceful Fallback ───────────────────────────────")
    # If pose model file does not exist, run_pose() should return []
    original = inference.POSE_MODEL
    try:
        inference.InferenceEngine._instance = None   # reset singleton
        # Patch path to nonexistent file
        import inference as inf_mod
        inf_mod.POSE_MODEL = "/nonexistent/yolov8n-pose.onnx"
        # Reset session
        eng = inference.InferenceEngine()
        eng._pose_session = None

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        results = eng.run_pose(dummy)
        result("Missing pose model returns [] gracefully", results == [])
    except Exception as e:
        result("Graceful fallback without crash", False, str(e))
    finally:
        # Restore
        import inference as inf_mod
        inf_mod.POSE_MODEL = original
        inference.InferenceEngine._instance = None
        inference._engine = None


def test_latency_benchmark():
    print("\n── Test 10: Inference Latency Benchmark ──────────────────────────")
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    N     = 5
    times = []
    engine = inference.get_engine()
    for _ in range(N):
        t0 = time.perf_counter()
        engine.run_detection(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn  = min(times)
    mx  = max(times)
    print(f"      Detection  avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")
    result(f"Average detection latency < 500ms", avg < 500, f"{avg:.1f} ms")


def test_fire_smoke_graceful_fallback():
    print("\n── Test 11: Fire/Smoke Graceful Fallback ─────────────────────────")
    import inference as inf_mod
    original_path = inf_mod.FIRE_SMOKE_MODEL
    try:
        # Reset singleton so new path takes effect
        inf_mod.InferenceEngine._instance = None
        inf_mod._engine = None
        inf_mod.FIRE_SMOKE_MODEL = "/nonexistent/fire_smoke.onnx"

        eng   = inf_mod.InferenceEngine()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        res   = eng.run_fire_smoke(dummy)
        result("Missing fire/smoke model returns [] gracefully", res == [])
    except Exception as e:
        result("Fire/smoke graceful fallback", False, str(e))
    finally:
        inf_mod.FIRE_SMOKE_MODEL = original_path
        inf_mod.InferenceEngine._instance = None
        inf_mod._engine = None


def test_fire_smoke_latency():
    print("\n── Test 12: Fire/Smoke Latency (if model present) ────────────────")
    import inference as inf_mod
    import os
    if not os.path.exists(inf_mod.FIRE_SMOKE_MODEL):
        print("      [SKIP] fire_smoke.onnx not yet downloaded")
        return

    engine = inf_mod.get_engine()
    dummy  = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
    N      = 5
    times  = []
    for _ in range(N):
        t0 = time.perf_counter()
        engine.run_fire_smoke(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn  = min(times)
    mx  = max(times)
    print(f"      Fire/Smoke avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")
    result("Fire/smoke latency < 500ms", avg < 500, f"{avg:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Project Argos – CV Test Suite v3 (with Fire/Smoke)")
    print("=" * 62)

    test_ort_detection()
    test_letterbox()
    test_nms_numpy()
    test_polygon_breach()
    test_torso_angle()
    test_fall_debounce()
    test_abandoned_bag()
    test_frame_gating()
    test_pose_graceful_fallback()
    test_latency_benchmark()
    test_fire_smoke_graceful_fallback()
    test_fire_smoke_latency()

    print("\n" + "=" * 62)
    print("  All checks complete.")
    print("=" * 62)
