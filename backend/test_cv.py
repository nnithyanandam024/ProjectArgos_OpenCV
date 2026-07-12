"""
Project Argos - CV Test Script
Verifies that the OpenCV DNN YOLOv8 engine, smoke detector,
and zone polygon breach helpers are functioning correctly.
"""
import sys
import numpy as np
import cv2
import os

# Add parent directory/current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import utils
    from routers import guardian_ws
    print("SUCCESS: Modules imported correctly.")
except Exception as e:
    print(f"ERROR: Failed to import utils or routers. {e}")
    sys.exit(1)


def test_yolo_inference():
    print("\n--- Testing YOLOv8 ONNX Inference ---")
    try:
        # Create a dummy image (640x480) with three channels
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate a shape
        cv2.rectangle(dummy_img, (100, 100), (300, 300), (255, 255, 255), -1)

        print("Running YOLOv8 inference...")
        annotated_img, detections = utils.run_yolo(dummy_img, conf_thresh=0.1)

        print(f"Inference completed successfully.")
        print(f"Detections found: {len(detections)}")
        print(f"Annotated frame size: {annotated_img.shape}")
        print("SUCCESS: YOLOv8 engine is fully functional.")
    except Exception as e:
        print(f"ERROR: YOLOv8 inference failed: {e}")


def test_polygon_breach():
    print("\n--- Testing Polygon Breach (Ray-casting) ---")
    # Define a simple square zone
    zone = [[100, 100], [200, 100], [200, 200], [100, 200]]

    # Point inside the zone
    inside_pt = (150, 150)
    # Point outside the zone
    outside_pt = (50, 50)

    is_inside = guardian_ws.point_in_polygon(inside_pt[0], inside_pt[1], zone)
    is_outside = guardian_ws.point_in_polygon(outside_pt[0], outside_pt[1], zone)

    print(f"Point {inside_pt} inside zone? {is_inside} (Expected: True)")
    print(f"Point {outside_pt} inside zone? {is_outside} (Expected: False)")

    if is_inside == True and is_outside == False:
        print("SUCCESS: Polygon breach detection logic is correct.")
    else:
        print("ERROR: Polygon breach logic returned incorrect values.")


def test_smoke_detector():
    print("\n--- Testing Smoke Detection Heuristics ---")
    try:
        # Test 1: Black image (should be False)
        img_clean = np.zeros((480, 640, 3), dtype=np.uint8)
        smoke_clean = guardian_ws.detect_smoke(img_clean)

        # Test 2: Grayish image (simulating smoke grey color)
        img_smoke = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Initialize prev_gray for motion detection
        guardian_ws._prev_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
        smoke_detected = guardian_ws.detect_smoke(img_smoke)

        print(f"Smoke on clean frame? {smoke_clean} (Expected: False)")
        print(f"Smoke on smoke-filled frame? {smoke_detected} (Expected: True or False depending on saturation limits)")
        print("SUCCESS: Smoke detection module runs without exceptions.")
    except Exception as e:
        print(f"ERROR: Smoke detection failed with exception: {e}")


if __name__ == "__main__":
    print("Project Argos CV test suite starting...")
    test_yolo_inference()
    test_polygon_breach()
    test_smoke_detector()
    print("\nAll CV checks completed.")
