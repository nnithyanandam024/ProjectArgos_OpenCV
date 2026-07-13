"""
Project Argos – Model Downloader
Downloads:
  1. yolov8n-pose.onnx  – from Ultralytics (skeleton keypoints / fall detection)
  2. fire_smoke.onnx    – from luminous0219/fire-and-smoke-detection-yolov8
                          (YOLOv8n trained 150 epochs on fire+smoke dataset)

Run once: python download_models.py
ultralytics is NOT required at runtime — only for exporting the pose model.
"""
import os
import sys
import shutil
import glob
import urllib.request

MODELS_DIR       = os.path.join(os.path.dirname(__file__), "models")
POSE_ONNX        = os.path.join(MODELS_DIR, "yolov8n-pose.onnx")
FIRE_SMOKE_PT    = os.path.join(MODELS_DIR, "fire_smoke_best.pt")
FIRE_SMOKE_ONNX  = os.path.join(MODELS_DIR, "fire_smoke.onnx")

# Direct LFS download URL for luminous0219/fire-and-smoke-detection-yolov8 best.pt
FIRE_SMOKE_PT_URL = (
    "https://github.com/luminous0219/fire-and-smoke-detection-yolov8"
    "/raw/main/weights/best.pt"
)


# ─────────────────────────────────────────────────────────────────────────────

def _progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / max(total_size, 1))
    print(f"\r    {min(pct,100):3d}%", end="", flush=True)


def download_pose_model() -> bool:
    if os.path.exists(POSE_ONNX):
        print(f"[OK] yolov8n-pose.onnx already exists ({os.path.getsize(POSE_ONNX)//1024} KB)")
        return True

    print("[*] Downloading yolov8n-pose.pt and exporting to ONNX ...")
    print("    (requires: pip install ultralytics)")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")
        model.export(format="onnx", imgsz=[640, 640], opset=12, simplify=True)

        candidates = (
            glob.glob(os.path.join(os.getcwd(), "yolov8n-pose.onnx")) +
            glob.glob(os.path.join(os.path.expanduser("~"), "yolov8n-pose.onnx"))
        )
        if candidates:
            shutil.move(candidates[0], POSE_ONNX)
            print(f"[OK] Saved -> {POSE_ONNX}")
            return True
        else:
            print("[!] Could not locate exported ONNX file.")
            return False
    except ImportError:
        print("[!] ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"[!] Pose export failed: {e}")
        return False


def download_fire_smoke_model() -> bool:
    """
    Download best.pt from luminous0219/fire-and-smoke-detection-yolov8
    then export to ONNX using ultralytics.
    Classes: 0=fire, 1=smoke  (2-class, YOLOv8n, 150 epochs)
    """
    if os.path.exists(FIRE_SMOKE_ONNX):
        print(f"[OK] fire_smoke.onnx already exists ({os.path.getsize(FIRE_SMOKE_ONNX)//1024} KB)")
        return True

    # Step 1: download best.pt if needed
    if not os.path.exists(FIRE_SMOKE_PT):
        print(f"[*] Downloading fire_smoke_best.pt from luminous0219 GitHub ...")
        try:
            urllib.request.urlretrieve(FIRE_SMOKE_PT_URL, FIRE_SMOKE_PT, _progress)
            print()  # newline after progress
            print(f"[OK] Downloaded -> {FIRE_SMOKE_PT} ({os.path.getsize(FIRE_SMOKE_PT)//1024} KB)")
        except Exception as e:
            print(f"\n[!] Download failed: {e}")
            print("    Manual fix: download best.pt from")
            print("    https://github.com/luminous0219/fire-and-smoke-detection-yolov8/tree/main/weights")
            print(f"    and place it at: {FIRE_SMOKE_PT}")
            return False
    else:
        print(f"[OK] fire_smoke_best.pt already downloaded ({os.path.getsize(FIRE_SMOKE_PT)//1024} KB)")

    # Step 2: export to ONNX
    print("[*] Exporting fire_smoke_best.pt to ONNX ...")
    try:
        from ultralytics import YOLO
        model = YOLO(FIRE_SMOKE_PT)
        export_path = model.export(
            format="onnx",
            imgsz=[640, 640],
            opset=12,
            simplify=True,
        )
        # export_path is the ONNX file path returned by ultralytics
        if export_path and os.path.exists(str(export_path)):
            shutil.copy(str(export_path), FIRE_SMOKE_ONNX)
        else:
            # Fallback: search for the exported file
            candidates = (
                glob.glob(os.path.join(os.path.dirname(FIRE_SMOKE_PT), "*.onnx")) +
                glob.glob(os.path.join(os.getcwd(), "*.onnx"))
            )
            # Exclude pose model
            candidates = [c for c in candidates if "pose" not in c.lower()]
            if candidates:
                shutil.copy(candidates[0], FIRE_SMOKE_ONNX)
            else:
                print("[!] Could not locate exported fire/smoke ONNX file.")
                return False

        print(f"[OK] Saved -> {FIRE_SMOKE_ONNX}")
        return True
    except ImportError:
        print("[!] ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"[!] Fire/smoke export failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 55)
    print("  Project Argos – Model Downloader")
    print("=" * 55)

    ok1 = download_pose_model()
    print()
    ok2 = download_fire_smoke_model()

    print()
    print("=" * 55)
    print(f"  Pose model:       {'[OK]' if ok1 else '[FAIL]'}")
    print(f"  Fire/smoke model: {'[OK]' if ok2 else '[FAIL]'}")
    print("=" * 55)

    sys.exit(0 if (ok1 and ok2) else 1)
