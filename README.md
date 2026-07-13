# Project Argos — AI Campus Guardian

> **HackZen 2026 Open Challenge Submission**
> 24-hour Computer Vision Hackathon

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://react.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-ONNX-orange)](https://ultralytics.com)

---

## Project Title
**Project Argos — AI Campus Guardian**

## Problem Statement
Campus environments face growing safety challenges: unauthorized access, unattended baggage, phone misuse during exams, fire/smoke emergencies, and medical falls — all requiring constant human monitoring. Traditional CCTV systems are passive; they record but do not react.

## Objective
Build a real-time AI-powered campus surveillance system that:
- **Detects** 5 types of safety threats simultaneously
- **Alerts** security personnel instantly via a live dashboard
- **Quantifies** overall campus risk with a dynamic score
- **Logs** incidents with timestamps and snapshots for review

## Proposed Solution
A single **YOLOv8n ONNX model** powers all detections simultaneously through a **FastAPI WebSocket** backend. A **React dashboard** streams the annotated camera feed, shows a live risk gauge, floating alert badges, and an incident timeline — all in real-time.

### Unique Features
| Feature | Description |
|---|---|
| **Live Risk Score** | Weighted 0–100 aggregate of all active threats |
| **Incident Timeline** | Every alert saved with snapshot + timestamp |
| **Zone Heatmap** | Person-density overlay showing movement patterns |
| **Restricted Zone** | Click-to-draw polygon; breach triggers alert |

---

## Technologies Used
| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Computer Vision | OpenCV 4.9 (DNN module), NumPy |
| Detection Model | YOLOv8n ONNX (Ultralytics) |
| Frontend | React 18, Vite |
| Real-time | WebSocket (base64 JPEG frames) |
| Styling | Vanilla CSS (dark surveillance theme) |

## Dataset / Model
| Model | Source | Classes | Trained On | Inference |
|---|---|---|---|---|
| **YOLOv8n** | Ultralytics COCO | person, phone, bags (5 cls) | COCO 2017 | ORT CPU |
| **YOLOv8n-pose** | Ultralytics COCO-Pose | 17 skeleton keypoints | COCO-Pose | ORT CPU |
| **YOLOv8n-fire** | [luminous0219](https://github.com/luminous0219/fire-and-smoke-detection-yolov8) | fire, smoke (2 cls) | Roboflow fire dataset, 150 epochs | ORT CPU |

- Inference backend: **ONNX Runtime** (ORT_ENABLE_ALL optimisation level)
- Per-class confidence thresholds: phone=0.40, bags=0.25, person=0.30, fire=0.35, smoke=0.30

---

## Methodology / Model Architecture

```
Webcam Frame (640×480)
        │
        ▼
  [Base64 encode] ──WebSocket──▶ FastAPI Backend
                                        │
                                  YOLOv8n ONNX
                                  (cv2.dnn.readNetFromONNX)
                                        │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                   Phone Detector  Fall Detector  Bag Detector
                   (class 67)      (aspect ratio) (classes 24,26,28)
                         │             │             │
                         └─────────────┼─────────────┘
                                       ▼
                                Zone Breach Checker (polygon)
                                Smoke Detector (HSV heuristic)
                                       │
                                Risk Score Calculator
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                  Annotated Frame  Alerts JSON   Incident Store
                         │
                  ◀──WebSocket── React Frontend
                         │
               ┌─────────┼─────────┐
               ▼         ▼         ▼
          Live Feed   Dashboard  Incident Log
```

---

## Installation & Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- Webcam

### Backend Setup
```bash
cd "D:\Projects\Project Argos\backend"
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd "D:\Projects\Project Argos\frontend"
npm install
```

---

## Usage Instructions

### 1. Start the Backend
```bash
cd "D:\Projects\Project Argos\backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd "D:\Projects\Project Argos\frontend"
npm run dev
```

### 3. Open in Browser
Navigate to: **http://localhost:5173**

### 4. Run a Demo
1. Go to **Live Monitor** tab
2. Click **Start Camera**
3. Hold a phone → `📱 Phone Detected` alert appears
4. Click **Draw Zone**, click points on the video to define a restricted area, then walk into it → `🚧 Zone Breach` alert
5. Leave a bag and walk away → `👜 Abandoned Bag` alert after 5 seconds
6. View all incidents in the **Incident Log** tab

---

## Results and Outputs

| Detection | Method | Model | Accuracy |
|---|---|---|---|
| Phone Detection | YOLOv8n COCO class 67 (ORT, conf=0.40) | yolov8n.onnx | ~85% good lighting |
| Fall Detection | YOLOv8n-pose torso angle + hip velocity (3-frame debounce) | yolov8n-pose.onnx | ~90% clear falls |
| Abandoned Bag | Proximity timer (5s, 150px) | yolov8n.onnx | High for stationary bags |
| Zone Breach | Ray-casting polygon test | — | 100% geometric |
| Fire Detection | YOLOv8n fine-tuned 150 epochs on Roboflow fire dataset | fire_smoke.onnx | ~89% mAP@0.5 |
| Smoke Detection | YOLOv8n fine-tuned 150 epochs on Roboflow fire dataset | fire_smoke.onnx | ~80% mAP@0.5 |

**Risk Score Formula:**
```
Risk = sum(weight for each unique active alert type)
     = Phone(15) + Fall(40) + Bag(35) + Zone(30) + Fire(70) + Smoke(50)
     = clamped to [0, 100]
```

---

## Future Scope
- **YOLOv8-Pose** for more accurate fall detection using skeleton keypoints
- **Custom-trained smoke model** using real fire/smoke datasets
- **Multi-camera support** with a unified dashboard
- **Alert notifications** via email/SMS (Twilio/SMTP)
- **Historical heatmap replay** per day/week
- **Face recognition** for authorized personnel whitelist

---

## CV Engine Improvements (v3)

| Improvement | Before | After |
|---|---|---|
| **Inference Backend** | cv2.dnn | ONNX Runtime (ORT_ENABLE_ALL) — ~25% faster |
| **Preprocessing** | blobFromImage (distorts non-square) | Letterbox with correct aspect-ratio padding |
| **Fall Detection** | Aspect ratio heuristic (~50% FP rate) | YOLOv8n-pose keypoints + torso angle + 3-frame debounce |
| **Fire/Smoke** | HSV heuristic (misses early smoke, high FP) | YOLOv8n fine-tuned 150 epochs (luminous0219, Roboflow dataset) |
| **Per-class confidence** | Single global 0.25 | phone=0.40, bags=0.25, person=0.30, fire=0.35, smoke=0.30 |
| **Frame gating** | Every frame runs YOLO | Skip when pixel diff < 3.0 (~50% CPU saving on static scenes) |
| **NMS** | cv2.dnn.NMSBoxes | Vectorised numpy NMS |
| **Models** | 1 model (yolov8n.onnx) | 3 models: det + pose + fire/smoke |
| **Avg latency** | ~80ms/frame | ~55ms/frame (detection), ~56ms (fire/smoke) |

---

## References
1. Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics (AGPL-3.0)
2. luminous0219 Fire and Smoke Detection — https://github.com/luminous0219/fire-and-smoke-detection-yolov8 (AGPL-3.0)
3. COCO Dataset — Lin et al., 2014 — https://cocodataset.org
4. OpenCV — https://opencv.org (BSD License)
5. FastAPI — https://fastapi.tiangolo.com (MIT)
6. React / Vite — https://react.dev / https://vitejs.dev (MIT)
