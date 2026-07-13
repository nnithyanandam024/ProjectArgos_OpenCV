#Project Argos — AI Campus Guardian

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
- **YOLOv8n** pre-trained on **COCO 2017** (80 classes)
- Classes used: `person`, `cell phone`, `backpack`, `handbag`, `suitcase`
- No fine-tuning required — all detections use standard COCO weights
- Smoke detection uses classical CV heuristics (HSV + optical flow)

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

| Detection | Method | Accuracy |
|---|---|---|
| Phone Detection | YOLOv8n COCO class 67 | ~85% in good lighting |
| Fall Detection | Aspect ratio heuristic (w/h > 1.4) | ~80% for clear falls |
| Abandoned Bag | Proximity timer (5s, 150px) | High for stationary bags |
| Zone Breach | Ray-casting polygon test | 100% geometric accuracy |
| Smoke Detection | HSV mask + optical flow | Effective for dense smoke |

**Risk Score Formula:**
```
Risk = Σ (weight_i for each active alert type)
     = Phone(15) + Fall(40) + Bag(35) + Zone(30) + Smoke(50)
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

## References
1. Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics (Apache 2.0)
2. COCO Dataset — Lin et al., 2014 — https://cocodataset.org
3. OpenCV — https://opencv.org (BSD License)
4. FastAPI — https://fastapi.tiangolo.com (MIT)
5. React / Vite — https://react.dev / https://vitejs.dev (MIT)
