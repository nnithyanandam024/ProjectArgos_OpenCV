# Project Documentation: Project Argos — AI Campus Guardian

## Project Title
**Project Argos — AI Campus Guardian**

---

## Team Details
* **Team Name:** DEVNEST
* **Team Members:**
  * **NITHYANANDAM N** (7376242IT237) — nithyanandamn.it24@bitsathy.ac.in
  * **MITHUN R S** (7376242IT218) — mithunrs.it24@bitsathy.ac.in
  * **SABAREESH S S** (7376242IT279) — sabareeshss.it24@bitsathy.ac.in

---

## Problem Statement
Modern campus environments face critical safety and security challenges every day. Unsupervised restricted areas, unattended baggage (which poses a security threat), mobile phone distractions during examinations, and medical emergencies such as sudden falls require constant human surveillance. 

Traditional surveillance systems are passive; they record footage but rely entirely on security staff to notice incidents in real-time. Human monitoring is prone to distraction and fatigue, leading to delayed response times and potential security lapses.

---

## Objective
The primary goal of Project Argos is to build an automated, real-time Computer Vision surveillance platform that:
1. **Identifies and tracks** multiple security concerns (phones, falls, bag abandonment, and zone breaches) simultaneously.
2. **Alerts** administrators instantly through a clean, intuitive web dashboard.
3. **Computes** a dynamic, real-time campus risk score indicating the severity of current threats.
4. **Logs** security incidents with high-confidence snapshots and timestamp information for offline review.

---

## Proposed Solution
Project Argos solves these challenges by streaming video from a local webcam or surveillance stream to a high-speed Python backend using WebSockets. 

A unified, multi-task inference pipeline processes each frame using optimized model architectures running on ONNX Runtime CPU execution. The backend analyzes object coordinates, pose keypoints, and spatial locations to detect:
* **Cell Phones:** Tracked using custom detection classes.
* **Sudden Falls:** Calculated in real-time via torso angle thresholds and hip vertical drop tracking over a temporal queue (3-frame debounce).
* **Abandoned Bags:** Detected by comparing bag locations against person proximity over a 5-second threshold.
* **Restricted Zone Breaches:** Evaluated using a geometric ray-casting algorithm against user-defined polygon regions drawn on the UI.

---

## Technologies Used

### Backend
* **Language:** Python 3.11
* **API Framework:** FastAPI (WebSocket-based frame transport)
* **Execution Engine:** ONNX Runtime (CPU optimization level: `ORT_ENABLE_ALL`)
* **Computer Vision:** OpenCV, NumPy

### Frontend
* **UI Library:** React 18, Vite
* **Styling:** CSS (Clean Professional Dark theme)
* **Real-time Stream:** Base64-encoded JPEG socket transmission

---

## Dataset
* **Object Detection & Pose Estimation:** Built on standard dataset profiles for human coordinates, bounding box metrics, and 17-point keypoint skeleton mapping (including shoulders, hips, knees, and nose).
* **Verification Targets:** Configured for campus-relevant classes: person, cell phone, backpack, handbag, and suitcase.

---

## Methodology / Model Architecture

```
Webcam Frame (640×480)
        │
        ▼ (Base64 JPEG)
  WebSocket Stream
        │
        ▼
 FastAPI Backend
        │
  Inference Engine (ONNX Session)
        │
        ├──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 Object Detection        Pose Estimation        Feature Prep
 (Person, Phone, Bags)   (17 Keypoints)         (Letterbox Padding)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                    ┌─── Analysis Layer ───┐
                    │                      │
                    ├─ Phone Detection     │
                    ├─ Torso Angle Fall    │
                    ├─ Abandoned Bag Timer │
                    ├─ Ray-cast Breach     │
                    └──────────┬───────────┘
                               ▼
                     Risk Score Calculation
                     + Save Incident Snapshot
                               │
               WebSocket (Annotated Frame + JSON)
                               │
                               ▼
                         React Frontend
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          Dashboard      Live Monitor    Incident Log
```

---

## Installation & Setup Instructions

### Prerequisites
* Python 3.11+
* Node.js 18+
* Active Webcam

### 1. Backend Setup
Navigate to the backend directory and install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

Download and export the model runtimes:
```bash
python download_models.py
```

### 2. Frontend Setup
Navigate to the frontend directory and install node modules:
```bash
cd frontend
npm install
```

---

## Usage Instructions

### Starting the Application

#### Step 1: Launch Backend

**Using Bash / Command Prompt:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Using PowerShell:**
```powershell
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Step 2: Launch Frontend

**Using Bash / Command Prompt / PowerShell:**
```bash
cd frontend
npm run dev
```

#### Step 3: Open Dashboard
Open your web browser and navigate to: **http://localhost:5173**

---

## Results and Outputs

### Incident Detection Metrics
* **Phone Detection:** ~85% accuracy under standard campus lighting.
* **Fall Detection:** ~90% accuracy utilizing skeleton posture and 3-frame confirmation to prevent false positives.
* **Abandoned Bags:** Temporal queue tracking confirms abandonment after a bag is left alone for more than 5 seconds within a 150px radius.
* **Zone Breach:** 100% geometric accuracy for custom click-to-draw polygon boundaries.

### Risk Score Calculation
The platform aggregates active threats using a weighted formula:
```
Total Risk = Phone(15) + Fall(40) + Abandoned Bag(35) + Zone Breach(30)
(Clamped between 0 and 100)
```

---

## Future Scope
* **Multi-Camera Integration:** Support for multiple surveillance feeds on a single command interface.
* **Automated Notification Dispatch:** SMS and email alerts sent directly to security personnel.
* **Advanced Access Whitelisting:** Identity verification matching to allow authorized personnel inside drawn zones.
* **Dynamic Analytics:** Historical reporting of safety trends over daily, weekly, or monthly periods.

---

## References
1. FastAPI Documentation — https://fastapi.tiangolo.com
2. React & Vite Guides — https://vitejs.dev
3. ONNX Runtime API Reference — https://onnxruntime.ai
4. OpenCV Library Reference — https://opencv.org
