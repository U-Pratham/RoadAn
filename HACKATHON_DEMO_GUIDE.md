# RoadSight AI — Hackathon Demonstration Guide

---

## 1. Quick Start & Execution Setup

### System Prerequisites
- **Operating System:** Windows 11 (64-bit)
- **Python Environment:** Python 3.11 (`.venv` initialized with `uv`)
- **Hardware Acceleration:** NVIDIA GeForce RTX 2050 (4 GB VRAM) running DirectX 12 / DirectML
- **Local Port:** `8000` (Default HTTP)

### Startup Commands

Open PowerShell or Command Prompt in the repository root (`C:\Users\prath\Downloads\Road`):

```powershell
# 1. Activate project directory
cd C:\Users\prath\Downloads\Road

# 2. Launch FastAPI Production Server with Live Reload
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```

- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **System Health Endpoint:** `http://127.0.0.1:8000/api/health`

---

## 2. Recommended Demonstration Image Sequence

The following verified test images are located in `data/test_images/` for live demonstration:

| Step | File Name | Damage Profile | Expected Detection | Operational Narrative |
| :---: | :--- | :--- | :--- | :--- |
| **Demo 1** | `01.jpg` | Urban Pothole | 2 Potholes (`D40`), Conf: ~51% & 39% | **High/Critical Urgency:** Demonstrates multi-defect bounding boxes, frame area proxy, and severe vehicle risk. |
| **Demo 2** | `07.jpg` | Pristine Paved Road | 0 Hazards (`SAFE / CLEAR`) | **Resilience / Zero False Positives:** Demonstrates system stability when inspecting clean asphalt with no false alarms. |
| **Demo 3** | `02.jpg` | Severe Alligator Cracking | Alligator Cracks (`D20`) | **Structural Fatigue:** Demonstrates pavement degradation scoring and road importance weighting. |
| **Demo 4** | Smartphone / Dashcam capture | Custom Road Photo | Real-time AI inference | **Live Ingestion:** Demonstrates drag-and-drop ingestion with custom GPS coordinates and road classification. |

---

## 3. Step-by-Step Live Demonstration Walkthrough

### Step 1: System Health & Hardware Discovery
- Point to the **Status Indicator Pill** on the top right of the navigation bar.
- Highlight: `NVIDIA GeForce RTX 2050 (DirectML / DX12) • Active`.
- Explain: Model inference runs locally on the GPU via DirectML, ensuring zero cloud dependency and low inference latency ($~12.5$ ms).

### Step 2: Overview & Municipal KPIs
- Show the **Executive Infrastructure Overview** at the top.
- Explain that all statistics (`Total Inspections`, `Total Hazards`, `Critical Count`, `Average Risk Score`, and `Defect Breakdown`) are queried live from SQLite (`roadsight.db`).

### Step 3: Road Damage Detection & Parameter Control
1. In the **New Inspection** workspace, drag and drop `data/test_images/01.jpg` into the dropzone.
2. Notice the immediate client-side preview.
3. Select **Road Category:** `Main Urban Thoroughfare (18 pts)`.
4. Click the coordinate chip: **Bengaluru Urban** (`12.9716, 77.5946`).
5. Set **Confidence Threshold** to `0.25`.
6. Click **Execute AI Road Analysis**.
7. In ~60 ms, the results populate:
   - **Dual Viewer:** Toggle between **Annotated** (showing bounding boxes and labels) and **Original Photograph**.
   - **Risk Gauge:** Displays overall score (~$79/100$) and **HIGH PRIORITY** badge.
   - **Multi-Factor Contribution Breakdown:** Point to the 4 progress bars:
     - *Factor A (Severity):* $40.0/40$ (Pothole hazard)
     - *Factor B (Area Proxy):* $3.1/20$
     - *Factor C (Confidence):* $7.6/15$
     - *Factor D (Road Importance):* $18.0/25$ (Main Urban)
   - **Civil Engineering Advisory:** Highlight that scores are explainable prioritization indices for maintenance teams.

### Step 4: GIS Infrastructure Map
- Scroll down to the **GIS Hazard Map**.
- Point out the newly added color-coded pin on OpenStreetMap at Bengaluru coordinates.
- Click the pin: an interactive popup shows the inspection summary and thumbnail.
- Click **View Detailed Audit** inside the popup to launch the deep-dive inspection modal.

### Step 5: Unlocated Inspections Registry
- Show the sidebar beside the map: **Unlocated Inspections**.
- Explain: If an operator uploads an image without GPS, RoadSight AI preserves coordinate integrity by never fabricating locations—it lists them separately for manual geocoding or audit.

### Step 6: Inspection History Audit
- Scroll down to the **Inspection Registry & Audit Log**.
- Demonstrate the live filters:
  - Type `01.jpg` in the search bar: table filters instantly.
  - Filter by Priority: Select `High Only` or `Critical Only`.
  - Sort by: `Highest Risk` vs `Newest First`.
- Click **Inspect** on any row to open the full-screen modal showing original vs annotated images, metadata, and child hazard bounding-box coordinates.

---

## 4. Fallback & Backup Procedures

| Potential Failure Point | Likelihood | Immediate Backup / Remediation |
| :--- | :---: | :--- |
| **GPU Provider Failure** | Very Low | The backend automatically falls back to `CPUExecutionProvider` without crashing. Health endpoint updates to `CPU Execution Active`. |
| **No Internet Access** | Low | Map controls and empty states remain active. The application can run entirely offline on `localhost`. |
| **Port 8000 Conflict** | Low | Run `.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8080` and access `http://127.0.0.1:8080/`. |
| **Corrupt Image Uploaded** | Medium | The backend catches invalid files, returns `HTTP 415`, and displays a non-blocking error banner on the UI. |

---

## 5. Known Limitations for Judges / Reviewers

1. **Monocular Visual Area Estimation:** Damaged area is calculated as a 2D bounding-box pixel ratio against image dimensions. True volumetric pothole depth requires stereo or LiDAR hardware.
2. **2D Tile Caching:** Map background tiles stream from OpenStreetMap CartoDB dark servers; offline presentations display marker plots and unlocated cards if cached tiles expire.
3. **Frame-by-Frame Ingestion:** System processes individual high-resolution captures; dashcam video stream chunking is roadmapped for production release.
