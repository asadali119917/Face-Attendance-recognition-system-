# ◈ FaceVision AI — University Face Recognition System

> **YOLOv8-powered Face Detection · Segmentation · Classification**  
> Full-stack Python project with a futuristic GUI

---

## 🎯 Features

| Module | Technology | Capability |
|--------|-----------|------------|
| **Detection** | YOLOv8n (COCO) | Person bounding boxes, confidence scores |
| **Segmentation** | YOLOv8n-seg | Instance segmentation masks per person |
| **Classification** | HOG + LBP descriptor | Face identity recognition |
| **Anti-Spoofing** | LBP texture variance | Liveness detection score |
| **Attributes** | CV heuristics | Emotion + age-group estimation |
| **GUI** | Tkinter + ttk | Futuristic cyberpunk-style interface |

---

## 📁 Project Structure

```
FaceRecognitionProject/
├── run.py                  ← Main launcher
├── requirements.txt
├── core/
│   └── face_engine.py      ← YOLOv8 + recognition engine
├── gui/
│   └── main_app.py         ← Full GUI application
├── database/
│   ├── faces.json          ← Enrolled persons (auto-created)
│   └── photos/             ← Registered face photos
├── snapshots/              ← Saved snapshots
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch GUI
python run.py

# 3. Demo (no camera needed)
python run.py --demo

# 4. Process single image
python run.py --headless path/to/image.jpg
```

**First run** will auto-download YOLOv8 weights (~6MB each for nano models).

---

## 🖥️ GUI Tabs

### 1. RECOGNITION
- Live detection info card (name, student ID, department)
- Confidence, liveness, and FPS metric bars
- Real-time subject photo display

### 2. REGISTER
- Capture live frame from camera
- Fill name, student ID, department
- One-click enrollment → stored in JSON database

### 3. DATABASE
- View all enrolled subjects
- Delete selected subject
- Refresh button

### 4. STATS
- Session statistics
- Engine status, YOLOv8 version, FPS

---

## 🔬 Analysis Modes

| Mode | Description |
|------|-------------|
| **Detection** | YOLOv8 draws bounding boxes around detected persons |
| **Segmentation** | YOLOv8 draws colored masks over each person |

Switch modes via the radio buttons above the video feed.

---

## 🧠 Recognition Pipeline

```
Camera Frame
    ↓
YOLOv8n Detection → Person bounding boxes
    ↓
Face region extraction (top 40% of person crop)
    ↓
HOG descriptor encoding (256-d vector)
    ↓
LBPH recognizer → Person ID + confidence
    ↓
Attribute classifier → Emotion + age group
    ↓
Liveness detector → LBP variance score
    ↓
HUD overlay → Draw result on frame
```

---

## 📊 Technical Details

- **YOLOv8 Nano** used for speed (30+ FPS capable)
- **LBPH Face Recognizer** — lightweight, works without GPU
- **HOG features** for face encoding (no heavy ML dependency)
- **Liveness** via LBP texture variance (real faces > 0.5)
- **Database** stored as JSON + numpy arrays (no SQL needed)

---

## 🎓 University Project Info

- **Subject**: Computer Vision / Biometric Systems
- **Techniques**: Object Detection, Semantic Segmentation, Classification
- **Framework**: YOLOv8 (Ultralytics), OpenCV, Python 3.10+
- **GUI**: Tkinter (cross-platform, no install needed)

---

## 📸 Usage Guide

1. **Start the app**: `python run.py`
2. **Click "▶ START CAMERA"** to activate webcam
3. **Register a person**: Go to REGISTER tab → click "📸 CAPTURE FRAME" → fill details → "✅ ENROLL SUBJECT"
4. **Switch modes**: Use Detection / Segmentation buttons
5. **Load image**: Click "🖼 LOAD IMAGE" to analyze a photo file
6. **View database**: DATABASE tab shows all enrolled subjects

---

*FaceVision AI — Built with YOLOv8 + OpenCV + Python*
