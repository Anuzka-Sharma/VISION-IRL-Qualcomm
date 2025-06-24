Here’s a complete and polished version of your **GitHub README.md** file for the **VISION IRL** repository. This combines all your information into a clean, professional, and developer/judge-friendly format:

---

# 👁️‍🗨️ VISION IRL – Real-Time Offline Visual Assistant for the Visually Impaired

**VISION IRL** is a **real-time, privacy-focused assistive AI system** designed to empower visually impaired users by helping them understand and interact with their surroundings. It provides **offline object detection**, **voice interaction**, **gesture/threat detection**, and **depth estimation**, all running on-device with **Snapdragon® NPU acceleration**.

🚀 **Demo Video**:
[📹 Watch the Demo](https://drive.google.com/file/d/1bQobFbd8DvbZDF8yxgi3RAKp73mLcX4y/view?usp=sharing)

---

## 🔧 Technical Highlights

### ✅ Snapdragon X Elite NPU Optimization

```python
self.model = YOLO("yolov8m.pt").to('qnn')  # Qualcomm NPU backend
```

* **Latency**: 18–22ms per frame
* **FPS**: 45–50 (NPU) vs 8–12 (CPU)
* **Power Draw**: \~3.2W average

### 🧠 Key Capabilities

* 🔍 **YOLOv8**-based object detection
* 🧍‍♂️ Pose tracking using **MediaPipe**
* 🎤 Offline voice interaction using **Vosk + pyttsx3**
* 🎥 RealSense-based depth estimation
* 🧑‍💼 Face & voice-based authentication
* ⚡ Optimized for **Snapdragon AI Stack** (QNN runtime)
* 🧭 Threat detection via multimodal inputs (gestures + objects)

---

## 🌟 Unique Innovations

### 👁️ Multimodal Threat Detection

```python
def detect_threats(self):
    return {
        'gestures': self.current_gestures,
        'objects': self.detected_objects,
        'suspicious_activity': self.suspicious_activities
    }
```

Combines pose, gesture, and object tracking to detect suspicious behavior or obstacles.

### 🎙️ Voice-First UX

* Wake-word-based zero-touch control
* Runs fully offline for privacy

---

## 🔒 Privacy-Preserving by Design

* **No Cloud Needed** – All AI runs locally
* **Offline Models** – YOLO, MediaPipe, Vosk, FaceNet
* **User Data Stays On-Device**

---

## 💻 System Requirements

| Component       | Details                               |
| --------------- | ------------------------------------- |
| **Device**      | Snapdragon® X Elite (e.g., Dell 7455) |
| **OS**          | Windows 11 ARM64                      |
| **Python**      | 3.12.x                                |
| **Camera**      | Intel® RealSense D435i (or similar)   |
| **NPU Runtime** | Snapdragon® AI SDK + QNN Tools        |

---

## 📦 Installation Guide

### 1. Clone the Repo

```bash
git clone https://github.com/Anuzka-Sharma/VISION-IRL-Copy.git
cd VISION-IRL-Copy
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If PyAudio fails:

```bash
pip install pipwin
pipwin install pyaudio
```

---

## ⚡ Running with QNN (NPU Acceleration)

### Requirements

* Snapdragon® AI SDK: [https://developer.qualcomm.com/software/ai-stack](https://developer.qualcomm.com/software/ai-stack)
* Qualcomm® AI Hub account
* Converted ONNX/TFLite model

### Steps

```bash
qnn-convert --onnx yolov8.onnx --output-dir ./models/qnn
```

Enable QNN backend in code:

```python
from qti.ai.model.api import QNNEngine
model = QNNEngine("models/qnn/yolov8.qnn", backend="npu")
```

Confirm QNN activity via logs:

```
Compute Unit(s): npu (851 ops)
```

---

## 🔁 Real-Time Processing Pipeline

```plaintext
[Camera Input]
     ↓
[YOLOv8 (QNN)] → Object Labels
     ↓
[RealSense] → Depth Map
     ↓
[MediaPipe] → Pose / Gesture
     ↓
[Vosk] ← Microphone Input
     ↓
[Command Logic + pyttsx3] → Voice Output
```

---

## 🧩 Project Structure

```
VISION-IRL/
├── backend/
│   ├── object_detection.py
│   ├── speech_model.py
│   ├── pose_module.py
│   └── auth/
├── models/
│   ├── yolov8.onnx
│   └── yolov8.qnn
├── assets/
├── main.py
├── requirements.txt
└── README.md
```

---

## 📚 Toolchain & Libraries

| Tool/Library       | Purpose                   |
| ------------------ | ------------------------- |
| `ultralytics`      | YOLOv8 object detection   |
| `pyrealsense2`     | Depth estimation          |
| `vosk` + `pyttsx3` | Offline speech I/O        |
| `MediaPipe`        | Pose and gesture tracking |
| `dlib`             | Face recognition          |
| `kivy`             | GUI frontend              |
| `qnn`              | NPU inference backend     |

---

## 🧠 Optional LLM Integration

Use Snapdragon® NPU Edge Chat App or AnythingLLM with QNN-compatible models to enable:

* Scene Q\&A
* Natural language queries
* Personal assistant-style interaction

---

## 📄 License

**MIT License** — Free for non-commercial use with attribution.

---

## 🤝 Team VISION IRL

| Name           | Email                                                             |
| -------------- | ----------------------------------------------------------------- |
| Saumya Kumari  | [saumyakumari2005a@gmail.com](mailto:saumyakumari2005a@gmail.com) |
| Anushka Sharma | [anushka.care@gmail.com](mailto:anushka.care@gmail.com)           |
| Ashmi Suman    | [ashmisuman8113@gmail.com](mailto:ashmisuman8113@gmail.com)       |
| Kanishka Raj   | [kanishkaraj2004@gmail.com](mailto:kanishkaraj2004@gmail.com)     |

---

**Empowering everyone to perceive the world – privately, offline, and in real time.**

---

Let me know if you'd like:

* A badge-style GitHub header
* A `.qnn` conversion example script
* Deployment guide for a .exe
* Submission-ready condensed summary (for judges)

Would you like this auto-converted into a Markdown file (`README.md`) or pushed to GitHub via command line instructions?
