# Deepfake Detector — Computer Vision Media Forensics Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MTCNN](https://img.shields.io/badge/MTCNN-Face_Detection-0284C7?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Task-Binary_Classification-10B981?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Research-F59E0B?style=for-the-badge)

A computer vision research and inference pipeline engineered to detect synthetic and manipulated facial media (deepfakes). The system processes video streams and image collections through automated frame decomposition, facial landmark extraction via MTCNN, standard tensor normalization, and deep neural binary classification.

---

## 🏛️ Pipeline Architecture

```mermaid
flowchart TD
    InputMedia["Input Video / Image Stream"] --> FrameExtract["OpenCV Frame Decomposition"]
    FrameExtract --> MTCNN["MTCNN Facial Localization & Cropping"]
    MTCNN --> Normalization["Tensor Preprocessing (224x224, ImageNet Norm)"]
    Normalization --> DeepNet["Fine-Tuned PyTorch CNN Architecture"]
    DeepNet --> Sigmoid["Sigmoid Probability Head"]
    Sigmoid --> Threshold{"Confidence Threshold (0.50)"}
    Threshold -->|Score >= 0.50| Manipulated["Flagged: Manipulated / Deepfake"]
    Threshold -->|Score < 0.50| Authentic["Verified: Authentic Media"]
    Manipulated --> Metrics["Frame-Level Classification & Logging"]
    Authentic --> Metrics
```

---

## 🔬 Key Engineering Features

1. **Automated Frame Extraction:** Decomposes video streams into sequential image frames using OpenCV (`cv2.VideoCapture`) with configurable sampling rates (`practice_coding/02_vid_to_frames.py`).
2. **MTCNN Facial Landmark Cropping:** Eliminates background noise by isolating facial regions with bounding-box margin buffers (`practice_coding/cropping_images_mtcnn.py`).
3. **Fine-Tuning & Model Training:** Custom PyTorch training pipeline implementing Cross-Entropy loss, Adam optimization, learning rate scheduling, and data augmentations (`Training_new/fine_tuning.py`).
4. **Benchmark Evaluation:** Evaluated against benchmark media criteria, tracking precision, recall, and frame-level ROC-AUC (`Research/Research_papers_Secondary_data/`).

---

## 📁 Directory Structure

```
Deepfake_detector/
├── Training_new/
│   ├── app.py                     # Inference entrypoint and evaluation web app
│   ├── deepfake_detector.py       # Core detector inference class
│   ├── evaluate_model.py          # Validation metrics calculation script
│   └── fine_tuning.py             # PyTorch model fine-tuning routine
├── Research/
│   └── Research_papers_Secondary_data/
│       └── DeepFake-Eval-2024 Benchmark.txt
└── practice_coding/
    ├── 01_image_cropping.py       # OpenCV baseline image processing
    ├── 02_vid_to_frames.py        # Video decomposition module
    ├── cropping_images_mtcnn.py   # MTCNN face alignment and isolation
    └── file_handling_and_image.py # Dataset parsing utility
```

---

## 🚀 Quickstart & Execution

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/sultanriaz/Deepfake_detector.git
cd Deepfake_detector
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python facenet-pytorch numpy pandas matplotlib
```

### 3. Run Inference & Web Interface
```bash
python Training_new/app.py
```
