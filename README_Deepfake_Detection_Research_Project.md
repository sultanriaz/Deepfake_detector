# Deepfake Video Detection System  
## A Research-Oriented, Laptop-Efficient Prototype for Detecting Manipulated Facial Video Content

**Project Type:** Final Year Project / Research-Based Experimental Prototype  
**Primary Language:** Python  
**Author:** Sultan Mehmood  
**Status:** Experimental research prototype — not production-ready for universal deepfake detection

---

## 1. Project Overview

This project explores the detection of deepfake video content using a Python-based workflow designed to run within normal student laptop constraints. The system was developed to process video frames, detect and crop faces, train classification models, evaluate predictions, and provide a simple application interface for testing uploaded media.

The initial objective was to build a working deepfake detector capable of classifying manipulated facial video content as **real** or **fake**. During development, the trained models produced strong results on the available training and testing data, with reported accuracy above 90% in the experimental setting. However, testing against newer and more sophisticated AI-generated deepfake videos exposed a major limitation: some modern fake samples were classified as real.

This limitation became the most important finding of the project.

Rather than presenting the system as a complete deepfake detection solution, this repository should be understood as a **research-based investigation into model generalization, dataset limitations, and the difficulty of detecting rapidly evolving synthetic media**.

---

## 2. Research Motivation

Deepfake generation methods are improving quickly. A detector trained on one dataset or manipulation style may perform well on familiar samples while failing on new generation methods, compression patterns, lighting conditions, identities, or video sources.

This project investigates the following practical question:

> **Can a laptop-efficient deepfake detection pipeline maintain reliable performance when tested on newer, unseen AI-generated video content?**

The experiments indicated that high evaluation accuracy on known or similar data does **not** automatically imply reliable detection of recent real-world deepfakes.

---

## 3. Key Outcome and Honest Project Claim

### What Worked

- A Python-based preprocessing and training pipeline was developed.
- Videos and images could be processed into usable frame-level inputs.
- Facial regions could be extracted for model input using face-detection experiments such as MTCNN-based cropping.
- Model training, evaluation, prediction, and application-testing scripts were created.
- A saved trained model artifact was produced.
- A simple interface structure was prepared for testing predictions and displaying metrics.
- The experimental model achieved reportedly strong accuracy on the evaluated dataset/split.

### What Did Not Generalize Reliably

- Newer, highly realistic AI-generated deepfake samples were sometimes predicted as **real**.
- Accuracy measured on the available test data did not fully represent performance on unseen deepfake generation methods.
- The model was not sufficient for dependable real-world deployment.

### Correct Interpretation

This project demonstrates:

- practical implementation of a deepfake detection pipeline;
- experimentation with laptop-efficient model development;
- understanding of frame extraction, face preprocessing, training and evaluation;
- evidence of a real research problem: **cross-dataset and cross-generation generalization failure**.

It does **not** claim:

- universal deepfake detection;
- production-grade reliability;
- guaranteed detection of modern synthetic media;
- deployment readiness for forensic, legal, or moderation decisions.

---

## 4. Project Objectives

The project was designed around the following objectives:

1. Extract image frames from input video files.
2. Detect and crop facial regions for model training and inference.
3. Organize data into training and testing structures.
4. Train laptop-efficient deepfake classification models.
5. Evaluate classification results using recorded metrics.
6. Load trained artifacts for prediction on uploaded samples.
7. Explore model behavior on previously unseen and newer deepfake content.
8. Study the limitations of accuracy-based evaluation when the test distribution differs from real-world inputs.

---

## 5. Technology Stack

| Area | Technology / Approach | Purpose |
| --- | --- | --- |
| Primary language | Python | End-to-end implementation and experimentation |
| Video/image processing | OpenCV-based scripts | Reading videos, extracting frames and testing media processing |
| Face preprocessing | MTCNN experiments | Detecting and cropping facial regions |
| Classical ML experimentation | Logistic Regression, XGBoost and other laptop-efficient approaches | Establishing manageable baseline classifiers |
| Model experimentation | Python training and evaluation scripts | Training, testing and metric collection |
| Saved model artifact | `best_deepfake_model.pth` | Loading the selected trained model for later testing |
| Application layer | Python application scripts and HTML templates | Upload/prediction interface and metrics display |
| Experiment records | CSV and JSON metrics files | Recording model-training results and evaluation outcomes |
| Literature review | Secondary research-paper collection | Studying deepfake detection techniques and limitations |

> **Note:** The repository structure indicates multiple experimental branches and scripts created during development. Some scripts may represent earlier trials, baselines, or alternative approaches rather than one final unified pipeline.

---

## 6. High-Level System Workflow

```mermaid
flowchart LR
    A[Input Video or Image] --> B[Frame Extraction]
    B --> C[Face Detection and Cropping]
    C --> D[Preprocessed Face Samples]
    D --> E[Training / Evaluation Pipeline]
    E --> F[Saved Model Artifact]
    F --> G[Prediction Application]
    G --> H[Real or Fake Prediction]
    H --> I[Metrics and Error Analysis]
    I --> J[Generalization Testing on New Deepfakes]
```

### Experimental Pipeline

1. **Input handling:** media files are loaded for preprocessing or prediction.
2. **Frame extraction:** video inputs are converted into selected image frames.
3. **Face-focused preprocessing:** detected facial areas are cropped to reduce irrelevant visual background content.
4. **Dataset preparation:** samples are divided into training/testing arrangements.
5. **Training:** classification models are trained using processed samples or extracted features.
6. **Evaluation:** accuracy and other available metrics are recorded.
7. **Inference testing:** uploaded or held-out samples are classified as real or fake.
8. **Failure analysis:** incorrect classifications on newer AI-generated deepfakes are studied as research evidence.

---

## 7. Repository Structure

The following structure is reconstructed from the available project files and folders.

```text
deepfake-detection-project/
│
├── 01_image_cropping.py
├── 02_vid_to_frames.py
├── chk.py
├── cropping_images_mtcnn.py
├── file_handling_and_image.py
├── folder_splitting.py
├── logistic_regression.py
├── model_training_dd2k25.py
├── modelTraining.py
├── mtcnn_practice.py
├── practice1.py
├── test_opencv.py
├── videocapture.py
│
├── Research_papers_Secondary_data/
│   ├── A Hybrid Deep Learning and Forensic Approach for Robust Deepfake Detection.pdf
│   ├── Audio Deepfake Detection What Has Been Achieved and What Lies Ahead.pdf
│   ├── Bridging the Gap A Framework for Real-World Video Deepfake Detection
│   │   via Social Network Compression Emulation.pdf
│   ├── Deepfake Detection that Generalizes Across Benchmarks.pdf
│   ├── Deepfake video detection challenges and opportunities.pdf
│   ├── DeepFake-Eval-2024 Benchmark.txt
│   ├── FaceForensics++ Learning to Detect Manipulated Facial Images.pdf
│   ├── Global Multimedia Deepfake Detection Challenge 2024.pdf
│   ├── Improving Fairness in Deepfake Detection.pdf
│   └── International Journal of Intelligent Systems - 2025 - Liu -
│       A Review of Deepfake and Its Detection From Generative.pdf
│
├── Training_new/
│   ├── app.py
│   ├── deepfake_detector.py
│   ├── evaluate_model.py
│   ├── fine_tuning.py
│   ├── training_metrics.csv
│   ├── training_metrics.json
│   └── templates/
│       └── index.html
│
└── Testing_1/
    ├── App.py
    ├── datasplitting.py
    ├── evaluate.py
    ├── predict.py
    ├── test_model.py
    ├── requirement.txt
    │
    ├── models/
    │   └── best_deepfake_model.pth
    │
    ├── static/
    │   └── uploads/
    │       └── extracted/tested_frame_images.jpg
    │
    └── templates/
        ├── index.html
        └── metrics.html
```

---

## 8. File and Folder Purpose

### Root-Level Experimental Scripts

| File | Likely Purpose in the Project Workflow |
| --- | --- |
| `01_image_cropping.py` | Initial image cropping experiments or dataset preprocessing |
| `02_vid_to_frames.py` | Extracting frames from video files |
| `cropping_images_mtcnn.py` | Face detection/cropping using MTCNN |
| `file_handling_and_image.py` | Managing input files and image-processing operations |
| `folder_splitting.py` | Organizing or splitting dataset folders |
| `logistic_regression.py` | Classical machine learning baseline experiment |
| `model_training_dd2k25.py` | Model training experiment for a deepfake dataset/version |
| `modelTraining.py` | Earlier or alternative training workflow |
| `mtcnn_practice.py` | Face detection experimentation |
| `test_opencv.py` | OpenCV setup and media-processing testing |
| `videocapture.py` | Video reading or camera/video-capture experiment |
| `chk.py`, `practice1.py` | Development/testing utilities or preliminary experiments |

### `Training_new/`

This folder appears to represent a later training and evaluation workflow.

| File | Purpose |
| --- | --- |
| `deepfake_detector.py` | Core detector/model workflow |
| `fine_tuning.py` | Fine-tuning or model refinement experiments |
| `evaluate_model.py` | Model evaluation logic |
| `training_metrics.csv` | Tabular record of experimental results |
| `training_metrics.json` | Structured metric output for application/reporting use |
| `app.py` | Application entry point for a training/demo interface |
| `templates/index.html` | Front-end page for interacting with the application |

### `Testing_1/`

This folder appears to contain the inference and testing application.

| File | Purpose |
| --- | --- |
| `App.py` | Application entry point for testing uploaded media |
| `predict.py` | Prediction/inference logic |
| `evaluate.py` | Test evaluation workflow |
| `test_model.py` | Model testing script |
| `datasplitting.py` | Dataset splitting or test-data preparation |
| `models/best_deepfake_model.pth` | Selected saved model artifact |
| `templates/index.html` | Upload or prediction user interface |
| `templates/metrics.html` | Metrics/results display page |
| `static/uploads/` | Uploaded or extracted media frames used during testing |
| `requirement.txt` | Project dependency specification |

### `Research_papers_Secondary_data/`

This folder documents the research-based direction of the project and includes papers related to:

- deepfake detection benchmarks;
- generalization across datasets;
- face manipulation detection;
- fairness considerations;
- real-world compression effects;
- recent deepfake detection challenges;
- broader literature review material.

---

## 9. Installation and Setup

### Prerequisites

- Python 3.9 or later recommended
- A virtual environment
- Dependencies listed in the testing implementation requirement file
- A normal laptop capable of running Python image/video preprocessing and model inference

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

From the repository root:

```bash
pip install -r Testing_1/requirement.txt
```

Because this repository contains experimental branches, additional packages may be required depending on which script is executed. Common expected dependencies for this type of workflow may include:

```bash
pip install numpy pandas opencv-python scikit-learn xgboost matplotlib pillow flask torch torchvision mtcnn
```

> The exact dependency list should be verified from the project environment before public release.

---

## 10. Running the Project

The screenshots indicate two possible application entry points. Use the implementation branch that matches the saved model and dependencies in your local copy.

### Testing / Prediction Application

```bash
python Testing_1/App.py
```

Expected purpose:

- open the testing interface;
- upload or select media;
- extract/process frames;
- load the trained model;
- generate deepfake predictions;
- display available metrics or prediction results.

### Training/Experiment Application

```bash
python Training_new/app.py
```

Expected purpose:

- access training-related functionality or demonstration interface;
- review recorded training metrics;
- run evaluation utilities where supported.

### Run Individual Evaluation or Prediction Scripts

```bash
python Testing_1/predict.py
python Testing_1/evaluate.py
python Training_new/evaluate_model.py
```

> Some scripts may require dataset paths, input media files, or model paths to be configured inside the script before execution.

---

## 11. Model Development Approach

The project was designed with laptop constraints in mind. Rather than assuming access to costly GPU infrastructure or very large production-scale datasets, the work focused on practical experimentation with efficient approaches.

### Approaches Explored

- image and video frame preprocessing;
- facial-region extraction using MTCNN experimentation;
- lightweight baseline classification using models such as Logistic Regression;
- laptop-efficient model experimentation including XGBoost and related approaches;
- trained model loading for application inference;
- evaluation and error analysis through recorded metrics and sample predictions.

### Why Laptop-Efficient Models Matter

For a student project, laptop-efficient methods make it possible to:

- complete repeated experiments without expensive cloud infrastructure;
- compare preprocessing and model choices;
- inspect model failures;
- reproduce results locally;
- focus on research questions rather than infrastructure expense.

However, smaller or efficient models may also have reduced ability to generalize across highly realistic, unseen deepfake generation methods unless they are supported by stronger datasets, augmentation, evaluation protocols and architectures.

---

## 12. Evaluation and Results

### Reported Experimental Result

During project development, the model achieved **above 90% accuracy** on the available experimental evaluation data.

### Important Qualification

This reported result should be interpreted only within the evaluation setup used during training and testing. It does not prove reliable performance against:

- newly generated deepfake videos;
- unseen manipulation methods;
- social-media compression artifacts;
- different camera qualities or video resolutions;
- new identities or lighting conditions;
- adversarial or carefully generated synthetic samples.

### Observed Failure Case

When tested against newer AI-generated deepfake videos, the model sometimes classified fake samples as **real**.

This result is not merely an implementation failure; it is a central research observation:

> A deepfake detector can score highly on a familiar test split while still failing on modern, unseen synthetic media.

---

## 13. Research Findings

### Finding 1: Accuracy Alone Is Not Enough

A high accuracy score on one dataset does not establish robustness. For deepfake detection, the evaluation must consider whether the test set contains manipulation types and visual conditions that differ from the training data.

### Finding 2: Generalization Is the Central Challenge

Modern deepfake generators may remove or reduce the artifacts learned by earlier detectors. A model can learn dataset-specific patterns rather than true, durable indicators of manipulation.

### Finding 3: Real-World Inputs Differ from Training Data

Videos encountered in actual use may be resized, compressed, filtered, re-recorded, or generated by newer systems. These transformations can significantly change classifier behavior.

### Finding 4: Negative Results Are Valuable

The model's inability to reliably detect some recent deepfake content is an academically meaningful result. It identifies a gap between controlled evaluation and real-world generalization, which is one of the major open challenges in deepfake detection research.

---

## 14. Limitations

This project has the following known limitations:

| Limitation | Impact |
| --- | --- |
| Dataset-dependent evaluation | Strong scores may not transfer to unseen deepfake types |
| Limited laptop-scale experimentation | Restricts model size, repeated training and large benchmark testing |
| Frame-level classification limitations | May miss temporal inconsistencies across video sequences |
| New-generation deepfake failure cases | Reduces real-world trustworthiness |
| Possible class imbalance or split bias | Can make reported accuracy appear stronger than general robustness |
| Limited cross-dataset evaluation | Prevents strong conclusions about generalization |
| No forensic deployment validation | Results must not be used as evidence for high-stakes decisions |

---

## 15. Recommended Future Improvements

The following improvements would convert the initial prototype into a stronger research study.

### 15.1 Improve Dataset Design

- Use multiple datasets rather than relying on one source.
- Separate training data from completely unseen generation methods.
- Include compressed, resized and social-media-style transformed videos.
- Document dataset class balance and identity overlap.

### 15.2 Evaluate Generalization Properly

Report results across:

- in-dataset testing;
- cross-dataset testing;
- unseen-generator testing;
- compressed-video testing;
- recent AI-generated sample testing.

Recommended metrics include:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- confusion matrix;
- false-negative rate, especially fake videos predicted as real.

### 15.3 Move Beyond Frame-Only Evidence

Possible later extensions include:

- temporal consistency analysis across sequences of frames;
- frequency-domain artifact analysis;
- facial landmark inconsistency features;
- ensemble methods;
- video-level aggregation of frame predictions;
- multimodal analysis where audio manipulation is also relevant.

### 15.4 Record Reproducible Experiments

For each experiment, store:

- dataset/source;
- preprocessing settings;
- frame sampling strategy;
- train/test split method;
- model name and hyperparameters;
- saved artifact version;
- metrics;
- failure cases;
- sample prediction screenshots.

### 15.5 Improve the Application Layer

- show prediction confidence carefully, without overstating certainty;
- allow multiple-frame video-level result aggregation;
- store test history;
- display model version used;
- include a warning that output is experimental;
- provide error analysis for false negatives.

---

## 16. Suggested Experimental Evaluation Table

This table can be completed from actual run results before submission or public release.

| Experiment | Model / Approach | Test Data Type | Accuracy | Precision | Recall | F1 | Key Observation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline 1 | Logistic Regression | Known test split | To be added | To be added | To be added | To be added | Baseline performance |
| Experiment 2 | XGBoost / efficient model | Known test split | Reported >90%* | To be added | To be added | To be added | Strong controlled result |
| Experiment 3 | Selected saved model | New AI-generated fake videos | To be added | To be added | To be added | To be added | Some fakes classified as real |
| Experiment 4 | Selected saved model | Compressed/resized samples | To be added | To be added | To be added | To be added | Generalization test |

\* Reported during development; final numerical evidence should be inserted from saved evaluation outputs or reproduced runs.

---

## 17. Example Prediction Flow

```text
User uploads video
        ↓
System reads video and samples frames
        ↓
Faces are detected and cropped from frames
        ↓
Frames are preprocessed for the selected model
        ↓
Saved model generates frame-level predictions
        ↓
Predictions are combined or displayed
        ↓
Application reports Real/Fake result with experimental warning
        ↓
Failure cases are recorded for research analysis
```

---

## 18. Ethical and Responsible Use Statement

Deepfake detection involves sensitive judgments about media authenticity. This prototype is intended for **academic experimentation and research learning only**.

It should not be used as the sole basis for:

- legal evidence;
- identity verification;
- public accusations;
- content-removal decisions;
- security screening;
- misinformation claims;
- forensic conclusions.

A prediction of **real** does not prove that a video is authentic, and a prediction of **fake** does not prove manipulation without further expert investigation.

---

## 19. Research Contribution of This Project

Although the original practical objective was not fully achieved, the project provides meaningful academic value:

- It demonstrates an implemented end-to-end experimental deepfake detection workflow.
- It records the difficulty of obtaining robust performance with laptop-efficient approaches.
- It identifies the gap between high test-set accuracy and failure on recent AI-generated fakes.
- It shifts the project from a simple application claim toward an evidence-based research question about generalization.
- It provides a foundation for future cross-dataset evaluation and model robustness studies.

### Suitable Final-Year Project Framing

A defensible project title would be:

> **An Experimental Investigation of Generalization Limitations in Laptop-Efficient Deepfake Video Detection Models**

Alternative title:

> **Deepfake Video Detection Using Python: A Prototype Study of Model Performance and Failure on Unseen AI-Generated Content**

---

## 20. Literature and Secondary Research Material

The repository contains secondary research material on topics including:

- robust deepfake detection;
- audio deepfake detection;
- real-world compression effects;
- benchmark generalization;
- FaceForensics++;
- fairness in deepfake detection;
- recent multimedia deepfake detection challenges;
- broad reviews of generative deepfake detection research.

These materials support the research-oriented interpretation of the project and provide a foundation for discussing why modern deepfake detection remains difficult.

---

## 21. Reproducibility Checklist

Before presenting this repository publicly, complete the following:

- [ ] Confirm the exact dataset names and licenses.
- [ ] Record the final preprocessing pipeline.
- [ ] Document frame sampling and face-cropping configuration.
- [ ] Identify the final selected model architecture or algorithm.
- [ ] Add confirmed package versions to `requirements.txt`.
- [ ] Re-run evaluation and export final metrics.
- [ ] Add a confusion matrix and sample failure analysis.
- [ ] State whether train/test identities overlap.
- [ ] Test at least one unseen or recent deepfake sample set.
- [ ] Add screenshots of the working application.
- [ ] Remove temporary uploads or private media files.
- [ ] Add model-file handling instructions if the artifact is too large for GitHub.
- [ ] Add dataset/privacy and responsible-use notes.

---

## 22. Suggested Repository Cleanup Before GitHub Publication

The current structure reflects active experimentation. For a professional GitHub portfolio version, it can later be reorganized as follows:

```text
deepfake-generalization-study/
│
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── error_analysis.ipynb
├── research/
│   └── literature_notes.md
├── src/
│   ├── preprocessing/
│   │   ├── extract_frames.py
│   │   └── crop_faces.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict.py
│   └── app/
│       └── app.py
├── templates/
├── static/
└── results/
    ├── metrics.csv
    ├── metrics.json
    └── failure_analysis.md
```

This proposed cleanup is not a claim that the repository is already structured this way; it is a recommended next step for presentation and reproducibility.

---

## 23. Portfolio and Resume-Safe Description

### Recommended Portfolio Description

Built a Python-based experimental deepfake video detection prototype involving video-frame extraction, face preprocessing, model training, prediction testing, and metric recording under laptop constraints. Achieved strong evaluation results on the available controlled data, then conducted failure analysis after newer AI-generated deepfake samples were misclassified as real. Reframed the work as a research study on model generalization limitations rather than claiming deployment-ready detection.

### Resume Bullet

- Developed a Python-based experimental deepfake video detection pipeline with frame extraction, facial-region preprocessing, laptop-efficient model training and inference testing; analyzed generalization failures when newer unseen AI-generated fake samples were predicted as real despite strong controlled evaluation results.

### Claims to Avoid

Do **not** claim:

- “Built an accurate production deepfake detector.”
- “Detects all AI-generated videos.”
- “Achieved 90%+ real-world deepfake detection accuracy.”
- “Developed a forensic-grade detection platform.”
- “Solved modern deepfake detection.”

---

## 24. Current Project Status

| Component | Status |
| --- | --- |
| Frame extraction experimentation | Implemented / present in repository |
| Face-cropping experimentation | Implemented / present in repository |
| Model training scripts | Implemented / present in repository |
| Metrics files | Present in training folder |
| Trained model artifact | Present in testing folder |
| Prediction/testing workflow | Implemented / present in repository |
| Simple application templates | Present in repository |
| Strong controlled evaluation result | Reported during development |
| Reliable recent-deepfake generalization | Not achieved |
| Production readiness | Not claimed |
| Research-based interpretation | Appropriate and recommended |

---

## 25. Conclusion

This project began as an attempt to build a functional deepfake video detector using Python and laptop-efficient approaches. It successfully reached the stage of data preprocessing, training, evaluation, saved-model testing and simple application integration. However, the detector's failure on newer AI-generated deepfake videos revealed a deeper and more valuable conclusion: **modern deepfake detection is fundamentally a generalization problem, and high controlled-test accuracy does not guarantee real-world reliability**.

For that reason, this repository is best presented not as a failed production application, but as an honest research-oriented prototype and learning outcome in the continuing challenge of detecting synthetic media.

---

## Author

**Sultan Mehmood**  
BS Computer Science Student  
Project focus: Python-based machine learning experimentation, deepfake detection and model generalization analysis
