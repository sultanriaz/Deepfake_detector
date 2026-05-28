DEEPFAKE VIDEO DETECTION SYSTEM
A Research-Based Python Project on Detecting Manipulated Facial Videos

Author: Sultan Mehmood
Program: BS Computer Science
Primary Language: Python
Project Type: Final Year Project / Research-Based Prototype


ABOUT THE PROJECT

This project began as my final year project with the goal of developing a deepfake video detection system using Python. The system processes videos, extracts frames, detects facial regions, trains machine learning models and predicts whether a given sample is real or fake.

I worked on this project with limited computing resources, so I focused on laptop-efficient experimentation. I used Python for the complete workflow and explored preprocessing, face detection, model training, evaluation and testing through a simple application interface.

During my experiments, the trained model achieved more than 90 percent accuracy on the available testing data. At first, this appeared to be a strong result. However, when I tested the system on newer and more realistic AI-generated deepfake videos, it failed to identify some of them correctly and classified fake videos as real.

This changed the direction of the project. Instead of presenting it as a complete deepfake detector, I now describe it as a research-based prototype that studies an important problem in deepfake detection: a model can perform well on familiar test data but still fail on new types of synthetic media.


PURPOSE OF THE PROJECT

The purpose of this project was to understand how deepfake detection systems are built and why they struggle with real-world generalization.

The project focused on:

1. Reading video and image inputs using Python.
2. Extracting useful frames from videos.
3. Detecting and cropping facial regions for further processing.
4. Preparing data for training and testing.
5. Experimenting with laptop-efficient machine learning approaches.
6. Training and evaluating a deepfake classification model.
7. Loading a saved model for prediction testing.
8. Studying failure cases involving recent AI-generated deepfake videos.


WHY THIS PROJECT MATTERS

Deepfake generation technology is improving very quickly. A detector trained on one dataset may learn patterns that work only for the examples it has already seen. When a newer generation method creates more realistic videos, those patterns may no longer be reliable.

This project taught me that good accuracy on a controlled dataset is not enough to prove that a deepfake detector will work in real situations. Detecting manipulated media requires testing on unseen data, different video qualities, compressed videos and newly generated deepfake samples.

The most valuable outcome of this project was not only building a working experimental pipeline, but also discovering its limitations through practical testing.


TECHNOLOGIES USED

Python was used as the primary programming language for the project.

The project includes work with:

- OpenCV for video reading, frame extraction and image processing.
- MTCNN-based experiments for face detection and facial-region cropping.
- Machine learning experiments using laptop-efficient approaches such as Logistic Regression and XGBoost.
- Model training, evaluation and prediction scripts written in Python.
- A saved trained model file for later testing.
- Basic HTML templates for a simple testing or demonstration interface.
- CSV and JSON files for storing training metrics.
- Research papers and benchmark notes to support the research study.


PROJECT WORKFLOW

The system follows a simple experimental workflow.

First, a video or image is provided as input. For a video, selected frames are extracted. The facial areas in those frames are then detected and cropped so that the model can focus on the face rather than the surrounding background.

After preprocessing, the prepared samples are used for training or testing. The trained model produces a prediction indicating whether the sample appears real or fake. The prediction results and evaluation metrics are then reviewed.

An important part of the project was testing the trained system on newer deepfake samples. These tests showed that a model can give strong results on the original evaluation data while making incorrect predictions on unfamiliar AI-generated videos.


PROJECT FOLDER STRUCTURE

The repository contains early experiments, later training work, testing files and supporting research papers.

Root-level Python files:

01_image_cropping.py
Used for image cropping and early preprocessing experiments.

02_vid_to_frames.py
Used for extracting frames from video files.

cropping_images_mtcnn.py
Used for experiments involving facial-region detection and cropping with MTCNN.

file_handling_and_image.py
Used for handling image files and related processing steps.

folder_splitting.py
Used for preparing or organizing data folders.

logistic_regression.py
Used for a lightweight machine learning baseline experiment.

model_training_dd2k25.py
Used for model training experiments.

modelTraining.py
Used for an alternative or earlier training workflow.

mtcnn_practice.py
Used while learning and testing face detection.

test_opencv.py
Used for testing OpenCV-based image and video processing.

videocapture.py
Used for experiments involving video reading or capture.

chk.py and practice1.py
Used during development and practice.


Training_new folder:

This folder contains the later training and evaluation work.

app.py
Application file associated with the training or demonstration flow.

deepfake_detector.py
Main detector-related implementation.

evaluate_model.py
Used for model evaluation.

fine_tuning.py
Used for further model-training experiments.

training_metrics.csv
Stores recorded training metrics.

training_metrics.json
Stores training metrics in JSON format.

templates/index.html
Interface template used with the application.


Testing_1 folder:

This folder contains the testing and prediction application.

App.py
Main testing application file.

datasplitting.py
Used for preparing or separating testing data.

evaluate.py
Used for evaluation during testing.

predict.py
Used for generating predictions.

test_model.py
Used for testing the trained model.

requirement.txt
Contains the required Python dependencies for the testing implementation.

models/best_deepfake_model.pth
Saved trained model artifact used for prediction testing.

templates/index.html
Application page for input or prediction testing.

templates/metrics.html
Page for displaying model metrics or results.

static/uploads/
Contains uploaded files or extracted frames used during application testing.


Research_papers_Secondary_data folder:

This folder contains the secondary research material that supported the research direction of the project. The collected papers cover deepfake detection, benchmark evaluation, generalization across datasets, face manipulation detection, fairness, real-world compression effects and recent challenges in detecting highly realistic AI-generated media.


HOW TO SET UP THE PROJECT

Python 3.9 or a later version is recommended.

Create a virtual environment:

python -m venv .venv

On Windows, activate it using:

.venv\Scripts\activate

On Linux or macOS, activate it using:

source .venv/bin/activate

Install the dependencies listed in the testing folder:

pip install -r Testing_1/requirement.txt

Depending on the branch of the project being executed, packages such as OpenCV, pandas, NumPy, scikit-learn, XGBoost, PyTorch, torchvision, Pillow, Flask and MTCNN may also be needed.


HOW TO RUN THE PROJECT

To run the testing application:

python Testing_1/App.py

This application is intended for loading the trained model, processing input media and generating predictions.

To run the training-related application:

python Training_new/app.py

The following scripts can also be used individually for prediction or evaluation work:

python Testing_1/predict.py
python Testing_1/evaluate.py
python Training_new/evaluate_model.py


MODEL DEVELOPMENT AND TESTING

The project was developed under normal student laptop constraints. Because of this, the work focused on practical and manageable experiments instead of expensive large-scale training.

I explored frame extraction, facial-region preprocessing, basic classification methods and model testing. Laptop-efficient models such as Logistic Regression and XGBoost were considered because they allowed repeated experiments without requiring high-end hardware.

A saved model artifact was produced and used during the testing phase. On the available experimental dataset, the trained system achieved more than 90 percent accuracy. This result showed that the model could learn patterns in the data it was trained and tested on.

However, the most important test came from using newer AI-generated deepfake samples. In those cases, the system sometimes predicted fake videos as real. This showed that the model had a generalization problem and could not be considered reliable for modern real-world deepfake detection.


MAIN FINDING

The main finding of this project is that accuracy on a known test dataset does not guarantee reliable deepfake detection in real-world situations.

The model performed well in the controlled testing setup, but failed on some newer AI-generated videos. This suggests that the model may have learned patterns related to the available dataset rather than features that remain dependable across new deepfake generation techniques.

This result is important because deepfake detection systems must be evaluated not only on familiar data, but also on unseen manipulation methods and realistic new samples.


LIMITATIONS

This project has several limitations.

The evaluation was mainly based on the available dataset and experimental setup. The system was not validated across a large number of independent datasets or recent deepfake generation tools.

The model focused mainly on processed facial frames, which may miss useful information across the complete video sequence.

The work was developed under laptop constraints, limiting the scale of training and experimentation.

Most importantly, the detector could not reliably identify all newer AI-generated fake videos. For this reason, it should not be treated as a real-world verification system or a forensic tool.


FUTURE WORK

This project can be extended by using more diverse datasets and evaluating the model on samples created by generation methods that were not present in the training data.

Future work can include cross-dataset testing, compressed-video testing, improved data augmentation, video-level prediction instead of only frame-level analysis and better recording of false-negative cases.

A stronger research version of the project should measure precision, recall, F1-score, confusion matrices and false-negative rates, especially for fake videos incorrectly classified as real.

The project could also be extended by studying temporal patterns across video frames, using ensemble techniques or comparing multiple efficient models under the same testing conditions.


RESPONSIBLE USE

This system is an academic and experimental prototype. It is not designed to provide final judgments about whether a video is authentic or manipulated.

A prediction marked as real does not prove that the content is genuine. A prediction marked as fake does not prove manipulation without further investigation.

The system should not be used alone for legal evidence, public accusations, identity verification, content-removal decisions, security screening or forensic conclusions.


PROJECT OUTCOME

Although the detector did not become a reliable solution for identifying all modern deepfake videos, the project was still a valuable technical and research experience.

Through this work, I gained experience in Python programming, image and video preprocessing, facial-region extraction, machine learning experimentation, model training, testing, evaluation and simple application development.

More importantly, I learned that real machine learning systems must be tested beyond their initial accuracy scores. In deepfake detection, the challenge is not only training a model that performs well on one dataset, but building a system that can remain reliable as synthetic media continues to improve.


RECOMMENDED PROJECT TITLE

An Experimental Study of Generalization Limitations in Laptop-Efficient Deepfake Video Detection Using Python


PORTFOLIO DESCRIPTION

This project is a Python-based experimental deepfake video detection prototype developed under student laptop constraints. It includes video-frame extraction, facial-region preprocessing, model training, evaluation and prediction testing through a simple application workflow. The model achieved strong results on the available experimental evaluation data, but further testing on newer AI-generated deepfake samples revealed important generalization failures. The project is therefore presented as a research study on the limitations of deepfake detection models rather than as a production-ready detection system.


RESUME DESCRIPTION

Developed a Python-based experimental deepfake video detection pipeline using video-frame extraction, facial-region preprocessing, laptop-efficient model training and inference testing; evaluated generalization limitations after newer unseen AI-generated fake samples were incorrectly classified as real despite strong controlled test performance.


FINAL NOTE

This repository represents both a working experimental implementation and an honest record of its limitations. The project demonstrates that failure cases are valuable in machine learning research because they show where a model must improve and why a high accuracy score alone cannot be treated as proof of real-world reliability.
