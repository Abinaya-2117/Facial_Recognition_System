## Face Recognition System (LFW Dataset)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-green.svg)
![Status](https://img.shields.io/badge/Project-Complete-success.svg)

------------------------------------------------------------------------

## Project Overview

This project implements a complete **Face Recognition System** using the
**LFW (Labeled Faces in the Wild -- Deepfunneled)** dataset.

### The system performs:

- Face Detection using **MTCNN**
- Feature Extraction using **EfficientNet Lite0 (TensorFlow Hub)**
- Classification using **Support Vector Machine (SVM)**
- Model Saving & Reloading
- Performance Evaluation
- Visual Demo with Confidence Scores

This project demonstrates an end-to-end deep learning + machine learning
pipeline for facial recognition.

------------------------------------------------------------------------

## Dataset

**Dataset Used:**\
LFW -- Labeled Faces in the Wild (Deepfunneled Version)

### Dataset Structure

    lfw-deepfunneled/
    └── lfw-deepfunneled/
        ├── Person_1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        ├── Person_2/
        │   ├── image1.jpg
        │   ├── image2.jpg

-   Each folder represents one individual.
-   Images are labeled using folder names.
-   Minimum samples per person controlled via `max_samples_per_person`.

------------------------------------------------------------------------

## System Architecture

### Face Detection

-   MTCNN detects bounding boxes.
-   First detected face is cropped.
-   Resized to **160 × 160**.

### Preprocessing

-   Convert BGR → RGB
-   Normalize to \[-1, 1\]

### Feature Extraction

-   **EfficientNet Lite0 (TF Hub)**
-   Generates embedding vector for each face.

### Classification

-   Label Encoding

-   Train/Test Split (80/20)

-   SVM Classifier:

    ``` python
    SVC(kernel="linear", C=0.1, probability=True)
    ```

------------------------------------------------------------------------

## Training Pipeline

1.  Mount Google Drive
2.  Unzip dataset
3.  Process dataset (detect + embed)
4.  Train SVM classifier
5.  Evaluate performance
6.  Save trained model (`.pkl`)

Saved model contains:

``` python
{
  "Classifier": trained_svm,
  "Label_Encoder": label_encoder
}
```

------------------------------------------------------------------------

## Evaluation Metrics

During Training:

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Training Distribution Plot

During Demo:

- Per-image prediction
- Confidence percentage
- Correct vs ❌ Incorrect predictions
- Overall demo accuracy

------------------------------------------------------------------------

## How to Run

### Install Dependencies

``` bash
pip install tensorflow mtcnn scikit-learn tensorflow_hub opencv-python matplotlib tqdm
```

###  Prepare Dataset

Place dataset in:

    /content/drive/MyDrive/lfw-deepfunneled/lfw-deepfunneled

###  Run Training Script

``` bash
python facerecognitionsystem.py
```

The script will:

-   Train the model
-   Display evaluation results
-   Save trained model
-   Run demo predictions

------------------------------------------------------------------------

##  Demo Output Example

    ========== METRICS ==========
    Accuracy: 80.00%
    Correct: 4/5
    =============================

    ✓ True: Person_A         Predicted: Person_A         Confidence: 92.34%
    ✗ True: Person_B         Predicted: Person_A         Confidence: 61.12%

------------------------------------------------------------------------

## Challenges

-   Limited samples per identity
-   Face detection failures
-   Similar-looking individuals
-   EfficientNet not face-specialized embedding model

------------------------------------------------------------------------

## Future Improvements

-   Replace EfficientNet with FaceNet / ArcFace
-   Add unknown face threshold detection
-   Increase dataset diversity
-   Deploy real-time webcam recognition
-   Optimize multi-face handling

------------------------------------------------------------------------

## Technologies Used

-   Python
-   TensorFlow
-   TensorFlow Hub
-   MTCNN
-   OpenCV
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Google Colab

------------------------------------------------------------------------

## Conclusion

This project combines deep learning feature extraction with classical
machine learning classification to build a complete facial recognition
system.

It demonstrates practical implementation of: - CNN-based embeddings -
SVM classification - Model persistence - Visual prediction system

------------------------------------------------------------------------
