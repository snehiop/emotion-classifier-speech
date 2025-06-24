# Emotion Classification from Speech - Deep Learning Project

This repository contains an end-to-end pipeline for classifying emotions from speech audio using deep learning. It combines robust audio feature extraction, data augmentation, and a custom CNN-BiLSTM-Attention model to accurately identify emotional states conveyed in speech.

# Project Objective

To build a high-performance emotion classification system using audio data. The pipeline leverages audio preprocessing, augmentation, deep learning models, and a hosted Streamlit web app for real-time emotion detection.

# Dataset

We used the RAVDESS dataset, split across two folders:

Audio_Speech_Actors_01-24
Audio_Song_Actors_01-24

# Emotion Mapping:

01 - neutral
02 - calm
03 - happy
04 - sad
05 - angry
06 - fearful
07 - disgust
08 - surprised

# Data Preprocessing

Log-Mel Spectrograms were extracted from audio (128 x 130 shape).
Applied Data Augmentation:

1.Noise injection
2.Pitch shifting
3.Time stretching

Combined original + augmented data to increase diversity.
Features were padded or truncated to ensure consistent dimensions.

# Model Architecture

Custom deep learning pipeline:
- 3× Conv1D blocks with Layer Normalization, LeakyReLU, MaxPooling, Dropout
- BiLSTM layer to capture temporal dependencies
- SeqSelfAttention layer to focus on relevant parts
- GlobalAveragePooling + Dense layers
- Loss: Focal Loss with gamma=2.0
- Optimizer: Adam (LR = 3e-4)

# Training Strategy:

- Stratified split (80/20)
- Oversampling for underrepresented classes (e.g., sad, angry)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

 # Evaluation Metrics

- Achieved strong performance on validation set:
- F1 Score: > 82%
- Overall Accuracy: > 83%
- All Class-wise Accuracies: > 80%

# Plotted Confusion Matrix and Classification Report to validate.

Confusion Matrix Observations:

- Happy and Sad classes had minor misclassifications with Neutral and Fearful respectively.
- Neutral samples were sometimes misclassified as Calm or Sad.
- Disgust had slightly lower accuracy compared to others, likely due to limited examples in dataset.
- Despite these, the classifier meets all required thresholds for F1 and per-class accuracy.

# Streamlit Web App

A lightweight web app was built using Streamlit:

- Upload a .wav file
- Extracts features and predicts emotion
- Shows result, confidence, and class-wise probabilities
- Hosted using Streamlit Cloud

# File Structure
.
├── app.py                  # Streamlit app and prediction
├── final_emotion_model.h5 # Trained DL model
├── requirements.txt       # Dependencies
├── README.md              # This file

# Demo Video

A short 2-minute demo video showing the web app usage is available in the repository or shared via Google Drive.

# Installation & Usage

# Clone the repo
https://github.com/snehiop/emotion-classifier
cd emotion-classifier

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# License

This project is under the MIT License. Attribution required when reused.

"Understanding human emotion is the first step toward building empathetic AI."
