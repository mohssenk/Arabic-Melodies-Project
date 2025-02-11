## Arabic Melodies Project: Maqamat Classification using AI
Developing an AI-powered system for classifying Quranic recitations based on Arabic musical scales (maqamat).

## Overview
The Arabic Melodies Project aims to classify Quranic recitations according to their maqam (melodic mode) using deep learning and audio processing techniques. 

This project addresses the high barrier to learning maqamat by providing an AI-based classification system to assist learners, reciters, and researchers.

## What are Maqamat?
Maqamat are eight classical Arabic musical scales used in Middle Eastern music and Quranic recitation. Each scale conveys a distinct emotion:

Rast – Confidence & Strength

Bayat – Spiritual Depth

Hejaz – Yearning & Awe

Nahawand – Tenderness (Minor Scale)

Ajam – Happiness & Joy (Major Scale)

Saba – Sorrow

Kurd – Simplicity

Sikah – Intimacy & Devotion

Understanding maqamat is an advanced skill typically requiring years of training. This AI-powered system aims to classify maqamat from recitations and assist learners in recognizing them.

## Project Pipeline
The project follows a three-phase workflow:

## Phase 1: Data Collection & Preprocessing
📌 Objective: Build a labeled dataset of Quranic recitations with maqam annotations.

Data Source: A 30.5-hour YouTube playlist of Quranic recitations from a single reciter.

Audio Clips: Each 30-second segment is assigned a maqam label.

Annotations: Stored in a JSON format with timestamps and maqam labels.

## Phase 2: Model Training
📌 Objective: Train a deep learning model to classify maqamat from Quranic recitations.

### Feature Extraction

For Feature Extraction, instead of just using MFCCs like previous research, additional features were extracted for better classification:

Chroma Values (12 features)

Root Mean Square Energy

Zero-Crossing Rate

Spectral Centroid

Spectral Bandwidth

Spectral Rolloff

### Neural Network Architecture:

A feedforward artificial neural network (ANN) was trained. The configuration will be detailed in a future update. 

### Results
Overall Accuracy: 84.5%

Per-Class Accuracy:

Hejaz: 96%

Saba: 90%

Nahawand: 89%

Kurd: 86%

Ajam: 84%

Sikah: 87%

Rast: 76%

Bayat: 70%

## Phase 3: Deployment & API
📌 Objective: Provide a usable system for maqam classification.

### Segment Classification:

The model classifies 30-second clips individually.

A majority voting system determines the maqam of an entire segment.

### API Service:
Built using FastAPI to process longform audio.
Limitations: Assumes the entire audio segment is in one maqam (future improvements planned).
Planned Web Application:
User Uploads Recitation → Model Predicts Maqam → Result Displayed
Hosted on AWS EC2, potential future integration with S3 storage.
Future Steps

# Technical Stack
🔧 Tools & Libraries Used:

Python 3.11

Machine Learning: TensorFlow, scikit-learn

Audio Processing: Librosa

OCR & Text Processing: OpenCV, Tesseract OCR, SpellChecker

Web Deployment: FastAPI, AWS EC2

# Stay Tuned!

This project is still evolving! Future updates will include dataset expansion, model improvements, and a web-based classification tool.

🔹 Follow this repository for updates!

---------------------------------------

**Contributors**
- Mohssen Kassir
