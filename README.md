## Arabic Melodies Project: Maqamat Classification using AI

In traditional Arabic music and recitation, maqamat are the foundation of melody, guiding the emotional and structural flow of a performance. These eight distinct melodic modes define how notes are arranged and transitioned, creating different moods such as joy, sorrow, intimacy, or awe. While expert reciters seamlessly switch between maqamat, recognizing them by ear is an advanced skill that takes years of practice.

This project leverages machine learning and audio processing to classify recitations based on their maqam. By analyzing recorded recitations, the system can identify the dominant maqam, providing insights into the melody and structure. The goal is to make maqam recognition more accessible to learners, researchers, and enthusiasts who want to explore this rich musical tradition through AI.

This repository outlines the end-to-end development pipeline, from dataset creation to deep learning model training and future deployment plans.

---------------------------------------

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

---------------------------------------

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

An API service is being built using FastAPI to process longform audio.

This is a work in progress and will be elaborated on further in future updates.

# Technical Stack

🔧 Tools & Libraries Used:

Python 3.11

Machine Learning: TensorFlow, scikit-learn

Audio Processing: Librosa

OCR & Text Processing: OpenCV, Tesseract OCR, SpellChecker

Web Deployment: FastAPI, AWS EC2

# Stay Tuned!

This project is still evolving! Code will be provided, as well as more details!

Future updates will include dataset expansion, model improvements, and a web-based classification tool.

🔹 Follow this repository for updates!

---------------------------------------

**Contributors**
- Mohssen Kassir
