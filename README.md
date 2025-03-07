# Arabic Melodic Scales (Maqams) Classification Project 

## Overview

This project aims to classify Arabic melodic scales (maqams) from audio clips using machine learning. The project processes audio files to extract features, trains a neural network to recognize different scales, and provides tools for visualizing the distribution of maqamat across a dataset. The repository is 1.6 GB to download (due to containing an original dataset).

## Features

This pipeline is 4 parts:

1. Extracting features from the data (run_feature_extraction.py)
2. Running a data distribution analysis (run_data_analysis.py)
3. Training and evaluating the model (run_model.py)
4. Testing classification on an example segment (run_example.py)

These 4 parts can function completely independently since all the necessary models and data are already uploaded.

## Introduction

The introduction explains the background knowledge, my unique contributions, how I developed a dataset, and the results of this project. It covers a larger scope than this readme, which focuses on the project structure and how to run it.


## Project Structure

```
.
├── data/
│   ├── clips/                   # Audio clips for feature extraction
│   ├── labels/                  # JSON files with labels for each clip
│   ├── example.mp3              # An example that is run in the final script
│   └── extracted_features.csv   # All the features from all the clips
├── outputs/
│   ├── logs/                    # Log files
│   ├── model/                   # Trained model file
│   ├── normalizing_scaler/      # Scaler object for data normalization
│   └── plots/                   # Plots and graphics generated from the data
├── src/
│   ├── data_preprocessor.py     # Script for data loading and preprocessing
│   ├── model.py                 # Script to build and compile the model
│   ├── train_eval_model.py      # Script for training and evaluating the model
│   └── visualize_results.py     # Script for visualizing results
├── utils/
│   ├── audio_features.py        # Script for audio feature extraction
│   └── audio_processor.py       # Script for Processing audio and contains segment classifier
├── images_for_introduction/     # Folder containing images used in INTRODUCTION.md
├── run_model.py                 # Executes the full model training and evaluation pipeline
├── run_data_analysis.py         # Performs data analysis on maqam distribution
├── run_example.py               # Processes a single example through the model
├── run_feature_extraction.py    # Extracts features from audio files
├── README.md                    # Main project documentation (structure & execution details)
├── INTRODUCTION.md              # In-depth explanation of the project, background, and methodology
└── requirements.txt             # Dependencies needed to run the project
```

## Structure of the Data

**Clips**

The clips are all composed from chopped up surahs (called chapters in English). Each surah is divided into X parts (with each unique part is recited in its own melodic scale). Each part is chopped into a Y clips (30-60 seconds long).

***Example Clip Name:*** 
```
surah2_part3_clip2.mp3
```

This is the second chapter in the reciter's playlist, and the third part. All clips belonging to the third part have the same label.

**Labels**

The corresponding label will have the same naming pattern.

***Example Label Name:***
```
surah2_part3.mp3.json
```

This label is paired with all clips from the second chapter and third part. 

Every label contains:
 - The timestamp from the youtube video where the part is played
 - The scale (aka maqam) used; written in Arabic
 - The branch used; written in Arabic (this data is extracted from playlist but not used in this project)
 - The pitch degree used (this data is extracted from playlist but not used in this project)
 - The sign of the pitch used (this data is extracted from playlist but not used in this project)
 

## How to Run the Pipeline

**Prerequisites**

Ensure you have Python 3.12 and pip installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

**Running the pipeline**

To run the feature extraction:
```bash
python run_feature_extraction.py
```

To perform data distribution analysis:
```bash
python run_data_analysis.py
```

To train and evaluate the model:
```bash
python run_model.py
```

To train the model:
```bash
python run_example.py
```

 - The correct classification to for this example is Hejaz

## Inputs and Outputs for Each Script 


| Script                      | Inputs                                      | Outputs                                              |
|-----------------------------|-----------------------------------------------|------------------------------------------------------|
| `run_feature_extraction.py` | `data/clips`                                | `data/extracted_features.csv`                        |
| `run_data_analysis.py`      | `data/labels`                               | `outputs/plots/scale_distribution.png` <br> `outputs/logs/feature_extraction.log` |
| `run_model.py`              | `data/extracted_features.csv` <br> `data/labels` | `outputs/models/model.h5` <br> `outputs/normalizing_scalers/scaler.pkl` <br> `outputs/plots/training_plots.png` <br> `outputs/plots/confusion_matrices.png` <br> `outputs/logs/training.log` |
| `run_example.py`            | `data/example.mp3` <br> `outputs/normalizing_scalers/scaler.pkl` <br> `outputs/models/model.h5` | `outputs/logs/example_prediction.log` |

## Future Improvements 

Future technical improvements include:
 - Consolidating the JSON label files into a single JSON file
 - Translation of the dataset labels into English
 - Further testing the segment classifier to ensure accurate maqam detection
 - Using Git LFS to manage the large dataset and make it more download-friendly
 
 This is still a work-in-progress, so stay tuned for many more updates, both minor and major!

## Acknowledgments

Thank you to the reciter Mustafa Shukeir for making his playlist public, which made this project possible!


🔹 Follow this repository for updates!

---------------------------------------

**Contributors**
- Mohssen Kassir
