import os
import numpy as np
import pandas as pd
import re
import librosa
from utils.audio_features import AudioFeatures
import logging
import time
from tqdm import tqdm


# Configure logging for important information
logging.basicConfig(
    level=logging.INFO,                                
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.StreamHandler(),                      
        logging.FileHandler("outputs/logs/feature_extraction.log")           
    ]
)


def extract_sort_key(file_name):
    """
    Extract sorting key from file name to maintain consistent order.
    
    Args:
        file_name (str): Name of the file to extract parts from.
    
    Returns:
        tuple: A tuple representing the sort key (surah, part, clip).
    """
    match = re.match(r'surah(\d+)_part(\d+)_clip(\d+)\.mp3', file_name)
    if match:
        surah = int(match.group(1))
        part = int(match.group(2))
        clip = int(match.group(3))
        return (surah, part, clip)
    else:
        return (float('inf'), float('inf'), float('inf')) # handles possible errors in naming convention
    
    
def define_columns():
    """
    Define column names for the CSV output based on feature extraction.

    Returns:
        list: List of column names.
    """
    column_title = ['filename']
    mfcc_columns = [f'mfcc_mean_{i+1}' for i in range(20)] + [f'mfcc_std_{i+1}' for i in range(20)]
    chroma_columns = [f'chroma_{i+1}' for i in range(12)]
    other_columns = ['rms_mean', 'zcr_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean']
    return column_title + mfcc_columns + chroma_columns + other_columns


def extract_features(audio_dir, output_file):
    """
    Extract audio features and save them to a CSV file.

    Args:
        audio_dir (str): Directory containing audio files.
        output_file (str): Path to save the extracted features CSV.
    """
    # Retrieve all MP3 files and sort them based on the defined key
    file_names = [file_name for file_name in os.listdir(audio_dir) if file_name.endswith('.mp3')]
    sorted_file_names = sorted(file_names, key=extract_sort_key)
    features_list = []
    columns = define_columns() # Define the column names for the CSV
    start_time = time.time() # Start timing the feature extraction process

    # Process each file, extract features, and collect them
    for file_name in tqdm(sorted_file_names, desc="Extracting features from all files"):
        file_path = os.path.join(audio_dir, file_name)
        y, sr = librosa.load(file_path)  # Load the audio file
        ext_features = AudioFeatures(y, sr)
        features = ext_features.compute_all_features()

        # Append file name to feature list to keep track of files
        features_with_filename = [file_name] + features.tolist()  # Prepend the file name to the features
        features_list.append(features_with_filename)

    # Create a DataFrame from the features list and specify column names
    features_df = pd.DataFrame(features_list, columns=columns)

    # Save the features to a CSV file
    features_df.to_csv(output_file, index=False)
    logging.info(f"Feature extraction complete and saved to {output_file}")
    
    # Log the total duration of the feature extraction
    total_duration = time.time() - start_time
    logging.info(f"Total feature extraction time: {total_duration:.2f} seconds")


if __name__=='__main__':
    audio_dir = 'data/clips'
    output_file = 'outputs/extracted_features.csv'
    extract_features(audio_dir, output_file)