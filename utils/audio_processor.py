import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from moviepy.editor import AudioFileClip
import shutil

class AudioProcessor:
    """
    Class to process audio files for feature extraction, segmentation, normalization,
    classification, and overall audio processing into labels.

    Attributes:
        file_path (str): Path to the audio file.
        scaler (MinMaxScaler): Scaler for normalizing features.
        features_method (callable): Class to compute audio features.
    """
    
    def __init__(self, file_path, features_method, scaler):
        self.file_path = file_path
        self.audio = None
        self.sr = None # Sample rate
        self.scaler = scaler
        self.segments = []
        self.features = []
        self.predictions = []
        self.summary = None
        self.compute_audio = features_method
        self.output_folder = 'temporary'

    
    def segment_audio(self):
        # Segment an audio file into smaller 30 second clips and save them temporarily
        
        # Create temporary folder to hold the clips
        if not os.path.exists(self.output_folder): 
            os.makedirs(self.output_folder)
            
        # Load the audio file using moviepy
        audio_clip = AudioFileClip(self.file_path)
        duration = audio_clip.duration  # Total duration in seconds

        # Duration of a 30-second segment in seconds
        segment_length = 30  
        start_time = 0
        clip_number = 1

        while start_time < duration: # Loop to extract clips
            end_time = min(start_time + segment_length, duration)

            # Extract the segment
            segment = audio_clip.subclip(start_time, end_time)

            # Save the segment to a file
            new_filename = f"clip{clip_number}.mp3"
            segment.write_audiofile(os.path.join(self.output_folder, new_filename))

            # Prepare for the next segment
            start_time = end_time
            clip_number += 1
        
    def extract_features(self):
        # Extracts features from each audio segment and compiles them into a feature list
        
        # Compile all clips
        file_paths = [os.path.join(self.output_folder, f) for f in os.listdir(self.output_folder) if f.endswith('.mp3')]
        
        # Compute chroma, ZCR, RMS, and spectral features for each clip
        for segment in file_paths:
            segment_loaded, self.sr = librosa.load(segment, sr=None)
            self.computed = self.compute_audio(segment_loaded, self.sr)
            chroma = self.computed.compute_chroma_mean()
            zcr = self.computed.compute_zcr_mean()
            rms = self.computed.compute_rms_energy()
            centroid = self.computed.compute_spectral_centroid_mean()
            bandwidth = self.computed.compute_spectral_bandwidth_mean()
            rolloff = self.computed.compute_spectral_rolloff_mean()
            self.features.append(np.hstack([chroma, zcr, rms, centroid, bandwidth, rolloff]))
        

    def normalize_features(self):
        # Normalize the features with MinMaxScaler used on training data
        self.features = self.scaler.fit_transform(self.features)
        

    def classify_segments(self, model):
        # Load model and classify segments
        model = load_model(model)

        english_scale_mapping = {
            0: 'Bayat',
            1: 'Hejaz',
            2: 'Rast',
            3: 'Seekah',
            4: 'Saba',
            5: 'Ajam',
            6: 'Kurd',
            7: 'Nahawand'
        }

        self.predictions = [
            np.argmax(model.predict(np.expand_dims(feature, axis=0)))
            for feature in self.features
            ]
        self.predictions_mapped = [english_scale_mapping[pred] for pred in self.predictions]

    
    def voting_system(self):
        # Applies a voting system to the classified segments to determine the most frequent prediction 
        np_list_predictions = np.array(self.predictions_mapped)
        unique, counts = np.unique(np_list_predictions, return_counts=True) # Tally predictions
        self.most_frequent = unique[np.argmax(counts)] # Choose most frequent
        return self.most_frequent

    
    def clear_temp(self):
        # Clear the temporary folder once done with classification
        shutil.rmtree(self.output_folder)

    
    def process(self, model):
        # High-level method to run all steps in sequence
        self.segment_audio()
        self.extract_features()
        self.normalize_features()
        self.classify_segments(model=model)
        self.voting_system()
        self.clear_temp()
        return self.most_frequent
