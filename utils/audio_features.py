import librosa
import numpy as np

class AudioFeatures:
    """
    A class for extracting various audio features using librosa.

    Attributes:
        y (np.array): Audio time series.
        sr (int): Sampling rate of y.
    """
    
    def __init__(self, y, sr):
        self.y, self.sr = y, sr    

        
    def compute_mfcc_mean(self, num_mfcc=20): # Compute 20 MFCC Means
        self.mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=num_mfcc)
        self.mfccs_mean = np.mean(self.mfccs.T, axis=0)
   
        return self.mfccs_mean
    
    def compute_mfcc_std(self, num_mfcc=20): # Compute 20 MFCC STDs
        self.mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=num_mfcc)
        self.mfccs_std = np.std(self.mfccs.T, axis=0)
        
        return self.mfccs_std
     
    def compute_chroma_mean(self): # Compute 12 Chroma Means
        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        self.chroma_mean = np.mean(self.chroma.T, axis=0)

        return self.chroma_mean
    
    def compute_zcr_mean(self): # Compute Zero Crossing Rate Mean
        self.zcr = librosa.feature.zero_crossing_rate(self.y)
        self.zcr_mean = np.mean(self.zcr.T, axis=0)

        return self.zcr_mean
        
    def compute_rms_energy(self): # Compute Root Mean Square Energy Mean
        self.rms = librosa.feature.rms(y=self.y)
        self.rms_mean = np.mean(self.rms.T, axis=0)

        return self.rms_mean
    
    def compute_spectral_centroid_mean(self): # Compute Centroid Mean
        self.spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        self.spectral_centroid_mean = np.mean(self.spectral_centroid.T, axis=0)

        return self.spectral_centroid_mean
    
    def compute_spectral_bandwidth_mean(self): # Compute Bandwidth Mean
        self.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        self.spectral_bandwidth_mean = np.mean(self.spectral_bandwidth.T, axis=0)
        
        return self.spectral_bandwidth_mean
    
    
    def compute_spectral_rolloff_mean(self): # Compute Rolloff Mean
        self.spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        self.spectral_rolloff_mean = np.mean(self.spectral_rolloff.T, axis=0)

        return self.spectral_rolloff_mean
    
    def compute_all_features(self): # Combine all 57 of the above features
        # Compute all features and concatenate them into a single feature vector
        features = np.hstack([
            self.compute_mfcc_mean(),
            self.compute_mfcc_std(),
            self.compute_chroma_mean(),
            self.compute_rms_energy(),
            self.compute_zcr_mean(),
            self.compute_spectral_centroid_mean(),
            self.compute_spectral_bandwidth_mean(),
            self.compute_spectral_rolloff_mean()
        ])
    
        return features