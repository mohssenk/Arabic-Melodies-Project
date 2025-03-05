import librosa
import numpy as np

class compute_audio:
    def __init__(self, y, sr):
        #self.name = file_name
        self.y, self.sr = y, sr
        #self.y, self.sr = librosa.load(self.name, duration=60)
    
    
    
    def compute_mfcc_mean(self, num_mfcc):
        self.mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=num_mfcc)
        self.mfccs_mean = np.mean(self.mfccs.T, axis=0)
    
        return self.mfccs_mean
    
    def compute_chroma_mean(self):
        self.chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        self.chroma_mean = np.mean(self.chroma.T, axis=0)

        return self.chroma_mean
    
    def compute_zcr_mean(self):
        self.zcr = librosa.feature.zero_crossing_rate(self.y)
        self.zcr_mean = np.mean(self.zcr.T, axis=0)

        return self.zcr_mean
        
    def compute_rms_energy(self):
        self.rms = librosa.feature.rms(y=self.y)
        self.rms_mean = np.mean(self.rms.T, axis=0)

        return self.rms_mean
    
    def compute_spectral_centroid_mean(self):
        self.spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        self.spectral_centroid_mean = np.mean(self.spectral_centroid.T, axis=0)

        return self.spectral_centroid_mean
    
    def compute_spectral_bandwidth_mean(self):
        self.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        self.spectral_bandwidth_mean = np.mean(self.spectral_bandwidth.T, axis=0)
        
        return self.spectral_bandwidth_mean
    
    
    def compute_spectral_rolloff_mean(self):
        self.spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        self.spectral_rolloff_mean = np.mean(self.spectral_rolloff.T, axis=0)

        return self.spectral_rolloff_mean