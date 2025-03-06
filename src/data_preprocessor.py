import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib
import os
import json


def get_scale(file_name, labels_dir):
    # Extract base name without '_clip' suffix and load corresponding JSON for maqam
    base_name = file_name.split('_clip')[0]
    json_filename = base_name + '.mp3.json'
    json_path = os.path.join(labels_dir, json_filename)
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data['maqam']
    

def load_data(csv_path):
    # Load dataset and extract feature columns based on predefined patterns
    df = pd.read_csv(csv_path)
    file_names = df.iloc[:, 0].values  # Save all the filenames seperately
    
    # 17 (out of 57) features are chosen and extracted
    feature_patterns = ['chroma', 'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']
    feature_columns = [col for col in df.columns if any(pattern in col for pattern in feature_patterns)]
    training_vals = df[feature_columns] # Save all features of interest in df
    return training_vals, file_names


def preprocess_features(features, scaler_path):
    # Scale features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # The scaler needs saved to scale future audio being inputed into the trained network
    joblib.dump(scaler, scaler_path)
    return scaled_features


def encode_labels(labels):
    # Encode labels using LabelEncoder and convert to categorical format
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    return encoded_labels, categorical_labels
  

def split_data(X, y_categorical, test_size=0.2, random_state=42):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
