import os
from src.data_preprocessor import load_data, preprocess_features, encode_labels, split_data, get_scale
from src.model import build_model
from src.train_eval_model import train_and_save, evaluate
from src.visualize_results import plot_confusion_matrix, save_training_plots
import numpy as np
import logging


# Configure logging for important information
logging.basicConfig(
    level=logging.INFO,                                
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.StreamHandler(),                      
        logging.FileHandler("outputs/logs/training.log")           
    ]
)


def run_model(csv_path, labels_dir, model_path, scaler_path, output_dir):
    """
    Executes the full model training and evaluation pipeline.
    
    Args:
        csv_path (str): Path to the CSV file containing extracted features.
        labels_dir (str): Directory containing label files in JSON format.
        model_path (str): Path where the trained model will be saved.
        scaler_path (str): Path where the scaler object will be saved.
        output_dir (str): Root directory to save outputs (plots, logs).
    
    Loads and preprocesses data, builds a model, trains, evaluates, and
    visualizes results. Outputs are saved in specified directories.
    """
    
    
    # Load and preprocess data
    features, file_names = load_data(csv_path)
    scaled_features = preprocess_features(features, scaler_path)
    labels = np.array([get_scale(file_name, labels_dir) for file_name in file_names])
    encoded_labels, categorical_labels = encode_labels(labels)
    
    # Prepare data
    X_train, X_test, y_train, y_test = split_data(scaled_features, encoded_labels)
    
    # Build and train model
    model = build_model(input_shape=X_train.shape[1])
    history = train_and_save(model, X_train, y_train, model_path)
    
    # Evaluate the model
    cm = evaluate(model, X_test, y_test)
    
    # Visualization of results
    plot_confusion_matrix(cm, output_dir=os.path.join(output_dir, 'plots'))
    save_training_plots(history, output_dir=os.path.join(output_dir, 'plots'))
    
    logging.info("Model training and evaluation complete. Results are saved in the output directory.")

    
if __name__ == '__main__':
    # Configuration of locations
    csv_path = 'data/extracted_features.csv'
    labels_dir = 'data/labels'
    model_path = 'outputs/models/model.h5'
    scaler_path = 'outputs/normalizing_scalers/scaler.pkl'
    output_dir = 'outputs'
    
    run_model(csv_path, labels_dir, model_path, scaler_path, output_dir)
    