from utils.audio_processor import AudioProcessor
from utils.audio_features import AudioFeatures
import joblib
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,                                
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.StreamHandler(),                      
        logging.FileHandler("outputs/logs/example_prediction.log")           
    ]
)

def run_example(file_path, AudioFeatures, NNscaler, model):
    """
    Process an audio example using a trained model and log the prediction.

    Args:
        file_path (str): Path to the audio file to process.
        AudioFeatures (class): AudioFeatures class used for feature extraction.
        NNscaler (object): Trained MinMaxScaler object for feature scaling.
        model (str): Path to the trained model file.
    """
    # Initialize the audio processor with provided audio features and scaler
    processor = AudioProcessor(file_path, AudioFeatures, NNscaler)
    
    # Process the audio file, predict its class, and log the result
    prediction = processor.process(model)
    logging.info('prediction for example: %s', prediction)


if __name__ == '__main__':
    file_path = 'data/example.mp3'
    NNscaler = joblib.load('outputs/normalizing_scalers/scaler.pkl')
    model = 'outputs/models/model.h5'
    run_example(file_path, AudioFeatures, NNscaler, model)
    
