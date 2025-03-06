from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Softmax

def build_model(input_shape):
    """
    Builds and returns a Sequential neural network model.

    This model is constructed with several densely connected layers,
    dropout for regularization, batch normalization for training stability,
    and LeakyReLU for non-linear activation.

    Args:
        input_shape (int): The shape (dimensionality) of the input data.

    Returns:
        keras.engine.sequential.Sequential: A compiled neural network model.
    """
    model = Sequential([
        # Input layer and first hidden layer with dropout and batch normalization
        Dense(1024, input_shape=(input_shape,)),
        Dropout(0.4),
        BatchNormalization(),
        LeakyReLU(),

        # Second hidden layer
        Dense(512),
        Dropout(0.4),
        BatchNormalization(),
        LeakyReLU(),

        # Third hidden layer
        Dense(256),
        Dropout(0.25),
        BatchNormalization(),
        LeakyReLU(),

        # Fourth hidden layer
        Dense(128),
        Dropout(0.25),
        BatchNormalization(),
        LeakyReLU(),

        # Fifth hidden layer
        Dense(64),
        BatchNormalization(),
        LeakyReLU(),

        # Sixth hidden layer
        Dense(32),
        BatchNormalization(),
        LeakyReLU(),

        # Output layer with softmax activation for multi-class classification
        Dense(8),  # 8 total classes
        Softmax()
    ])
    
    return model
