from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import logging


def train_and_save(model, X_train, y_train, model_path, learning_rate=0.0001, num_epochs=500, batch_size=64):
    """
    Trains the model based on the chosen parameters.
    
    """
    
    # Compile Model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load Balancing to mitigate poorly distributed data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    # Fit model
    history = model.fit(X_train, to_categorical(y_train), epochs=num_epochs, batch_size=batch_size, class_weight=class_weights_dict, validation_split=0.2)
    model.save(model_path) # Save
    
    return history


def evaluate(model, X_test, y_test):
    """
    Evaluates test data on 4 important metrics and the confusion matrix.
    Returns the confusion matrix to be used for plotting later.
    
    """
    
    # Predict on the test data
    y_test = to_categorical(y_test)
    y_pred_probs = model.predict(X_test)  # Predicted probabilities
    y_pred = y_pred_probs.argmax(axis=1)  # Convert probabilities to class predictions
    y_true = y_test.argmax(axis=1)        # Convert one-hot labels to class indices

    # Compute overall metrics
    accuracy = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Log all the metrics
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    # Compute per-class metrics
    translated_class_names = ['Bayat', 'Hejaz', 'Rast', 'Seekah', 'Saba', 'Ajam', 'Kurd', 'Nahawand']
    
    # For detailed metrics
    logging.info(classification_report(y_true, y_pred, target_names=translated_class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return cm