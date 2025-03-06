import matplotlib.pyplot as plt  
import seaborn as sns  
import os  
import logging 


def plot_confusion_matrix(cm, output_dir='outputs/plots', filename='confusion_matrices.png'):
    """
    Plot and save a confusion matrix and its normalized versions.
    
    This function visualizes the confusion matrix and its row and column normalized versions.
    It also logs the per-class recall.

    Args:
        cm (numpy.ndarray): The confusion matrix to plot.
        output_dir (str): Directory to save the generated plot.
        filename (str): Filename for the saved plot.
        
    """
    
    # Scale names in English for translation
    transl_class_names = ['Bayat', 'Hejaz', 'Rast', 'Seekah', 'Saba', 'Ajam', 'Kurd', 'Nahawand']
    
    # Find the recall for each class
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    
    # Normalize confusion matrix by row
    cm_normalized_row = 100 * cm / cm.sum(axis=1, keepdims=True)
    # Normalize confusion matrix by column
    cm_normalized_col = 100 * cm / cm.sum(axis=0, keepdims=True)
    
    # Log the class recalls
    for idx, class_name in enumerate(transl_class_names):
        logging.info(f"Recall for {class_name}: {per_class_recall[idx]:.4f}")

    # Visualize the confusion matrix
    plt.figure(figsize=(24, 8))  # Larger figure to hold all subplots

    # Subplot 1: Regular confusion matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=transl_class_names, yticklabels=transl_class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Subplot 2: Confusion Matrix - Normalized by Row
    plt.subplot(1, 3, 2)
    sns.heatmap(cm_normalized_row, annot=True, fmt='.1f', cmap='Blues', xticklabels=transl_class_names, yticklabels=transl_class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Normalized by Row')

    # Subplot 3: Confusion Matrix - Normalized by Column
    plt.subplot(1, 3, 3)
    sns.heatmap(cm_normalized_col, annot=True, fmt='.1f', cmap='Blues', xticklabels=transl_class_names, yticklabels = transl_class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Normalized by Column')

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    
def save_training_plots(history, output_dir='outputs/plots', filename='training_plots.png'):
    """
    Save plots of training and validation accuracy and loss.
    Args:
        history (History): Training history object from Keras containing performance metrics.
        output_dir (str): Directory to save the generated plots.
        filename (str): Filename for the saved plot.
    """

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    