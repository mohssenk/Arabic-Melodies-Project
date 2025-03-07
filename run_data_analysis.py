import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def time_to_seconds(timestamp):
    """
    Helper function to convert time string to seconds
    
    """
    minutes, seconds = map(float, timestamp.replace('m', '').replace('s', '').split())
    return minutes * 60 + seconds


def calculate_cumulative_lengths(labels_dir):
    """
    Calculate the cumulative length in seconds for each maqam from JSON label files.

    """
    maqam_length = defaultdict(float)  # Default dictionary to store cumulative lengths
    
    for filename in os.listdir(labels_dir):    
        if filename.endswith('.json'):
            with open(os.path.join(labels_dir, filename), 'r', encoding='utf-8') as f:
                label = json.load(f)  # Load the JSON data from file
                timestamp = label['timestamp']  # Extract the length of the maqam
                maqam = label['maqam'][0] if isinstance(label['maqam'], list) else label['maqam'] # Handle maqam being list or single value
                length = time_to_seconds(timestamp)  # Convert timestamp to seconds
                maqam_length[maqam] += length   # Accumulate the length for this maqam

    return maqam_length


def plot_cumulative_data(data, title, xlabel, ylabel, output_dir='outputs/plots', filename='scale_distribution.png'):
    """
    Create and save a bar plot showing the distribution of data.

    """
    # Set up the plot
    plt.figure(figsize=(10, 5))
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    
def translate_items(scales_with_len):
    """
    Translate maqam names from Arabic to English using a provided dictionary.

    """
    translate = {
        'بيات': 'Bayat',
        'حجاز': 'Hejaz',
        'راست': 'Rast',
        'سيكاه': 'Seekah',
        'صبا': 'Saba',
        'عجم': 'Ajam',
        'كرد': 'Kurd',
        'نهاوند': 'Nahawand'
    }
    translated_scales = defaultdict(float, {
                        translate[key.strip("[]'")]: value for key, value in scales_with_len.items()
                        })
    return translated_scales


if __name__ == '__main__':
    # Find the lengths of the scales (in arabic)
    maqam_len = calculate_cumulative_lengths(labels_dir='data/labels')
    # Translate the scales to english
    translated_len = translate_items(maqam_len)
    # Plot the data distribution
    plot_cumulative_data(translated_len, 'Length Per Scale', 'Scale', 'Length (seconds)', output_dir='outputs/plots')