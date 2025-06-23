import os
import librosa
import numpy as np
import pandas as pd

DATASET_DIR = "dataset"
CSV_FILE = "speech_dataset.csv"

def load_data():
    """ Load dataset and extract features. """
    df = pd.read_csv(CSV_FILE)
    X, y = [], []

    for _, row in df.iterrows():
        file_path = row["filename"]
        features = extract_features(file_path)
        X.append(features)
        y.append(row["emotion"])

    return np.array(X), np.array(y)

def extract_features(file_path):
    """ Extract MFCC features from an audio file. """
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)  # Convert to fixed-length vector
