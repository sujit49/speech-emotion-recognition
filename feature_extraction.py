import librosa
import numpy as np

def extract_live_features(filename):
    """ Extract MFCC features from recorded audio. """
    y, sr = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)  # Ensure consistent feature size


