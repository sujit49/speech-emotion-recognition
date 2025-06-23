import torch
import librosa
import numpy as np
from model import EmotionModel
from sklearn.preprocessing import LabelEncoder

# Load label encoder
y = np.load("y.npy", allow_pickle=True)  # Ensure allow_pickle=True to load NumPy array
encoder = LabelEncoder()
encoder.fit(y)

# Define input size (Ensure this matches feature extraction output)
input_size = 13  # Ensure this is consistent with MFCC feature extraction

# Load trained model
model = EmotionModel(input_size)
model.load_state_dict(torch.load("trained_emotion_model.pth"))
model.eval()
