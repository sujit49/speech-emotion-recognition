import torch
from model import EmotionModel  # Load trained model
from feature_extraction import extract_live_features  # Feature extraction function
from recorder import record_audio  # Audio recording function

MODEL_PATH = "trained_emotion_model.pth"  # Ensure this is the saved model path

# Load trained model
input_size = 13  # Ensure this matches the MFCC feature size
model = EmotionModel(input_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set model to evaluation mode

print("🎤 Recording for 10 seconds... Please speak naturally.")

# Record for 3 minutes (180 seconds)
recorded_audio_path = record_audio(duration=10, filename="recorded_audio.wav")

print("✅ Recording complete. Now analyzing emotions...")

# Extract features from recorded audio
features = extract_live_features(recorded_audio_path)

# Convert features to tensor
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Predict emotion
with torch.no_grad():
    output = model(features_tensor)
    predicted_emotion = torch.argmax(output, dim=1).item()

# Emotion labels
emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
predicted_label = emotion_labels[predicted_emotion]

print(f"🎭 Detected Emotion: {predicted_label}")

