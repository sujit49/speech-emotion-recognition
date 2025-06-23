import os
import pandas as pd

# Path to the dataset folder (Update this if needed)
DATASET_DIR = "dataset"
CSV_FILE = "speech_dataset.csv"

# Emotion mapping based on RAVDESS naming conventions
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Check if dataset folder exists
if not os.path.exists(DATASET_DIR):
    print(f"❌ ERROR: Dataset folder '{DATASET_DIR}' does not exist.")
    exit()

data = []

# Walk through all subfolders in dataset/
for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) < 3:
                print(f"⚠️ Skipping invalid file: {file}")
                continue  # Skip invalid filenames

            emotion_code = parts[2][:2]  # Extract emotion code
            emotion = EMOTIONS.get(emotion_code, "unknown")  # Map to label

            file_path = os.path.join(root, file)  # Full path to file
            data.append([file_path, emotion])  # Save full path

print(f"✅ Found {len(data)} valid .wav files.")  # Debugging print

# Save to CSV
df = pd.DataFrame(data, columns=["filename", "emotion"])
df.to_csv(CSV_FILE, index=False)

print(f"✅ CSV file '{CSV_FILE}' created successfully!")


