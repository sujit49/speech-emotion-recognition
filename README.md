# 🗣️ EmotionSpeak: Real-time Voice Emotion Detection

Hey there! Welcome to **EmotionSpeak**, your friendly neighborhood project that can detect emotions from your voice in real-time!  
Ever wondered if your computer can tell if you're happy, sad, or maybe a little bit angry? Well, with EmotionSpeak, it can take a pretty good guess.

This project uses a bit of machine learning magic to listen to your voice, figure out what emotion you're expressing, and then tell you.  
It's built with Python and uses some cool libraries like `librosa` for audio analysis and `PyTorch` for the brainy stuff (the neural network!).

---

## ✨ What Does It Do?

EmotionSpeak does three main things:

1. **Gets your voice ready**  
   It processes raw audio files to extract meaningful features from them using MFCCs (Mel Frequency Cepstral Coefficients).

2. **Learns from emotions**  
   It uses these features to learn patterns associated with different emotions through a neural network.

3. **Predicts in real-time**  
   Once trained, it can listen to new audio (like your voice right now!) and predict what emotion you're conveying.

---

## 🚀 Getting Started

### 1. Setting Up Your Environment

First things first, let's get your computer ready:

- Make sure Python (version 3.7 or newer) is installed.
- Install all necessary libraries:

```bash
pip install torch torchaudio pandas numpy scikit-learn librosa sounddevice
2. The Dataset (Your Voice's Teacher!)
To teach our model about emotions, we need a dataset of audio clips labeled with emotions.
This project supports datasets like RAVDESS or any custom dataset where emotion codes are part of filenames.

Create a dataset folder
In the same directory as the project files, create a folder called:

nginx
Copy
Edit
dataset
Add your audio files
Place your .wav files in the dataset folder.
Filenames should follow a naming convention like:

Copy
Edit
Actor_XX-XX-XX-EMOTIONCODE-XX-XX-XX.wav
(Where EMOTIONCODE like 01 means neutral, 03 means happy, etc.)

3. Prepare the Data (Making Sense of the Sounds)
Once the dataset is ready, generate a CSV file to map filenames to emotion labels.

Run:

bash
Copy
Edit
python generate_csv.py
This will generate:

Copy
Edit
speech_dataset.csv
You'll see:
✅ CSV file speech_dataset.csv created successfully!

4. Train the Emotion Model (Teaching the AI to Feel!)
Now train the neural network to recognize emotions:

bash
Copy
Edit
python trained_model.py
What happens:

Loads and processes audio data.

Extracts MFCC features.

Encodes emotion labels numerically.

Splits data into training and testing sets.

Trains the model and saves it as:

Copy
Edit
trained_emotion_model.pth
✅ Model trained and saved!

5. Predict Emotions in Real-time (Let Your Voice Be Heard!)
Now, test your model in real-time:

bash
Copy
Edit
python main.py
Steps:

Loads your trained model.

Prompts: “Recording for 10 seconds…”

Records your voice.

Analyzes it and prints the detected emotion:

yaml
Copy
Edit
Detected Emotion: Happy
📂 Project Structure
text
Copy
Edit
├── generate_csv.py             # Generates CSV from dataset
├── utils.py                    # Feature extraction (MFCCs)
├── model.py                    # Neural network definition
├── feature_extraction.py       # MFCC extraction from live input
├── trained_model.py            # Trains the emotion model
├── main.py                     # Real-time emotion prediction
├── predict.py                  # Extra prediction script
├── speech_dataset.csv          # (Generated) File to emotion mapping
├── trained_emotion_model.pth   # (Generated) Trained model
├── recorded_audio.wav          # (Generated) Temporary recording
└── dataset/                    # Your input .wav files
🙏 Credits
This project uses the following open-source libraries:

PyTorch – Neural networks

Librosa – Audio analysis

Sounddevice – Audio recording

Pandas – Data handling

NumPy – Numerical operations

Scikit-learn – Label encoding & splitting

Enjoy detecting emotions with your voice! If you have any suggestions, improvements, or questions, feel free to fork the project or open an issue.
