# 🗣️ EmotionSpeak: Real-time Voice Emotion Detection

Hey there! 👋 Welcome to **EmotionSpeak**, your friendly neighborhood project that can detect emotions from your voice in real-time! Ever wondered if your computer can tell if you're happy, sad, or maybe a little bit angry? Well, with EmotionSpeak, it can take a pretty good guess!

This project uses a bit of machine learning magic ✨ to listen to your voice, figure out what emotion you're expressing, and then tell you. It's built with Python and uses some cool libraries like `librosa` for audio analysis and `PyTorch` for the brainy stuff (the neural network!).

## ✨ What Does It Do?

In a nutshell, EmotionSpeak does three main things:

1.  **Gets your voice ready**: It processes raw audio files to extract meaningful features from them. Think of it like teaching the computer to "hear" the important parts of your voice, not just the sounds themselves.
2.  **Learns from emotions**: It then uses these "features" to learn patterns associated with different emotions. We train a special kind of artificial brain (a neural network) to recognize these patterns.
3.  **Predicts in real-time**: Once trained, it can listen to new audio (like your voice right now!) and predict what emotion you're conveying.

## 🚀 Getting Started

Ready to give it a try? Follow these simple steps!

### 1. Setting Up Your Environment

First things first, let's get your computer ready.

* **Python**: Make sure you have Python installed (version 3.7 or newer is recommended).
* **Install Dependencies**: Open your terminal or command prompt and run this command. It will install all the necessary libraries:
    ```bash
    pip install torch torchaudio pandas numpy scikit-learn librosa sounddevice
    ```

### 2. The Dataset (Your Voice's Teacher! 🧑‍🏫)

To teach our model about emotions, we need a dataset of voices with known emotions. This project is designed to work with a dataset where audio filenames contain emotion codes (like the [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)).

* **Create a `dataset` folder**: In the same directory where you've saved all these project files, create a new folder named `dataset`.
* **Populate `dataset`**: Place your `.wav` audio files inside this `dataset` folder. **Important**: The filenames should ideally follow a convention that allows for emotion extraction (e.g., `Actor_XX-XX-XX-EMOTIONCODE-XX-XX-XX.wav`, where `EMOTIONCODE` is a two-digit number like "01" for neutral, "03" for happy, etc. This is what the `generate_csv.py` script expects).

### 3. Prepare the Data (Making Sense of the Sounds 🎧)

Now that you have your audio files, let's create a list of them with their corresponding emotions.

* **Run `generate_csv.py`**:
    ```bash
    python generate_csv.py
    ```
    This script will go through your `dataset` folder, look for `.wav` files, try to figure out their emotions from their filenames, and then create a file called `speech_dataset.csv`. This CSV file will be a handy list of all your audio files and their detected emotions. You should see a message like "✅ CSV file 'speech_dataset.csv' created successfully!" if all goes well.

### 4. Train the Emotion Model (Teaching the AI to Feel! ❤️)

This is where the magic happens! We'll now train our neural network to understand emotions from the audio features.

* **Run `trained_model.py`**:
    ```bash
    python trained_model.py
    ```
    This script will:
    * Load the audio data and extract features from each file.
    * Encode the emotion labels into numbers (because computers understand numbers better than words like "happy" or "sad").
    * Split your data into training and testing sets.
    * Train the `EmotionModel` using your data. You'll see progress updates like "Epoch [X/Y], Loss: Z.ZZZZ" as it learns.
    * Once training is complete, it will save the trained model as `trained_emotion_model.pth`. This is your model's "brain"! You should see "✅ Model trained and saved!"

### 5. Predict Emotions in Real-time (Let Your Voice Be Heard! 🗣️)

Finally, the exciting part! Let's see your model in action.

* **Run `main.py`**:
    ```bash
    python main.py
    ```
    This script will:
    * Load the `trained_emotion_model.pth` that you just created.
    * Prompt you to speak: "🎤 Recording for 10 seconds... Please speak naturally."
    * **Record your voice**: It will record your voice for 10 seconds. Try to express an emotion!
    * **Analyze and predict**: After recording, it will analyze your voice and tell you the detected emotion! You'll see something like "🎭 Detected Emotion: Happy" (hopefully!).

## 📂 Project Structure

Here's a quick peek at what each file does:

* `generate_csv.py`: Scans your `dataset` folder and creates `speech_dataset.csv`.
* `utils.py`: Contains helper functions for loading data and extracting audio features (MFCCs).
* `model.py`: Defines the neural network architecture (`EmotionModel`).
* `feature_extraction.py`: Specifically for extracting MFCC features from live audio recordings.
* `trained_model.py`: The script that trains the `EmotionModel` and saves it.
* `main.py`: The main script to record your voice and predict emotions in real-time.
* `predict.py`: (Note: This file seems to be a partial script for prediction, `main.py` handles the full real-time prediction flow).
* `speech_dataset.csv`: (Generated) The CSV file mapping audio file paths to emotions.
* `trained_emotion_model.pth`: (Generated) The saved "brain" of your emotion detection model.
* `recorded_audio.wav`: (Generated) The temporary audio file of your recording.

## 🙏 Credits

This project leverages the power of several open-source libraries:

* [PyTorch](https://pytorch.org/): For building and training the neural network.
* [Librosa](https://librosa.org/): For audio analysis and feature extraction.
* [Pandas](https://pandas.pydata.org/): For data manipulation.
* [NumPy](https://numpy.org/): For numerical operations.
* [Scikit-learn](https://scikit-learn.org/): For data splitting and label encoding.
* [Sounddevice](https://python-sounddevice.readthedocs.io/): For audio recording.

Enjoy detecting emotions with your voice! If you have any questions or run into issues, feel free to ask!
