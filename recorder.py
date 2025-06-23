import sounddevice as sd
import sounddevice as sd
import wavio

def record_audio(filename="live_audio.wav", duration=3, samplerate=44100):
    """ Record live audio and save it. """
    print("🎙️ Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wavio.write(filename, audio, samplerate, sampwidth=2)
    print(f"✅ Audio saved as {filename}")
    return filename  # Return the saved filename

