# register.py
import sounddevice as sd
import numpy as np
import wave
import os
from resemblyzer import VoiceEncoder, preprocess_wav # Use resemblyzer imports
from pathlib import Path
import time
from db_utils import check_user_exists, save_voiceprint_db, EXPECTED_EMBEDDING_DTYPE
# import noisereduce as nr # Noise reduction disabled

# --- Configuration ---
RECORDING_DIR = "recordings"
DURATION = 10 # Increased duration
SAMPLE_RATE = 16000 # Resemblyzer preferred rate
PASSPHRASE = "The quick brown fox jumps over the lazy dog" # Phonetically rich
NUM_ENROLL_SAMPLES = 3
# --- End Configuration ---

os.makedirs(RECORDING_DIR, exist_ok=True)

# Initialize Resemblyzer VoiceEncoder
try:
    # Optionally specify device 'cpu' if torch GPU issues arise
    # encoder = VoiceEncoder(device='cpu')
    encoder = VoiceEncoder()
    print("Resemblyzer VoiceEncoder initialized.")
except Exception as e:
    print(f"Error initializing VoiceEncoder: {e}")
    exit()

def record_audio(filename, duration, samplerate, prompt):
    """Records audio, saves WAV, returns float32 numpy array."""
    print(f"\n{prompt}")
    print("Recording in 3..."); time.sleep(1)
    print("Recording in 2..."); time.sleep(1)
    print("Recording in 1..."); time.sleep(1)
    print("Recording...")
    try:
        recording_int16 = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        print("Recording finished.")
        # Save backup WAV using the int16 data
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(recording_int16.tobytes())
        print(f"Original recording saved to {filename}")
        # Convert to float32 for potential processing (though Resemblyzer can take path)
        # Resemblyzer's preprocess_wav works best with file paths
        # Let's keep returning the path for Resemblyzer's preprocess_wav
        # Return the filename instead of audio data
        return filename
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        return None

def create_single_voiceprint_resemblyzer(audio_filepath):
    """Creates a single embedding using Resemblyzer from a file path."""
    if audio_filepath is None: return None

    try:
        # --- Noise Reduction (DISABLED) ---
        # If re-enabling, load audio, process with nr, save temp file, pass temp path
        processed_audio_path = Path(audio_filepath)
        # ----------------------------------

        if not processed_audio_path.exists():
            print(f"Error: Audio file not found at {processed_audio_path}")
            return None

        print("Preprocessing audio with Resemblyzer...")
        # Use preprocess_wav which handles VAD and uses the internal sample rate
        original_wav = preprocess_wav(processed_audio_path)
        if original_wav is None or len(original_wav) < int(0.5 * SAMPLE_RATE): # Basic length check after VAD
            print(f"Warning: Audio might be too short or silent after VAD for {audio_filepath}")
            # Optional: return None if too short

        print("Creating embedding using Resemblyzer...")
        # Generate embedding from the preprocessed wav
        embedding = encoder.embed_utterance(original_wav)

        print(f"Embedding created (shape: {embedding.shape}, type: {embedding.dtype}).")
        # Ensure correct dtype for DB
        return embedding.astype(EXPECTED_EMBEDDING_DTYPE)

    except Exception as e:
        print(f"Error creating voiceprint: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("--- Voice Registration (Resemblyzer - Multi-Sample) ---")
    while True:
        username = input("Enter username for registration: ").strip().lower()
        if not username: print("Username cannot be empty.")
        elif check_user_exists(username):
            overwrite = input(f"Username '{username}' already exists. Overwrite? (y/n): ").lower()
            if overwrite == 'y': break
            else: print("Please choose a different username.")
        else: break

    enroll_embeddings = []
    print(f"\nNeed to record {NUM_ENROLL_SAMPLES} samples for enrollment.")

    for i in range(NUM_ENROLL_SAMPLES):
        print("-" * 20)
        print(f"Recording sample {i+1}/{NUM_ENROLL_SAMPLES}...")
        prompt = f"Sample {i+1}/{NUM_ENROLL_SAMPLES}: Please say '{PASSPHRASE}' clearly."
        # Generate filename for this specific sample
        reg_audio_filepath = os.path.join(RECORDING_DIR, f"{username}_reg_{i+1}_{int(time.time())}.wav")

        # Record audio, get filepath back
        audio_path = record_audio(reg_audio_filepath, DURATION, SAMPLE_RATE, prompt)
        if audio_path is None:
            print(f"Registration failed: Could not record audio for sample {i+1}.")
            exit()

        # Create embedding for this sample using the filepath
        voiceprint_sample = create_single_voiceprint_resemblyzer(audio_path)
        if voiceprint_sample is None:
            print(f"Registration failed: Could not create voiceprint for sample {i+1}.")
            # Optional: clean up audio file
            # try: os.remove(audio_path)
            # except OSError: pass
            exit()

        enroll_embeddings.append(voiceprint_sample)
        print(f"Sample {i+1} processed.")
        if i < NUM_ENROLL_SAMPLES - 1: time.sleep(1)

        # Optional: Clean up the individual recording WAV file after processing
        # try: os.remove(audio_path)
        # except OSError as e: print(f"Warning: Could not remove temp file {audio_path}: {e}")

    if len(enroll_embeddings) != NUM_ENROLL_SAMPLES:
         print(f"Error: Did not collect enough enrollment samples. Aborting.")
         exit()

    print("\nAveraging enrollment embeddings...")
    try:
        averaged_voiceprint = np.mean(np.stack(enroll_embeddings), axis=0)
        averaged_voiceprint = averaged_voiceprint.astype(EXPECTED_EMBEDDING_DTYPE)
        print(f"Averaged voiceprint calculated (shape: {averaged_voiceprint.shape}, type: {averaged_voiceprint.dtype}).")
    except Exception as e:
         print(f"Error averaging embeddings: {e}")
         exit()

    if not save_voiceprint_db(username, averaged_voiceprint):
        print("Registration failed: Could not save averaged voiceprint to database.")
        exit()

    print(f"\nRegistration successful for user: {username}")
    print(f"(Averaged from {NUM_ENROLL_SAMPLES} samples)")