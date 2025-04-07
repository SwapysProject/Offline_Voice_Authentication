# authenticate.py

import sounddevice as sd
import numpy as np
import wave
import os
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from scipy.spatial.distance import cosine
import time
from db_utils import load_voiceprint_db, EXPECTED_EMBEDDING_DTYPE
import noisereduce as nr

# --- Configuration ---
RECORDING_DIR = "recordings"
DURATION = 5
SAMPLE_RATE = 16000
PASSPHRASE = "Hey there Siri, its me, wake up"
SIMILARITY_THRESHOLD = 0.75
# --- Added: Energy Threshold ---
# This value needs tuning. Start low. If it rejects quiet speech, increase it slightly.
# Assumes audio_data is normalized approximately to [-1, 1] range.
MIN_RMS_ENERGY = 0.00125 # Adjust this threshold as needed

# Ensure recording directory exists
os.makedirs(RECORDING_DIR, exist_ok=True)

# Initialize VoiceEncoder
try:
    encoder = VoiceEncoder()
    print("VoiceEncoder initialized.")
except Exception as e:
    print(f"Error initializing VoiceEncoder: {e}")
    exit()

def record_audio(filename, duration, samplerate):
    # ... (function remains the same as before) ...
    print(f"\nGet ready to speak the passphrase: '{PASSPHRASE}'")
    print("Recording in 3..."); time.sleep(1)
    print("Recording in 2..."); time.sleep(1)
    print("Recording in 1..."); time.sleep(1)
    print("Recording...")
    try:
        recording_int16 = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        print("Recording finished.")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(recording_int16.tobytes())
        print(f"Original recording saved to {filename}")
        recording_float32 = recording_int16.astype(np.float32) / 32767.0
        return recording_float32
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        return None

# --- LIVENESS DETECTION (PLACEHOLDER) ---
def is_live_speech(audio_data, samplerate):
    # ... (function remains the same placeholder) ...
    if audio_data is None or len(audio_data) == 0:
        print("Liveness check: No audio data.")
        return False
    print("Performing liveness check (Placeholder)...")
    is_live = True
    print(f"Liveness check result (Placeholder): {'Live' if is_live else 'Potentially Spoofed'}")
    return is_live

# --- Added: Function to calculate RMS Energy ---
def calculate_rms(audio_data):
    """Calculates the Root Mean Square energy of the audio data."""
    if audio_data is None or len(audio_data) == 0:
        return 0.0
    # Ensure it's a numpy array for calculation
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
    return np.sqrt(np.mean(audio_data**2))
# --- End RMS Function ---

def create_embedding_from_audio(audio_data, samplerate):
    """Creates an embedding from audio data, applying noise reduction and energy check."""
    if audio_data is None:
        print("Error: Audio data is None for embedding creation.")
        return None

    try:
        # --- Pre-processing for Noise Reduction ---
        if not isinstance(audio_data, np.ndarray): audio_data = np.array(audio_data)
        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1: audio_data = np.squeeze(audio_data)
        if audio_data.ndim != 1:
            print(f"Error: Audio data is not 1D after processing (shape: {audio_data.shape}). Cannot process.")
            return None
        # print(f"DEBUG: Input to noisereduce - Shape: {audio_data.shape}, Dtype: {audio_data.dtype}") # Keep for debug if needed

        # --- Noise Reduction Step ---
        print("Applying noise reduction...")
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=samplerate, stationary=False, prop_decrease=0.85)
        print("Noise reduction applied.")
        # print(f"DEBUG: Output from noisereduce - Shape: {reduced_noise_audio.shape}, Dtype: {reduced_noise_audio.dtype}") # Keep for debug if needed

        # --- Added: Energy Check ---
        rms_energy = calculate_rms(reduced_noise_audio)
        print(f"Audio RMS Energy after noise reduction: {rms_energy:.6f}")
        if rms_energy < MIN_RMS_ENERGY:
            print(f"Error: Audio energy ({rms_energy:.6f}) is below the threshold ({MIN_RMS_ENERGY}). Assuming silence or insufficient signal.")
            return None # Fail if energy is too low
        # --- End Energy Check ---

        # --- Feature Extraction ---
        if samplerate != 16000:
             print(f"Warning: Resemblyzer expects 16kHz, but sample rate is {samplerate}. Ensure recording uses 16kHz.")

        # Check length (optional redundancy now that we have energy check)
        min_length_samples = int(0.5 * samplerate)
        if len(reduced_noise_audio) < min_length_samples:
            print(f"Warning: Authentication audio might be too short after noise reduction ({len(reduced_noise_audio)} samples).")
            # Could return None here too if desired

        print("Creating embedding...")
        embedding = encoder.embed_utterance(reduced_noise_audio)
        embedding = embedding.astype(EXPECTED_EMBEDDING_DTYPE)
        return embedding

    except np.core._exceptions._ArrayMemoryError as mem_err: # Keep memory error handler
        print(f"\nCRITICAL ERROR: Memory allocation failed during noise reduction.")
        print(f"  Error details: {mem_err}")
        # ... (rest of memory error message) ...
        return None
    except Exception as e:
        print(f"Error creating embedding from audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_embeddings(embedding1, embedding2):
    # ... (function remains the same) ...
    if embedding1 is None or embedding2 is None:
        print("Comparison failed: One or both embeddings are None.")
        return 0.0
    if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
        print("Comparison failed: Inputs must be NumPy arrays.")
        return 0.0
    if embedding1.shape != embedding2.shape:
        print(f"Comparison failed: Embedding shapes mismatch: {embedding1.shape} vs {embedding2.shape}")
        return 0.0
    if embedding1.dtype != EXPECTED_EMBEDDING_DTYPE or embedding2.dtype != EXPECTED_EMBEDDING_DTYPE:
         print(f"Warning: Embedding dtype mismatch during comparison. Ensure both are {EXPECTED_EMBEDDING_DTYPE}.")
         embedding1 = embedding1.astype(EXPECTED_EMBEDDING_DTYPE)
         embedding2 = embedding2.astype(EXPECTED_EMBEDDING_DTYPE)

    try:
        similarity = 1 - cosine(embedding1, embedding2)
        if np.isnan(similarity): return 0.0
        return similarity
    except Exception as e:
        print(f"Error during cosine similarity calculation: {e}")
        return 0.0

# --- VISUALIZATION ---
def display_similarity_bar(score, threshold, width=50):
    # ... (function remains the same) ...
    score = max(0.0, min(1.0, score))
    threshold_pos = int(threshold * width)
    score_pos = int(score * width)
    bar = ['-'] * width
    if 0 <= threshold_pos < width: bar[threshold_pos] = '|'
    if 0 <= score_pos < width: bar[score_pos] = '#'
    threshold_marker_display = f"(Threshold at {'%.2f'%threshold})"
    bar_str = f"0.0 [{''.join(bar)}] 1.0  {threshold_marker_display}"
    print(bar_str)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Voice Authentication ---")
    username = input("Enter username to authenticate: ").strip().lower()
    if not username: print("Username cannot be empty."); exit()

    # 1. Load Stored Voiceprint
    stored_voiceprint = load_voiceprint_db(username)
    if stored_voiceprint is None: exit()

    # Recording Filename
    auth_audio_file_orig = os.path.join(RECORDING_DIR, f"{username}_auth_orig_{int(time.time())}.wav")

    # 2. Record Audio
    auth_audio_data = record_audio(auth_audio_file_orig, DURATION, SAMPLE_RATE)
    if auth_audio_data is None: print("Authentication failed: Could not record audio."); exit()

    # 3. Perform Liveness Check (Placeholder)
    if not is_live_speech(auth_audio_data, SAMPLE_RATE):
        print("Authentication failed: Liveness check failed (Placeholder indicates potential issue or insufficient data).")
        exit()

    # 4. Create Embedding (Now includes energy check)
    auth_embedding = create_embedding_from_audio(auth_audio_data, SAMPLE_RATE)
    # This check is now crucial - if energy was too low, auth_embedding will be None
    if auth_embedding is None:
        print("Authentication failed: Could not create embedding from authentication audio (check signal energy/quality).")
        exit()

    # 5. Compare Embeddings
    similarity_score = compare_embeddings(stored_voiceprint, auth_embedding)

    # 6. Display Results
    print(f"\n--- Authentication Result ---")
    print(f"Similarity Score: {similarity_score:.4f}")
    display_similarity_bar(similarity_score, SIMILARITY_THRESHOLD)
    print(f"Required Threshold: {SIMILARITY_THRESHOLD}")

    # 7. Make Decision
    if similarity_score > SIMILARITY_THRESHOLD:
        print(f"Authentication SUCCESSFUL for {username}!")
    else:
        print(f"Authentication FAILED for {username}.")