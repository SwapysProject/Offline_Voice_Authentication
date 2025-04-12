import sounddevice as sd
import numpy as np
import wave
import os
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from scipy.spatial.distance import cosine
import time
from db_utils import load_voiceprint_db, EXPECTED_EMBEDDING_DTYPE

RECORDING_DIR = "recordings"
DURATION = 10
SAMPLE_RATE = 16000
SIMILARITY_THRESHOLD = 0.75
MIN_RMS_ENERGY = 0.00125
AUTH_PROMPT_MESSAGE = f"Please speak clearly for {DURATION} seconds to authenticate."

os.makedirs(RECORDING_DIR, exist_ok=True)

try:
    encoder = VoiceEncoder()
    print("Resemblyzer VoiceEncoder initialized.")
except Exception as e:
    print(f"Error initializing VoiceEncoder: {e}")
    exit()

def record_audio(filename, duration, samplerate, prompt):
    print(f"\n{prompt}")
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
        return filename
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        return None

def is_live_speech(audio_data, samplerate):
    print("Performing liveness check (Placeholder)...")
    is_live = True
    print(f"Liveness check result (Placeholder): {'Live' if is_live else 'Potentially Spoofed'}")
    return is_live

def calculate_rms_from_file(filepath):
    try:
        filepath_str = str(filepath)
        with wave.open(filepath_str, 'rb') as wf:
            n_frames = wf.getnframes()
            if n_frames == 0: return 0.0
            frames = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
            samplerate = wf.getframerate()

            min_duration_frames = int(0.5 * samplerate)
            if n_frames < min_duration_frames:
                print(f"Warning: File {filepath_str} is very short ({n_frames/samplerate:.2f}s).")

            if sampwidth == 2:
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0
            elif sampwidth == 1:
                audio_int8 = np.frombuffer(frames, dtype=np.uint8)
                audio_float32 = (audio_int8.astype(np.float32) - 128.0) / 128.0
            else:
                print(f"Warning: Unsupported sample width {sampwidth} for RMS calculation from file {filepath_str}.")
                return 0.01

            return np.sqrt(np.mean(audio_float32**2))
    except FileNotFoundError:
        print(f"Error calculating RMS: File not found at {str(filepath)}")
        return 0.0
    except Exception as e:
        print(f"Error calculating RMS from file {str(filepath)}: {e}")
        return 0.0

def create_embedding_from_audio_resemblyzer(audio_filepath):
    if audio_filepath is None:
        print("Error: Audio filepath is None for embedding creation.")
        return None

    try:
        processed_audio_path = Path(audio_filepath)

        if not processed_audio_path.exists():
            print(f"Error: Audio file not found at {processed_audio_path}")
            return None

        rms_energy = calculate_rms_from_file(processed_audio_path)
        print(f"Audio RMS Energy: {rms_energy:.6f}")
        if rms_energy < MIN_RMS_ENERGY:
            print(f"Error: Audio energy ({rms_energy:.6f}) below threshold ({MIN_RMS_ENERGY}). Authentication rejected.")
            return None

        print("Preprocessing audio with Resemblyzer...")
        original_wav = preprocess_wav(processed_audio_path)
        if original_wav is None:
             print(f"Error: Resemblyzer preprocess_wav failed for {processed_audio_path}. Audio might be corrupted or invalid format.")
             return None
        if len(original_wav) < int(0.5 * SAMPLE_RATE):
            print(f"Warning: Audio might be too short after VAD ({len(original_wav)/SAMPLE_RATE:.2f}s) for {processed_audio_path}")

        print("Creating embedding using Resemblyzer...")
        embedding = encoder.embed_utterance(original_wav)

        print(f"Embedding created (shape: {embedding.shape}, type: {embedding.dtype}).")
        return embedding.astype(EXPECTED_EMBEDDING_DTYPE)

    except Exception as e:
        print(f"Error creating embedding from audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_embeddings(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        print("Comparison failed: One or both embeddings are None.")
        return 0.0
    if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
        print("Comparison failed: Inputs must be NumPy arrays.")
        return 0.0
    if embedding1.shape != embedding2.shape:
        print(f"Compare Error: Shape mismatch {embedding1.shape} vs {embedding2.shape}")
        return 0.0
    if embedding1.dtype != EXPECTED_EMBEDDING_DTYPE:
        print(f"Warning: Converting embedding1 dtype from {embedding1.dtype} to {EXPECTED_EMBEDDING_DTYPE} for comparison.")
        embedding1 = embedding1.astype(EXPECTED_EMBEDDING_DTYPE)
    if embedding2.dtype != EXPECTED_EMBEDDING_DTYPE:
        print(f"Warning: Converting embedding2 dtype from {embedding2.dtype} to {EXPECTED_EMBEDDING_DTYPE} for comparison.")
        embedding2 = embedding2.astype(EXPECTED_EMBEDDING_DTYPE)
    try:
        similarity = 1 - cosine(embedding1, embedding2)
        if np.isnan(similarity):
            print("Warning: Cosine similarity resulted in NaN.")
            return 0.0
        return max(0.0, min(1.0, similarity))
    except Exception as e:
        print(f"Error during cosine similarity calculation: {e}")
        return 0.0

def display_similarity_bar(score, threshold, width=50):
    score = max(0.0, min(1.0, score))
    threshold_pos = int(threshold * width)
    score_pos = int(score * width)

    bar = ['-'] * width
    if 0 <= threshold_pos < width:
        bar[threshold_pos] = '|'

    if 0 <= score_pos < width:
         bar[score_pos] = '#'

    threshold_marker_display = f"(Threshold at {'%.2f'%threshold})"

    bar_str = f"0.0 [{''.join(bar)}] 1.0  {threshold_marker_display}"
    print(bar_str)

if __name__ == "__main__":
    print("--- Voice Authentication (Resemblyzer - Enhanced) ---")
    username = input("Enter username to authenticate: ").strip().lower()
    if not username: print("Username cannot be empty."); exit()

    stored_voiceprint = load_voiceprint_db(username)
    if stored_voiceprint is None: exit()

    auth_audio_filepath = os.path.join(RECORDING_DIR, f"{username}_auth_orig_{int(time.time())}.wav")

    audio_path = record_audio(auth_audio_filepath, DURATION, SAMPLE_RATE, AUTH_PROMPT_MESSAGE)
    if audio_path is None: print("Authentication failed: Could not record audio."); exit()

    if not is_live_speech(None, SAMPLE_RATE):
        print("Authentication failed: Liveness check failed (Placeholder).")
        try: os.remove(audio_path)
        except OSError: pass
        exit()

    auth_embedding = create_embedding_from_audio_resemblyzer(audio_path)
    if auth_embedding is None:
        print("Authentication failed: Could not create embedding (check signal energy/quality).")
        try: os.remove(audio_path)
        except OSError: pass
        exit()

    try:
        os.remove(audio_path)
        print(f"Cleaned up temporary recording: {audio_path}")
    except OSError as e:
        print(f"Warning: Could not remove temp file {audio_path}: {e}")

    similarity_score = compare_embeddings(stored_voiceprint, auth_embedding)

    print(f"\n--- Authentication Result ---")
    print(f"Similarity Score: {similarity_score:.4f}")
    display_similarity_bar(similarity_score, SIMILARITY_THRESHOLD)
    print(f"Required Threshold: {SIMILARITY_THRESHOLD} (NEEDS TUNING!)")

    if similarity_score > SIMILARITY_THRESHOLD:
        print(f"Authentication SUCCESSFUL for {username}!")
    else:
        print(f"Authentication FAILED for {username}.")