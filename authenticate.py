# authenticate.py
import sounddevice as sd
import numpy as np
import wave
import os
from resemblyzer import VoiceEncoder, preprocess_wav # Use resemblyzer imports
from pathlib import Path
from scipy.spatial.distance import cosine
import time
from db_utils import load_voiceprint_db, EXPECTED_EMBEDDING_DTYPE # Make sure db_utils uses correct DB name and DIM
# import noisereduce as nr # Noise reduction disabled

# --- Configuration ---
RECORDING_DIR = "recordings"
DURATION = 10 # Increased duration, match registration
SAMPLE_RATE = 16000 # Resemblyzer preferred rate
# --- IMPORTANT: Tune this threshold ---
# Start lower (e.g., 0.70 or 0.65) and adjust based on genuine vs imposter tests
# This is for Resemblyzer WITHOUT noise reduction.
SIMILARITY_THRESHOLD = 0.75 # <<<--- NEEDS CAREFUL TUNING
# --- Energy Threshold ---
MIN_RMS_ENERGY = 0.00125 # Keep energy check, may need tuning
# --- Prompt ---
AUTH_PROMPT_MESSAGE = f"Please speak clearly for {DURATION} seconds to authenticate."
# --- End Configuration ---

# Ensure recording directory exists
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
    """Records audio, saves WAV, returns filepath."""
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
        # Return the filename for further processing
        return filename
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        return None

# --- LIVENESS DETECTION (PLACEHOLDER) ---
def is_live_speech(audio_data, samplerate): # Needs modification if check requires audio data
    # Current implementation doesn't receive audio data easily
    # when create_embedding takes filepath. Proper liveness needs adjustments.
    print("Performing liveness check (Placeholder)...")
    is_live = True # Placeholder value
    print(f"Liveness check result (Placeholder): {'Live' if is_live else 'Potentially Spoofed'}")
    return is_live

# --- RMS Calculation (Handles Path objects) ---
def calculate_rms_from_file(filepath): # filepath might be a Path object or string
    """Calculates RMS from a WAV file."""
    try:
        # --- FIX: Convert Path object to string for wave.open() ---
        filepath_str = str(filepath)
        # --- End Fix ---

        # Use the string version of the path
        with wave.open(filepath_str, 'rb') as wf:
            n_frames = wf.getnframes()
            if n_frames == 0: return 0.0
            frames = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
            samplerate = wf.getframerate() # Get samplerate for duration check if needed

            # Basic check for minimum duration (e.g., 0.5 sec) based on frames
            min_duration_frames = int(0.5 * samplerate)
            if n_frames < min_duration_frames:
                print(f"Warning: File {filepath_str} is very short ({n_frames/samplerate:.2f}s).")
                # Decide if this should cause RMS check to fail or just warn

            # Convert bytes to numpy array based on sample width
            if sampwidth == 2: # int16
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                # Normalize to float32 for RMS calculation consistency
                audio_float32 = audio_int16.astype(np.float32) / 32767.0
            elif sampwidth == 1: # int8
                audio_int8 = np.frombuffer(frames, dtype=np.uint8) # Assuming unsigned 8-bit
                audio_float32 = (audio_int8.astype(np.float32) - 128.0) / 128.0 # Center and scale
            else: # Add other formats if needed (float32, etc.)
                print(f"Warning: Unsupported sample width {sampwidth} for RMS calculation from file {filepath_str}.")
                return 0.01 # Return non-zero to avoid false rejection if unsure

            return np.sqrt(np.mean(audio_float32**2))
    except FileNotFoundError:
        print(f"Error calculating RMS: File not found at {str(filepath)}") # Use original object for error msg
        return 0.0
    except Exception as e:
        # Include the filepath string in the error message for clarity
        print(f"Error calculating RMS from file {str(filepath)}: {e}")
        return 0.0 # Assume failure means low energy


def create_embedding_from_audio_resemblyzer(audio_filepath):
    """Creates an embedding using Resemblyzer from a file path + energy check."""
    if audio_filepath is None:
        print("Error: Audio filepath is None for embedding creation.")
        return None

    try:
        # --- Noise Reduction (DISABLED) ---
        # If re-enabling, load audio data here first, process with nr,
        # save to a temporary file, and use the temp file path below.
        processed_audio_path = Path(audio_filepath) # Use original path
        # ----------------------------------

        if not processed_audio_path.exists():
            print(f"Error: Audio file not found at {processed_audio_path}")
            return None

        # --- Energy Check (from file) ---
        # Pass the Path object, calculate_rms_from_file handles it
        rms_energy = calculate_rms_from_file(processed_audio_path)
        print(f"Audio RMS Energy: {rms_energy:.6f}")
        if rms_energy < MIN_RMS_ENERGY:
            print(f"Error: Audio energy ({rms_energy:.6f}) below threshold ({MIN_RMS_ENERGY}). Authentication rejected.")
            return None # Fail if energy is too low

        # --- Feature Extraction (Resemblyzer) ---
        print("Preprocessing audio with Resemblyzer...")
        # preprocess_wav handles reading from the path and VAD
        original_wav = preprocess_wav(processed_audio_path)
        if original_wav is None:
             print(f"Error: Resemblyzer preprocess_wav failed for {processed_audio_path}. Audio might be corrupted or invalid format.")
             return None
        if len(original_wav) < int(0.5 * SAMPLE_RATE): # Basic length check after VAD
            # This check might be redundant given the RMS check, but can catch VAD issues
            print(f"Warning: Audio might be too short after VAD ({len(original_wav)/SAMPLE_RATE:.2f}s) for {processed_audio_path}")
            # Optional: return None if too short

        print("Creating embedding using Resemblyzer...")
        # Generate embedding from the preprocessed wav numpy array
        embedding = encoder.embed_utterance(original_wav)

        print(f"Embedding created (shape: {embedding.shape}, type: {embedding.dtype}).")
        # Ensure correct dtype for comparison
        return embedding.astype(EXPECTED_EMBEDDING_DTYPE)

    except Exception as e:
        print(f"Error creating embedding from audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_embeddings(embedding1, embedding2):
    """Calculates cosine similarity. Higher is more similar."""
    if embedding1 is None or embedding2 is None:
        print("Comparison failed: One or both embeddings are None.")
        return 0.0
    if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
        print("Comparison failed: Inputs must be NumPy arrays.")
        return 0.0
    if embedding1.shape != embedding2.shape:
        print(f"Compare Error: Shape mismatch {embedding1.shape} vs {embedding2.shape}")
        return 0.0
    # Ensure correct dtype before comparison
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
        return max(0.0, min(1.0, similarity)) # Clamp
    except Exception as e:
        print(f"Error during cosine similarity calculation: {e}")
        return 0.0

# --- VISUALIZATION ---
def display_similarity_bar(score, threshold, width=50):
    """Displays a simple text-based similarity bar."""
    score = max(0.0, min(1.0, score)) # Clamp score between 0 and 1
    threshold_pos = int(threshold * width)
    score_pos = int(score * width)

    bar = ['-'] * width
    # Place threshold marker (make sure it's within bounds)
    if 0 <= threshold_pos < width:
        bar[threshold_pos] = '|' # Threshold marker

    # Place score marker (make sure it's within bounds)
    if 0 <= score_pos < width:
         # Use '#' for score, potentially overwriting threshold if they coincide
         bar[score_pos] = '#'

    # Mark threshold explicitly if score overwrites it
    threshold_marker_display = f"(Threshold at {'%.2f'%threshold})"

    # Build the string
    bar_str = f"0.0 [{''.join(bar)}] 1.0  {threshold_marker_display}"
    print(bar_str)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Voice Authentication (Resemblyzer - Enhanced) ---")
    username = input("Enter username to authenticate: ").strip().lower()
    if not username: print("Username cannot be empty."); exit()

    # 1. Load Stored Voiceprint (averaged during registration)
    stored_voiceprint = load_voiceprint_db(username)
    if stored_voiceprint is None: exit()

    # Generate filename for the authentication recording
    auth_audio_filepath = os.path.join(RECORDING_DIR, f"{username}_auth_orig_{int(time.time())}.wav")

    # 2. Record audio, get filepath back
    audio_path = record_audio(auth_audio_filepath, DURATION, SAMPLE_RATE, AUTH_PROMPT_MESSAGE)
    if audio_path is None: print("Authentication failed: Could not record audio."); exit()

    # 3. Placeholder Liveness check (doesn't use audio data currently)
    # If proper liveness is implemented, it needs access to audio (data or path)
    if not is_live_speech(None, SAMPLE_RATE):
        print("Authentication failed: Liveness check failed (Placeholder).")
        # Optional: clean up audio file
        try: os.remove(audio_path)
        except OSError: pass
        exit()

    # 4. Create embedding from filepath (includes energy check from file)
    auth_embedding = create_embedding_from_audio_resemblyzer(audio_path)
    if auth_embedding is None:
        print("Authentication failed: Could not create embedding (check signal energy/quality).")
        # Optional: clean up audio file
        try: os.remove(audio_path)
        except OSError: pass
        exit()

    # 5. Optional: Clean up the authentication recording WAV file immediately after processing
    # You might want to keep it for debugging, comment this block out if needed
    try:
        os.remove(audio_path)
        print(f"Cleaned up temporary recording: {audio_path}")
    except OSError as e:
        print(f"Warning: Could not remove temp file {audio_path}: {e}")

    # 6. Compare Embeddings
    similarity_score = compare_embeddings(stored_voiceprint, auth_embedding)

    # 7. Display Results
    print(f"\n--- Authentication Result ---")
    print(f"Similarity Score: {similarity_score:.4f}")
    display_similarity_bar(similarity_score, SIMILARITY_THRESHOLD)
    print(f"Required Threshold: {SIMILARITY_THRESHOLD} (NEEDS TUNING!)")

    # 8. Make Decision
    if similarity_score > SIMILARITY_THRESHOLD:
        print(f"Authentication SUCCESSFUL for {username}!")
    else:
        print(f"Authentication FAILED for {username}.")