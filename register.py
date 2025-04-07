import sounddevice as sd
import numpy as np
import wave
import os
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import time
from db_utils import check_user_exists, save_voiceprint_db, EXPECTED_EMBEDDING_DTYPE
import noisereduce as nr

RECORDING_DIR = "recordings"
DURATION = 5
SAMPLE_RATE = 16000
PASSPHRASE = "Hey there Siri, its me, wake up"


os.makedirs(RECORDING_DIR, exist_ok=True)

try:
    encoder = VoiceEncoder()
    print("VoiceEncoder initialized.")
except Exception as e:
    print(f"Error initializing VoiceEncoder: {e}")
    exit()

def record_audio(filename, duration, samplerate):
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

def create_voiceprint(username, audio_data, samplerate):
    if audio_data is None:
        print("Error: Audio data is None.")
        return None

    try:
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)

        if audio_data.dtype != np.float32:
            print(f"DEBUG: Converting audio data from {audio_data.dtype} to float32.")
            audio_data = audio_data.astype(np.float32)

        if audio_data.ndim > 1:
            print(f"DEBUG: Reshaping audio data from {audio_data.shape} to 1D.")
            audio_data = np.squeeze(audio_data) 

        if audio_data.ndim != 1:
            print(f"Error: Audio data is not 1D after processing (shape: {audio_data.shape}). Cannot apply noise reduction.")
            return None

        print(f"DEBUG: Input to noisereduce - Shape: {audio_data.shape}, Dtype: {audio_data.dtype}")

        print("Applying noise reduction...")
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=samplerate, stationary=False, prop_decrease=0.85)
        print("Noise reduction applied.")
        print(f"DEBUG: Output from noisereduce - Shape: {reduced_noise_audio.shape}, Dtype: {reduced_noise_audio.dtype}")


        if samplerate != 16000:
             print(f"Warning: Resemblyzer expects 16kHz, but sample rate is {samplerate}. Ensure recording uses 16kHz.")

        min_length_samples = int(0.5 * samplerate)
        if len(reduced_noise_audio) < min_length_samples:
            print(f"Warning: Audio for {username} might be too short after noise reduction ({len(reduced_noise_audio)} samples).")

        print("Creating embedding...")
        embedding = encoder.embed_utterance(reduced_noise_audio)
        embedding = embedding.astype(EXPECTED_EMBEDDING_DTYPE)
        print(f"Voiceprint created for {username}.")
        return embedding

    except np.core._exceptions._ArrayMemoryError as mem_err:
        print(f"\nCRITICAL ERROR: Memory allocation failed during noise reduction.")
        print(f"  Error details: {mem_err}")
        print(f"  This usually means the input audio format was unexpected or there's an issue in the 'noisereduce' library.")
        print(f"  Input audio details just before error:")
        print(f"      Shape: {audio_data.shape if 'audio_data' in locals() else 'N/A'}")
        print(f"      Dtype: {audio_data.dtype if 'audio_data' in locals() else 'N/A'}")
        print(f"  Consider simplifying noise reduction parameters or reporting the issue to the library authors.")
        print(f"  Skipping noise reduction for this attempt might work, but is not ideal.")
        return None

    except Exception as e:
        print(f"Error creating voiceprint for {username}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("--- Voice Registration ---")
    while True:
        username = input("Enter username for registration: ").strip().lower()
        if not username:
            print("Username cannot be empty.")
        elif check_user_exists(username):
            overwrite = input(f"Username '{username}' already exists in database. Overwrite? (y/n): ").lower()
            if overwrite == 'y':
                break
            else:
                print("Please choose a different username.")
        else:
            break

    reg_audio_file_orig = os.path.join(RECORDING_DIR, f"{username}_reg_orig_{int(time.time())}.wav")

    recorded_audio_data = record_audio(reg_audio_file_orig, DURATION, SAMPLE_RATE)
    if recorded_audio_data is None:
        print("Registration failed: Could not record audio.")
        exit()

    voiceprint = create_voiceprint(username, recorded_audio_data, SAMPLE_RATE)

    if voiceprint is None:
        print("Registration failed: Could not create voiceprint (check audio quality/length, noise reduction).")
        exit()

    if not save_voiceprint_db(username, voiceprint):
        print("Registration failed: Could not save voiceprint to database.")
        exit()

    print(f"\nRegistration successful for user: {username}")