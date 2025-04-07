# Simple Offline Voice Authentication System (Python)

## Objective

This project implements a basic voice authentication system in Python that operates entirely offline after an initial setup phase. It allows users to register their voiceprint associated with a username (stored in an SQLite database) and then authenticate using their voice later. Key features include basic noise reduction, an energy check to prevent authentication on silence, similarity score display with visualization, and a placeholder for liveness detection.

## How it Works

1.  **Database Setup:** A simple SQLite database (`voice_auth.db`) is used, managed by `db_utils.py`. It contains a single table `users` with columns: `username` (Primary Key), `voiceprint` (BLOB), and `registration_date` (TEXT). The database and table are automatically created on the first run if they don't exist.

2.  **Registration (`register.py`):**
    *   The user provides a username.
    *   The script checks if the username already exists in the database. If so, it prompts for overwrite confirmation.
    *   The user is prompted to speak a fixed passphrase ("My voice is my password, verify me.").
    *   A short audio clip (e.g., 5 seconds) is recorded using `sounddevice` and saved as a float32 NumPy array. An original WAV file backup is also saved in `recordings/`.
    *   Basic **noise reduction** (spectral gating via `noisereduce` library) is applied to the recorded audio data.
    *   `resemblyzer` is used to extract a speaker embedding (a vector representing the voice characteristics) from the *noise-reduced* audio. This embedding is the "voiceprint".
    *   The voiceprint (NumPy array) is converted to raw bytes (BLOB).
    *   The username, voiceprint BLOB, and current timestamp are saved into the `users` table in `voice_auth.db` using an `INSERT OR REPLACE` command via `db_utils.py`.

3.  **Authentication (`authenticate.py`):**
    *   The user provides their username.
    *   The script queries the `voice_auth.db` database to retrieve the stored voiceprint BLOB for that username using `db_utils.py`.
    *   The BLOB is converted back into a NumPy array.
    *   The user is prompted to speak the *same* fixed passphrase.
    *   A new audio clip is recorded as a float32 NumPy array (original WAV backup saved in `recordings/`).
    *   A **placeholder Liveness Check** is performed. **WARNING:** This check currently always returns `True` and does not provide real security against recorded playback.
    *   **Noise reduction** is applied to the new audio data.
    *   An **Energy Check** (Root Mean Square - RMS) is performed on the noise-reduced audio. If the energy is below a defined threshold (`MIN_RMS_ENERGY`), authentication fails immediately (prevents passing on silence).
    *   If the energy check passes, `resemblyzer` extracts an embedding from the noise-reduced authentication audio.
    *   The system calculates the cosine similarity between the stored voiceprint and the newly generated embedding using `scipy.spatial.distance.cosine` (Similarity = 1 - Cosine Distance).
    *   The result is displayed, including the numerical **similarity score** and a text-based **visualization bar** showing the score relative to the acceptance threshold.
    *   If the similarity score exceeds a predefined `SIMILARITY_THRESHOLD` (default: 0.75), the energy check passed, and the (placeholder) liveness check passed, authentication is successful; otherwise, it fails.

## Features

*   **Voice Registration:** Records user voice speaking a fixed passphrase and saves a voiceprint to an SQLite database.
*   **Voice Authentication:** Compares new voice input against a stored voiceprint from the database.
*   **Offline Capability:** Works entirely offline *after* initial library installation and model download.
*   **Basic CLI:** Simple command-line interface for registration and authentication.
*   **Multi-user Support:** Managed via unique usernames (primary key) in the SQLite database.
*   **Database Storage:** Uses SQLite (`voice_auth.db`) for storing voiceprints, managed via `db_utils.py`.
*   **Noise Reduction:** Applies basic spectral gating noise reduction (`noisereduce` library) to recorded audio before feature extraction.
*   **Silence Detection:** Includes an RMS energy check to fail authentication if the input signal after noise reduction is too quiet (likely silence).
*   **Liveness Detection (Placeholder):** Includes a placeholder function where liveness detection logic would reside. **WARNING: Does not provide actual anti-spoofing.**
*   **Confidence Score:** Displays the calculated cosine similarity score during authentication.
*   **Similarity Visualization:** Displays a simple text-based bar in the console showing the similarity score relative to the threshold.

## Requirements

*   Python 3.7+
*   Libraries listed in `requirements.txt`:
    *   `numpy`
    *   `scipy`
    *   `sounddevice`
    *   `wave`
    *   `resemblyzer`
    *   `librosa`
    *   `torch` (CPU version is sufficient)
    *   `torchaudio`
    *   `noisereduce` (for noise reduction)
*   **PortAudio:** A cross-platform audio I/O library required by `sounddevice`. Installation varies by OS (see Installation section).

## Installation

1.  **Clone or download the repository.**
2.  **Install PortAudio:**
    *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install libportaudio2 portaudio19-dev`
    *   **macOS:** `brew install portaudio`
    *   **Windows:** Download pre-compiled binaries or use `choco install portaudio`. Sometimes installing `pip install sounddevice` or `pip install PyAudio` (using unofficial wheels if needed) handles this.
3.  **Install Python dependencies:** Navigate to the project directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Resemblyzer Model (First Run Only):**
    The *first time* you run `python register.py` or `python authenticate.py`, the `resemblyzer` library will automatically download its pre-trained speaker encoder model files from the internet and cache them locally. **This step requires an internet connection.** Subsequent runs will use the downloaded model and work entirely offline.

## Database

*   The system uses an SQLite database file named `voice_auth.db`, created in the same directory as the scripts.
*   It contains one table: `users (username TEXT PRIMARY KEY, voiceprint BLOB, registration_date TEXT)`.
*   The `voiceprint` column stores the NumPy embedding array as a Binary Large Object (BLOB).
*   All database interactions are handled by functions in `db_utils.py`.

## How to Run

1.  **Setup:** Ensure all requirements from `requirements.txt` and PortAudio are installed. Make sure you have run the scripts at least once *with internet access* to allow the Resemblyzer model to download.
2.  **Run Registration:**
    ```bash
    python register.py
    ```
    *   Follow prompts to enter a username and record the passphrase.
3.  **Run Authentication:**
    ```bash
    python authenticate.py
    ```
    *   Follow prompts to enter the registered username and record the passphrase.
    *   The system performs checks (liveness placeholder, noise reduction, energy check), compares embeddings, and displays the similarity score, visualization bar, and PASS/FAIL result.
4.  **Database & Recordings:** The `voice_auth.db` file stores user data. The `recordings/` directory contains original WAV backups of each recording session.

## Offline Capability Explained

*   **Initial Setup (Requires Internet):** Installing Python libraries (`pip install`) and the automatic one-time download of the speaker embedding model by `resemblyzer`.
*   **Runtime (Fully Offline):** Once set up, all operations run locally:
    *   Audio I/O (`sounddevice`).
    *   Noise reduction (`noisereduce`).
    *   Speaker embedding (`resemblyzer` using the cached local model).
    *   Database operations (`sqlite3`).
    *   Calculations (`numpy`, `scipy`).

## Preprocessing Steps

1.  **Audio Recording:** Captured via `sounddevice`.
2.  **Noise Reduction:** `noisereduce.reduce_noise` applied to the raw audio data.
3.  **Energy Check:** RMS energy calculated on noise-reduced audio; below-threshold signals are rejected.
4.  **Resemblyzer Internal:** The `resemblyzer.embed_utterance` function processes the validated, noise-reduced audio (likely performing VAD, resampling) before feeding it to the speaker embedding model.

## Performance Observations & Tuning

*   **Threshold Tuning (`SIMILARITY_THRESHOLD`):** Crucial for balancing security (FAR) and usability (FRR). Default is 0.75; adjust based on testing.
*   **Energy Threshold (`MIN_RMS_ENERGY`):** Prevents authentication on silence. Default is 0.005; adjust based on microphone sensitivity and ambient noise. If it rejects quiet-but-valid speech, decrease slightly. If it passes on obvious background noise, increase slightly.
*   **Passphrase Consistency:** Saying the passphrase similarly improves accuracy.
*   **Noise Reduction:** Helps with steady background noise. May need tuning (`prop_decrease`) or might not handle dynamic noise well.
*   **Microphone Consistency:** Use the same mic for registration and authentication if possible.
*   **Speaker Distinguishability:** Should reliably reject different speakers, but very similar voices might require a higher threshold. Test this thoroughly!

## Limitations & Potential Improvements

*   **Liveness Detection:** **THIS IS THE MOST SIGNIFICANT LIMITATION.** The placeholder provides no real anti-spoofing capability against recorded playback. Robust offline liveness detection is a major challenge.
*   **Noise Robustness:** Basic noise reduction helps but isn't perfect. Performance degrades in highly noisy/dynamic environments. Advanced techniques exist but add complexity.
*   **Security:** This is a demonstration. Real-world systems need rigorous analysis (EER, DET curves), hardening against various attacks, and secure key management if integrated into larger systems.
*   **Visualization:** Text bar is basic. GUI libraries (Tkinter, PyQt, etc.) could offer better plots.
*   **Error Handling:** Could be more granular (e.g., specific mic access errors).
*   **Database Schema Evolution:** Changes to embedding format require manual database updates or migration scripts (not included).
*   **C++ Implementation:** A C++ version (bonus point) requires significantly different libraries and model handling.