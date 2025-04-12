# Simple Offline Voice Authentication System (Python - Resemblyzer Enhanced)

## Objective

This project implements a basic voice authentication system in Python that operates entirely offline after an initial setup phase. It allows users to register their voiceprint associated with a username (stored in an SQLite database) and then authenticate using their voice later. Key features include multi-sample enrollment for robustness, an energy check to prevent authentication on silence, similarity score display with visualization, and a placeholder for liveness detection. Noise reduction is available but currently disabled due to test results showing better speaker discrimination without it in this setup.

## How it Works

1.  **Database Setup:** A simple SQLite database (`voice_auth_resemblyzer.db`) is used, managed by `db_utils.py`. It contains a single table `users` with columns: `username` (Primary Key), `voiceprint` (BLOB - stores a 256-dimension embedding), and `registration_date` (TEXT). The database and table are automatically created on the first run if they don't exist.

2.  **Registration (`register.py`):**
    *   The user provides a username.
    *   The script checks if the username already exists in the database. If so, it prompts for overwrite confirmation.
    *   The user is prompted to speak a **fixed, phonetically rich passphrase** (e.g., "The quick brown fox...") **multiple times** (default: 3).
    *   Each recording (default: 10 seconds) is saved as a WAV file backup in `recordings/`.
    *   `resemblyzer` (`preprocess_wav` and `embed_utterance`) extracts a speaker embedding (voiceprint) from each valid recording.
    *   These multiple embeddings are **averaged** to create a more stable reference voiceprint for the user.
    *   The averaged voiceprint (NumPy array converted to BLOB) and timestamp are saved into the `users` table in `voice_auth_resemblyzer.db`.

3.  **Authentication (`authenticate.py`):**
    *   The user provides their username.
    *   The script queries the database to retrieve the stored (averaged) voiceprint BLOB.
    *   The BLOB is converted back into a NumPy array.
    *   The user is given a **generic prompt** to speak clearly for the set duration (e.g., 10 seconds). While any speech can be used, using the * via unique usernames (primary key) in the SQLite database.
*   **Database Storage:** Uses SQLite (`voice_auth_resemblyzer.db`) for storing voiceprints.
*   **Silence Detection:** Includes an RMS energy check on the recorded audio file to fail authentication if the input signal is too quiet.
*   **Liveness Detection (Placeholder):** Includes a placeholder function where liveness detection logic would reside. **WARNING: Does not provide actual anti-spoofing.**
*   **Confidence Score:** Displays the calculated cosine similarity score.
*   **Similarity Visualization:** Displays a simple text-based bar comparing the score to the threshold.
*   **Increased Duration:** Uses longer recording time (10s) to potentially capture more stable embeddings.

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
*   **PortAudio:** A cross-platform audio I/O library required by `sounddevice`.

## Installation

1.  **Clone or download the repository.**
2.  **Install PortAudio:** (See previous instructions based on OS).
3.  **Install Python dependencies:** Navigate to the project directory and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Resemblyzer Model (First Run Only):** The *first time* you run `python register.py` or `python authenticate.py`, `resemblyzer` will download its model (requires internet). Subsequent runs are offline.

## Database

*   The system uses an SQLite database file named `voice_auth_resemblyzer.db`.
*   It contains one table: `users (username TEXT PRIMARY KEY, voiceprint BLOB, registration_date TEXT)`.
*   The `voiceprint` column stores the *averaged* NumPy embedding array as a BLOB.

## How to Run

1.  **Setup:** Ensure requirements are installed and the Resemblyzer model has been downloaded (run once with internet). Delete any older `.db` files if switching configurations.
2.  **Run Registration:**
    ```bash
    python register.py
    ```
    *   Follow prompts to enter a username.
    *   Record the fixed passphrase (e.g., "The quick brown fox...") clearly **3 times**.
3.  **Run Authentication:**
    ```bash
    python authenticate.py
    ```
    *   Enter the registered username.
    *   Speak clearly for 10 seconds (any phrase).
    *   The system performs checks (liveness placeholder, energy check), compares embeddings, and displays the result.
4.  **Database & Recordings:** `voice_auth_resemblyzer.db` stores user data. `recordings/` contains original WAV backups.

## Offline Capability Explained

*   **Initial Setup (Requires Internet):** Installing Python libraries (`pip install`) and the automatic one-time download of the speaker embedding model by `resemblyzer`.
*   **Runtime (Fully Offline):** Once set up, all operations run locally using `sounddevice`, `resemblyzer` (with cached model), `sqlite3`, `numpy`, `scipy`. Noise reduction (if re-enabled) also runs offline.

## Preprocessing Steps

1.  **Audio Recording:** Captured via `sounddevice`, saved as WAV.
2.  **Energy Check (Auth):** RMS energy calculated from the saved WAV file; below-threshold signals rejected.
3.  **Resemblyzer `preprocess_wav`:** Reads audio from file path, performs internal processing (likely VAD, resampling).
4.  **Resemblyzer `embed_utterance`:** Generates embedding from preprocessed audio data.
5.  **Averaging (Registration):** Embeddings from multiple recordings are averaged.

## Performance Observations & Tuning

*   **Threshold Tuning (`SIMILARITY_THRESHOLD`):** **CRITICAL.** Start around 0.70 and adjust based *rigorously* on genuine user vs. imposter testing. The goal is to find a value that maximizes the separation.
*   **Multi-Sample Enrollment:** Averaging multiple registration samples aims to create a more stable and representative voiceprint, potentially improving robustness.
*   **Longer Duration:** 10 seconds provides more data for embedding generation, potentially improving stability.
*   **Energy Threshold (`MIN_RMS_ENERGY`):** Prevents authentication on silence. Tune if it incorrectly rejects quiet speech or accepts background noise.
*   **Passphrase (Registration):** Using a phonetically rich phrase during multi-sample registration might help capture more voice characteristics. Authentication can be attempted with any phrase.
*   **Speaker Distinguishability:** This remains the hardest challenge. Even with enhancements, Resemblyzer may struggle with very similar voices. The observed gap between genuine and imposter scores dictates the system's practical security.


## Features

*   **Voice Registration (Multi-Sample):** Records user voice multiple times for a fixed passphrase, averages embeddings, and saves the robust voiceprint to an SQLite database.
*   **Voice Authentication:** Compares new voice input (generic prompt) against a stored voiceprint.
*   **Offline Capability:** Works entirely offline *after* initial library installation and Resemblyzer model download.
*   **Basic CLI:** Simple command-line interface.
*   **Multi-user Support:** Managed via unique usernames in the database.
*   **Database Storage:** Uses SQLite (`voice_auth_resemblyzer.db`) for storing voiceprints (256-dim float64).
*   **Silence Detection:** Includes an RMS energy check (reading from file) to fail authentication on quiet input.
*   **Liveness Detection (Placeholder):** Includes a non-functional placeholder.
*   **Confidence Score:** Displays the calculated cosine similarity score.
*   **Similarity Visualization:** Displays a simple text-based bar showing the score relative to the threshold.

## Requirements

*   Python 3.7+
*   Libraries listed in `requirements.txt` (or install manually):
    *   `numpy`
    *   `scipy`
    *   `sounddevice`
    *   `wave`
    *   `resemblyzer`
    *   `librosa`
    *   `torch` (CPU version is sufficient)
    *   `torchaudio`
    *   `noisereduce` (*Optional*: Only needed if noise reduction is re-enabled in the code)
*   **PortAudio:** A cross-platform audio I/O library required by `sounddevice`.

## Installation

1.  **Clone or download the repository.**
2.  **Install PortAudio:** (See previous instructions for Linux/macOS/Windows).
3.  **Install Python dependencies:** Navigate to the project directory and run:
    ```bash
    pip install numpy scipy sounddevice wave resemblyzer librosa torch torchaudio # Add noisereduce if needed
    ```
    (Or use the updated `requirements.txt` if provided).
4.  **Download Resemblyzer Model (First Run Only):** The first time you run `register.py` or `authenticate.py`, `resemblyzer` downloads its model (requires internet). Subsequent runs are offline.

## Database

*   Uses SQLite file: `voice_auth_resemblyzer.db`.
*   Table: `users (username TEXT PRIMARY KEY, voiceprint BLOB, registration_date TEXT)`.
*   `voiceprint` stores the 256-dimension float64 NumPy embedding array as a BLOB.
*   Managed via `db_utils.py`.

## How to Run

1.  **Setup:** Install dependencies, ensure PortAudio is working. Run once with internet for model download. **Delete any older `.db` files** if switching between versions.
2.  **Run Registration:**
    ```bash
    python register.py
    ```
    *   Enter username. Record the **same fixed passphrase** clearly for each of the 3 requested samples. An averaged voiceprint is saved.
3.  **Run Authentication:**
    ```bash
    python authenticate.py
    ```
    *   Enter registered username. Speak clearly for the duration when prompted (using the registration passphrase recommended for consistency).
    *   System performs checks (liveness placeholder, energy check), compares embeddings, displays results.
4.  **Database & Recordings:** `voice_auth_resemblyzer.db` stores user data. `recordings/` contains WAV backups.

## Offline Capability Explained

Relies on the locally cached Resemblyzer model, standard Python libraries (`sqlite3`, `wave`, etc.), and audio processing libraries (`numpy`, `scipy`, `sounddevice`) that run locally. Internet needed only for initial setup.

## Preprocessing Steps

1.  **Audio Recording:** Captured via `sounddevice`, saved as WAV.
2.  **Energy Check (Authentication):** RMS calculated from the WAV file; fails if below threshold.
3.  **Resemblyzer `preprocess_wav`:** Handles reading audio, performs Voice Activity Detection (VAD), and ensures correct internal sample rate.
4.  **Resemblyzer `embed_utterance`:** Generates the speaker embedding from the preprocessed audio.
5.  **Averaging (Registration):** Embeddings from multiple samples are averaged.

