# db_utils.py
import sqlite3
import numpy as np
from datetime import datetime
import os

DB_FILE = "voice_auth.db"
# Resemblyzer embeddings are typically float64
# If you encounter issues, double-check the dtype of your embeddings
EXPECTED_EMBEDDING_DTYPE = np.float64
# Determine the size in bytes for validation
# Use a dummy array to get the size reliably
_dummy_embedding = np.zeros(256, dtype=EXPECTED_EMBEDDING_DTYPE) # Assuming 256 dimensions from Resemblyzer
EXPECTED_BLOB_SIZE = len(_dummy_embedding.tobytes())


def connect_db():
    """Connects to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    # Optional: Enable foreign key constraints if you add related tables later
    # conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the database table if it doesn't exist."""
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    voiceprint BLOB NOT NULL,
                    registration_date TEXT NOT NULL
                )
            """)
            conn.commit()
            print(f"Database '{DB_FILE}' initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database Error during initialization: {e}")
        # Depending on severity, you might want to exit
        # exit(1)


def check_user_exists(username):
    """Checks if a username already exists in the database."""
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
            return cursor.fetchone() is not None
    except sqlite3.Error as e:
        print(f"Database Error checking user existence: {e}")
        return False # Assume not exists on error to be safe, or handle differently

def save_voiceprint_db(username, embedding):
    """Saves or replaces the voiceprint for a user in the database."""
    if not isinstance(embedding, np.ndarray) or embedding.dtype != EXPECTED_EMBEDDING_DTYPE:
        print(f"Error: Embedding must be a NumPy array with dtype {EXPECTED_EMBEDDING_DTYPE}")
        return False

    embedding_blob = embedding.tobytes()
    if len(embedding_blob) != EXPECTED_BLOB_SIZE:
         print(f"Error: Embedding blob size mismatch. Expected {EXPECTED_BLOB_SIZE}, got {len(embedding_blob)}. Check embedding dimension/dtype.")
         return False

    registration_time = datetime.now().isoformat()

    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            # Use INSERT OR REPLACE to handle both new users and updates (overwrites)
            cursor.execute("""
                INSERT OR REPLACE INTO users (username, voiceprint, registration_date)
                VALUES (?, ?, ?)
            """, (username, embedding_blob, registration_time))
            conn.commit()
            print(f"Voiceprint for '{username}' saved to database.")
            return True
    except sqlite3.Error as e:
        print(f"Database Error saving voiceprint for '{username}': {e}")
        return False

def load_voiceprint_db(username):
    """Loads the voiceprint for a user from the database."""
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT voiceprint FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()

            if result:
                voiceprint_blob = result[0]
                if len(voiceprint_blob) != EXPECTED_BLOB_SIZE:
                     print(f"Error: Loaded voiceprint blob size mismatch for '{username}'. Expected {EXPECTED_BLOB_SIZE}, got {len(voiceprint_blob)}. Data might be corrupted or saved with wrong format.")
                     return None

                # Convert blob back to NumPy array
                embedding = np.frombuffer(voiceprint_blob, dtype=EXPECTED_EMBEDDING_DTYPE)
                print(f"Voiceprint loaded for '{username}' from database.")
                return embedding
            else:
                print(f"Error: No voiceprint found in database for username '{username}'. Please register first.")
                return None
    except sqlite3.Error as e:
        print(f"Database Error loading voiceprint for '{username}': {e}")
        return None
    except Exception as e:
        # Catch potential numpy errors during conversion
        print(f"Error converting blob to numpy array for '{username}': {e}")
        return None

# Call init_db() when this module is imported to ensure DB exists
init_db()