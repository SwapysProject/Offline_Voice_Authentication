# db_utils.py
import sqlite3
import numpy as np
from datetime import datetime
import os

DB_FILE = "voice_auth_resemblyzer.db" # Use a new DB file name
EXPECTED_EMBEDDING_DTYPE = np.float64 # Resemblyzer often outputs float64
EMBEDDING_DIM = 256 # Resemblyzer default

# Recalculate expected blob size
_dummy_embedding = np.zeros(EMBEDDING_DIM, dtype=EXPECTED_EMBEDDING_DTYPE)
EXPECTED_BLOB_SIZE = len(_dummy_embedding.tobytes())
print(f"DEBUG DB_UTILS: Expected embedding dim: {EMBEDDING_DIM}, dtype: {EXPECTED_EMBEDDING_DTYPE}, blob size: {EXPECTED_BLOB_SIZE}")

# --- Function Definition Added ---
def connect_db():
    """Connects to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    # Optional: Enable foreign key constraints if you add related tables later
    # conn.execute("PRAGMA foreign_keys = ON;")
    return conn
# --- End Function Definition Added ---

def init_db():
    """Initializes the database table if it doesn't exist."""
    try:
        # Now connect_db() is defined and can be called
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
        exit(1)

def check_user_exists(username):
    """Checks if a username already exists in the database."""
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
            return cursor.fetchone() is not None
    except sqlite3.Error as e:
        print(f"Database Error checking user existence: {e}")
        return False

def save_voiceprint_db(username, embedding):
    """Saves or replaces the voiceprint for a user in the database."""
    if not isinstance(embedding, np.ndarray): return False
    if embedding.dtype != EXPECTED_EMBEDDING_DTYPE:
        embedding = embedding.astype(EXPECTED_EMBEDDING_DTYPE)
    if embedding.ndim != 1 or embedding.shape[0] != EMBEDDING_DIM: return False
    embedding_blob = embedding.tobytes()
    if len(embedding_blob) != EXPECTED_BLOB_SIZE: return False
    registration_time = datetime.now().isoformat()
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO users (username, voiceprint, registration_date)
                VALUES (?, ?, ?)
            """, (username, embedding_blob, registration_time))
            conn.commit()
            print(f"Voiceprint for '{username}' saved to database '{DB_FILE}'.")
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
                if len(voiceprint_blob) != EXPECTED_BLOB_SIZE: return None
                embedding = np.frombuffer(voiceprint_blob, dtype=EXPECTED_EMBEDDING_DTYPE)
                if embedding.shape[0] != EMBEDDING_DIM: return None
                print(f"Voiceprint loaded for '{username}' from database.")
                return embedding
            else:
                print(f"Error: No voiceprint found for '{username}'.")
                return None
    except sqlite3.Error as e:
        print(f"Database Error loading voiceprint for '{username}': {e}")
        return None
    except Exception as e:
        print(f"Error converting blob to numpy array for '{username}': {e}")
        return None

# Initialize DB on import
init_db()