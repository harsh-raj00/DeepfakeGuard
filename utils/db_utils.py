"""
=============================================================================
 Database Utilities — SQLite CRUD Operations
 Handles user storage, authentication history, and audit logging.
=============================================================================
"""

import sqlite3
import os
import sys
import bcrypt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_db_connection():
    """
    Get a connection to the SQLite database.
    Creates the database and tables if they don't exist.

    Returns:
        sqlite3.Connection: Database connection with Row factory.
    """
    db_path = config.DATABASE_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    return conn


def init_database():
    """
    Initialize the database schema.
    Creates users, login_history, and audit_logs tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Users table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            encoding_path TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            num_encodings INTEGER DEFAULT 0
        )
    """)

    # ── Login History table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            face_confidence REAL,
            liveness_blinks INTEGER,
            deepfake_confidence REAL,
            ip_address TEXT,
            alert_type TEXT,
            details TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # ── Audit Logs table ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            username TEXT,
            details TEXT,
            severity TEXT DEFAULT 'INFO'
        )
    """)

    conn.commit()
    conn.close()
    print("[INFO] Database initialized successfully.")


# ── User Operations ─────────────────────────────────────────────────────────

def create_user(username, email, password, num_encodings=0):
    """
    Create a new user in the database.

    Args:
        username (str): Unique username.
        email (str): User email.
        password (str): Plain-text password (will be hashed).
        num_encodings (int): Number of face encodings stored.

    Returns:
        dict: Result with 'success', 'message', and optional 'user_id'.
    """
    conn = get_db_connection()
    try:
        # Hash the password
        password_hash = bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        encoding_path = os.path.join(config.ENCODINGS_DIR, f"{username}.pkl")

        conn.execute("""
            INSERT INTO users (username, email, password_hash, encoding_path, num_encodings)
            VALUES (?, ?, ?, ?, ?)
        """, (username, email, password_hash, encoding_path, num_encodings))

        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        log_audit("USER_REGISTERED", username, f"User registered with {num_encodings} encodings.")

        return {"success": True, "message": "User registered successfully.", "user_id": user_id}

    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return {"success": False, "message": "Username already exists."}
        elif "email" in str(e):
            return {"success": False, "message": "Email already registered."}
        return {"success": False, "message": str(e)}
    finally:
        conn.close()


def get_user_by_username(username):
    """Get a user record by username."""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_email(email):
    """Get a user record by email."""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return dict(user) if user else None


def verify_password(username, password):
    """
    Verify a user's password.

    Args:
        username (str): Username to verify.
        password (str): Plain-text password to check.

    Returns:
        bool: True if password matches, False otherwise.
    """
    user = get_user_by_username(username)
    if user is None:
        return False
    return bcrypt.checkpw(
        password.encode('utf-8'),
        user['password_hash'].encode('utf-8')
    )


def get_all_users():
    """Get all registered users (without password hashes)."""
    conn = get_db_connection()
    users = conn.execute(
        "SELECT id, username, email, registered_at, is_active, num_encodings FROM users"
    ).fetchall()
    conn.close()
    return [dict(u) for u in users]


# ── Login History Operations ────────────────────────────────────────────────

def log_login_attempt(username, status, face_confidence=None,
                      liveness_blinks=None, deepfake_confidence=None,
                      ip_address=None, alert_type=None, details=None):
    """
    Record a login attempt in the history table.

    Args:
        username (str): Attempted username.
        status (str): 'SUCCESS', 'DENIED', 'ALERT'.
        face_confidence (float): Face recognition confidence.
        liveness_blinks (int): Number of blinks detected.
        deepfake_confidence (float): Deepfake real confidence.
        ip_address (str): Client IP address.
        alert_type (str): Type of alert if any.
        details (str): Additional details.
    """
    conn = get_db_connection()
    user = get_user_by_username(username)
    user_id = user['id'] if user else None

    conn.execute("""
        INSERT INTO login_history
        (user_id, username, status, face_confidence, liveness_blinks,
         deepfake_confidence, ip_address, alert_type, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, username, status, face_confidence, liveness_blinks,
          deepfake_confidence, ip_address, alert_type, details))

    conn.commit()
    conn.close()


def get_login_history(username=None, limit=50):
    """
    Get login history, optionally filtered by username.

    Args:
        username (str): Filter by username (None = all users).
        limit (int): Maximum number of records to return.

    Returns:
        list: List of login history records as dicts.
    """
    conn = get_db_connection()
    if username:
        rows = conn.execute(
            "SELECT * FROM login_history WHERE username = ? ORDER BY timestamp DESC LIMIT ?",
            (username, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM login_history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Audit Log Operations ───────────────────────────────────────────────────

def log_audit(event_type, username=None, details=None, severity="INFO"):
    """
    Log an audit event.

    Args:
        event_type (str): Type of event (e.g., 'USER_REGISTERED', 'DEEPFAKE_ALERT').
        username (str): Associated username.
        details (str): Event details.
        severity (str): 'INFO', 'WARNING', 'CRITICAL'.
    """
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO audit_logs (event_type, username, details, severity)
        VALUES (?, ?, ?, ?)
    """, (event_type, username, details, severity))
    conn.commit()
    conn.close()


def get_audit_logs(limit=100):
    """Get recent audit logs."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Initialize database on import ──────────────────────────────────────────
init_database()
