from __future__ import annotations

import sqlite3
import json
from typing import Optional, List, Dict, Any

DB_PATH = "app.db"

# --- Initialization ---

def init_db() -> None:
    """Create tables if they don't exist (users, predictions)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection (thread-safe)."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# --- User helpers ---

def create_user(email: str, password_hash: str) -> int:
    """Insert a new user and return its ID (raises if email not unique)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, password_hash),
        )
        conn.commit()
        return cur.lastrowid

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Return user record as dict by email, or None if not found."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        return dict(row) if row else None

# --- Prediction helpers ---

def create_prediction(
    user_id: int, model_version: str, input_data: Dict[str, Any], output_data: Dict[str, Any]
) -> int:
    """Insert a prediction record and return its ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (user_id, model_version, input_data, output_data)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, model_version, json.dumps(input_data), json.dumps(output_data)),
        )
        conn.commit()
        return cur.lastrowid

def list_predictions(user_id: int) -> List[Dict[str, Any]]:
    """Return all predictions for a user, newest first."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, model_version, input_data, output_data, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        )
        return [dict(row) for row in cur.fetchall()]

def get_prediction(user_id: int, prediction_id: int) -> Optional[Dict[str, Any]]:
    """Return a prediction by ID for the given user, or None."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, model_version, input_data, output_data, created_at
            FROM predictions
            WHERE id = ? AND user_id = ?
            """,
            (prediction_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

def get_stats(user_id: int) -> Dict[str, Any]:
    """Return summary stats for predictions (total, by class, by model)."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Total count
        cur.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = ?", (user_id,))
        total = cur.fetchone()["total"] or 0

        # Counts by label and model version
        cur.execute("SELECT model_version, output_data FROM predictions WHERE user_id = ?", (user_id,))
        by_class = {"POSITIVE": 0, "NEGATIVE": 0}
        by_model_version: Dict[str, int] = {}

        for row in cur.fetchall():
            version = row["model_version"]
            by_model_version[version] = by_model_version.get(version, 0) + 1
            try:
                out = json.loads(row["output_data"])
                label = out.get("label")
                if label in by_class:
                    by_class[label] += 1
            except Exception:
                pass  # ignore parse errors

        return {"total": total, "by_class": by_class, "by_model_version": by_model_version}