"""
Screening History — lightweight SQLite-backed audit log.

Every screening result (single-doc or batch) is persisted with a
timestamp so border officers can review past screenings, spot
patterns, and export records for compliance.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screening_history.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screening_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT,
            ground_truth TEXT,
            validation_score REAL,
            tamper_score REAL,
            face_match_score REAL,
            risk_score REAL,
            verdict TEXT,
            notes TEXT,
            ela_score REAL,
            copy_move_score REAL,
            weights TEXT,
            threshold REAL
        )
    """)
    conn.commit()
    return conn


def log_screening(
    filename: str,
    validation_score: float,
    tamper_score: float,
    face_match_score: float,
    risk_score: float,
    verdict: str,
    notes: str = "",
    ground_truth: str = "",
    ela_score: float = 0.0,
    copy_move_score: float = 0.0,
    weights: Optional[dict] = None,
    threshold: float = 25.0,
):
    """Insert a single screening result into the audit log."""
    try:
        conn = _get_connection()
        conn.execute(
            """INSERT INTO screening_log
               (timestamp, filename, ground_truth, validation_score, tamper_score,
                face_match_score, risk_score, verdict, notes, ela_score,
                copy_move_score, weights, threshold)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                filename,
                ground_truth,
                validation_score,
                tamper_score,
                face_match_score,
                risk_score,
                verdict,
                notes,
                ela_score,
                copy_move_score,
                json.dumps(weights) if weights else "{}",
                threshold,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Failed to log screening result: %s", e)


def get_history(limit: int = 100) -> list[dict]:
    """Fetch the most recent screening results."""
    try:
        conn = _get_connection()
        cursor = conn.execute(
            "SELECT * FROM screening_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("Failed to read screening history: %s", e)
        return []


def clear_history():
    """Clear all screening history."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM screening_log")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Failed to clear screening history: %s", e)
