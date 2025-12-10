import sqlite3
import time
import uuid
from datetime import datetime

DB_NAME = "traffic_logs.db"


def init_db():
    """Create the logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # CLEAN SQL: No comments inside the string to prevent syntax errors
    c.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            request_id TEXT,
            prompt TEXT,
            response_source TEXT,
            latency REAL,
            tokens_saved INTEGER,
            model TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_request(
    request_id: str, prompt: str, source: str, latency: float, full_text: str
):
    print(f"DEBUG LOG TYPES:")

    print(f" - request_id: {type(request_id)} -> {request_id}")
    print(f" - prompt: {type(prompt)}")
    print(f" - source: {type(source)}")
    print(f" - latency: {type(latency)}")

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        log_entry_id = uuid.uuid4().hex

        print(
            f" - log_entry_id: {type(log_entry_id)} -> {log_entry_id}"
        )  # Check this too

        tokens = len(full_text) / 4 if full_text else 0

        c.execute(
            """
            INSERT INTO request_logs (id, request_id, prompt, response_source, latency, tokens_saved, model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(log_entry_id),
                str(request_id),
                prompt,
                source,
                latency,
                int(tokens),
                "qwen2.5:0.5b",
            ),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log request: {e}")
