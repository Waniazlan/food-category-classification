import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path("outputs/predictions.db")
UPLOAD_DIR = Path("uploads/predictions")


def init_database(db_path=DATABASE_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uploaded_image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                prediction_datetime TEXT NOT NULL
            )
            """
        )


def safe_filename(filename):
    path = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "uploaded_image"
    suffix = path.suffix.lower() or ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"


def save_uploaded_image(filename, image_bytes, upload_dir=UPLOAD_DIR):
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    image_path = upload_dir / safe_filename(filename)
    image_path.write_bytes(image_bytes)
    return image_path


def save_prediction(uploaded_image_path, predicted_class, confidence_score, db_path=DATABASE_PATH):
    init_database(db_path)
    prediction_datetime = datetime.now().astimezone().isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                uploaded_image_path,
                predicted_class,
                confidence_score,
                prediction_datetime
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                str(uploaded_image_path),
                predicted_class,
                float(confidence_score),
                prediction_datetime,
            ),
        )
        return cursor.lastrowid


def fetch_recent_predictions(limit=10, db_path=DATABASE_PATH):
    init_database(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                uploaded_image_path,
                predicted_class,
                confidence_score,
                prediction_datetime
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def fetch_predictions(db_path=DATABASE_PATH):
    init_database(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id,
                uploaded_image_path,
                predicted_class,
                confidence_score,
                prediction_datetime
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
