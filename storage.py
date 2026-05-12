import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DB_PATH = Path(__file__).resolve().parent / "age_detection.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_label TEXT,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_session(summary: Dict[str, object], source_type: str, source_label: Optional[str] = None) -> int:
    init_db()
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO sessions (source_type, source_label, summary_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                source_type,
                source_label or "",
                json.dumps(summary),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_sessions(limit: int = 50) -> List[Dict[str, object]]:
    init_db()
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, source_type, source_label, summary_json, created_at
            FROM sessions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out: List[Dict[str, object]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "source_type": r["source_type"],
                    "source_label": r["source_label"],
                    "summary": json.loads(r["summary_json"]),
                    "created_at": r["created_at"],
                }
            )
        return out
    finally:
        conn.close()


def get_session(session_id: int) -> Optional[Dict[str, object]]:
    init_db()
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT id, source_type, source_label, summary_json, created_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "source_type": row["source_type"],
            "source_label": row["source_label"],
            "summary": json.loads(row["summary_json"]),
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def aggregate_overview(limit: int = 200) -> Dict[str, object]:
    rows = list_sessions(limit=limit)
    if not rows:
        return {
            "sessions": 0,
            "avg_estimated_age": 0.0,
            "avg_faces_per_frame": 0.0,
            "total_frames": 0,
            "total_tracks": 0,
            "top_bucket": None,
        }
    total_age = 0.0
    total_faces_per_frame = 0.0
    total_frames = 0
    total_tracks = 0
    bucket_totals: Dict[str, int] = {}
    for row in rows:
        summary = row["summary"]
        total_age += float(summary.get("estimated_avg_age", 0.0))
        total_faces_per_frame += float(summary.get("avg_faces_per_frame", 0.0))
        total_frames += int(summary.get("frames_processed", 0))
        total_tracks += int(summary.get("unique_people_tracks", 0))
        buckets = summary.get("bucket_distribution", {}) or {}
        for k, v in buckets.items():
            bucket_totals[k] = bucket_totals.get(k, 0) + int(v)
    top_bucket = max(bucket_totals.items(), key=lambda kv: kv[1])[0] if bucket_totals else None
    n = len(rows)
    return {
        "sessions": n,
        "avg_estimated_age": round(total_age / n, 2),
        "avg_faces_per_frame": round(total_faces_per_frame / n, 3),
        "total_frames": total_frames,
        "total_tracks": total_tracks,
        "top_bucket": top_bucket,
    }
