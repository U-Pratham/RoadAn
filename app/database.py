"""
RoadSight AI - SQLite Database Layer
Implements relational storage for road hazard analyses and individual detections.
"""

import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import DB_PATH

def get_db_connection() -> sqlite3.Connection:
    """Create a connection with foreign keys enabled and row dictionary access."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize relational database tables and performance indices."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_filename TEXT NOT NULL,
        upload_timestamp TEXT NOT NULL,
        image_path TEXT NOT NULL,
        annotated_path TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        road_category TEXT NOT NULL,
        overall_risk_score INTEGER NOT NULL,
        priority_category TEXT NOT NULL,
        inference_ms REAL NOT NULL,
        total_ms REAL NOT NULL,
        summary_explanation TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL,
        damage_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL,
        bbox_y1 REAL NOT NULL,
        bbox_x2 REAL NOT NULL,
        bbox_y2 REAL NOT NULL,
        individual_risk_score INTEGER NOT NULL,
        priority_category TEXT NOT NULL,
        explanation TEXT NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
    );
    """)

    # Performance indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_analysis ON detections(analysis_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(damage_class);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_priority ON analyses(priority_category);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(upload_timestamp);")

    conn.commit()
    conn.close()

def save_analysis(
    original_filename: str,
    image_path: str,
    annotated_path: str,
    latitude: Optional[float],
    longitude: Optional[float],
    road_category: str,
    overall_risk_score: int,
    priority_category: str,
    inference_ms: float,
    total_ms: float,
    summary_explanation: str,
    detections: List[Dict[str, Any]]
) -> int:
    """Atomic insertion of an analysis record and its child detections."""
    conn = get_db_connection()
    cur = conn.cursor()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cur.execute("""
        INSERT INTO analyses (
            original_filename, upload_timestamp, image_path, annotated_path,
            latitude, longitude, road_category, overall_risk_score,
            priority_category, inference_ms, total_ms, summary_explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            original_filename, now_iso, image_path, annotated_path,
            latitude, longitude, road_category, overall_risk_score,
            priority_category, inference_ms, total_ms, summary_explanation
        ))
        analysis_id = cur.lastrowid

        for d in detections:
            box = d.get("bounding_box", [0, 0, 0, 0])
            cur.execute("""
            INSERT INTO detections (
                analysis_id, damage_class, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                individual_risk_score, priority_category, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                d.get("damage_class", "unknown"),
                float(d.get("confidence", 0.0)),
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                int(d.get("risk_score", 0)),
                d.get("priority", "Low"),
                d.get("explanation", "")
            ))

        conn.commit()
        return analysis_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve full analysis record with associated detections."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, original_filename, upload_timestamp, image_path, annotated_path,
           latitude, longitude, road_category, overall_risk_score,
           priority_category, inference_ms, total_ms, summary_explanation
    FROM analyses
    WHERE id = ?
    """, (analysis_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return None

    analysis = dict(row)

    cur.execute("""
    SELECT id, damage_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
           individual_risk_score, priority_category, explanation
    FROM detections
    WHERE analysis_id = ?
    ORDER BY individual_risk_score DESC
    """, (analysis_id,))
    det_rows = cur.fetchall()

    detections = []
    for d in det_rows:
        detections.append({
            "id": d["id"],
            "damage_class": d["damage_class"],
            "confidence": d["confidence"],
            "bounding_box": [d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]],
            "individual_risk_score": d["individual_risk_score"],
            "priority_category": d["priority_category"],
            "explanation": d["explanation"]
        })

    analysis["detections"] = detections
    analysis["detection_count"] = len(detections)
    conn.close()
    return analysis

def get_analyses_history(
    limit: int = 50,
    offset: int = 0,
    priority: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve paginated inspection history with summary counts."""
    conn = get_db_connection()
    cur = conn.cursor()

    base_query = """
    SELECT a.id, a.original_filename, a.upload_timestamp, a.image_path,
           a.annotated_path, a.latitude, a.longitude, a.road_category,
           a.overall_risk_score, a.priority_category, a.inference_ms,
           a.total_ms, a.summary_explanation,
           COUNT(d.id) as detection_count
    FROM analyses a
    LEFT JOIN detections d ON a.id = d.analysis_id
    """
    params = []
    if priority:
        base_query += " WHERE a.priority_category = ? "
        params.append(priority)

    base_query += " GROUP BY a.id ORDER BY a.id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cur.execute(base_query, params)
    rows = cur.fetchall()

    count_query = "SELECT COUNT(*) FROM analyses"
    count_params = []
    if priority:
        count_query += " WHERE priority_category = ?"
        count_params.append(priority)
    cur.execute(count_query, count_params)
    total_records = cur.fetchone()[0]

    records = [dict(r) for r in rows]
    conn.close()

    return {
        "total": total_records,
        "limit": limit,
        "offset": offset,
        "items": records
    }

def get_system_statistics() -> Dict[str, Any]:
    """Calculate actual statistical aggregates from the database."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM analyses")
    total_analyses = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM detections")
    total_detections = cur.fetchone()[0]

    cur.execute("SELECT AVG(overall_risk_score), AVG(inference_ms) FROM analyses")
    avg_row = cur.fetchone()
    avg_risk = round(float(avg_row[0]), 2) if avg_row and avg_row[0] is not None else 0.0
    avg_latency = round(float(avg_row[1]), 2) if avg_row and avg_row[1] is not None else 0.0

    cur.execute("""
    SELECT damage_class, COUNT(*) as cnt
    FROM detections
    GROUP BY damage_class
    """)
    class_counts = {r["damage_class"]: r["cnt"] for r in cur.fetchall()}

    cur.execute("""
    SELECT priority_category, COUNT(*) as cnt
    FROM analyses
    GROUP BY priority_category
    """)
    priority_counts = {r["priority_category"]: r["cnt"] for r in cur.fetchall()}

    cur.execute("""
    SELECT road_category, COUNT(*) as cnt
    FROM analyses
    GROUP BY road_category
    """)
    road_counts = {r["road_category"]: r["cnt"] for r in cur.fetchall()}

    conn.close()

    return {
        "total_analyses_count": total_analyses,
        "total_detections_count": total_detections,
        "average_risk_score": avg_risk,
        "average_inference_latency_ms": avg_latency,
        "detections_by_class": class_counts,
        "analyses_by_priority": priority_counts,
        "analyses_by_road_category": road_counts
    }

# Ensure schema initialized
init_db()
