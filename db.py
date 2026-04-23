import json
import os
import sqlite3
from datetime import datetime, timezone

from config import DATABASE_PATH, INSTANCE_FOLDER
from services.cache_store import cache_clear, cache_get, cache_set, hazard_cache
from config import HAZARD_CACHE_TTL_SECONDS


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    connection = get_db_connection()
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hazard_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                location_label TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                hazard_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_score INTEGER NOT NULL,
                hazard_count INTEGER NOT NULL,
                frames_analyzed INTEGER NOT NULL,
                notes TEXT,
                source_location TEXT NOT NULL DEFAULT '',
                severity_breakdown TEXT NOT NULL,
                hazard_breakdown TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(hazard_records)").fetchall()
        }
        if "source_location" not in columns:
            connection.execute(
                """
                ALTER TABLE hazard_records
                ADD COLUMN source_location TEXT NOT NULL DEFAULT ''
                """
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS emergency_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                road_location TEXT NOT NULL,
                source_location TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                trigger_mode TEXT NOT NULL,
                dispatch_status TEXT NOT NULL,
                services_notified TEXT NOT NULL,
                emergency_contact_name TEXT,
                emergency_contact_phone TEXT,
                notes TEXT,
                linked_hazard_id INTEGER,
                alert_message TEXT NOT NULL,
                recipient_numbers TEXT NOT NULL DEFAULT '[]',
                requested_channels TEXT NOT NULL DEFAULT '[]',
                provider TEXT NOT NULL DEFAULT 'simulation',
                notification_results TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        emergency_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(emergency_alerts)").fetchall()
        }
        emergency_column_defs = {
            "recipient_numbers": "TEXT NOT NULL DEFAULT '[]'",
            "requested_channels": "TEXT NOT NULL DEFAULT '[]'",
            "provider": "TEXT NOT NULL DEFAULT 'simulation'",
            "notification_results": "TEXT NOT NULL DEFAULT '[]'",
        }
        for column_name, column_def in emergency_column_defs.items():
            if column_name not in emergency_columns:
                connection.execute(
                    f"""
                    ALTER TABLE emergency_alerts
                    ADD COLUMN {column_name} {column_def}
                    """
                )
    connection.close()


def serialize_hazard_row(row):
    data = dict(row)
    data["severity_breakdown"] = json.loads(data["severity_breakdown"] or "{}")
    data["hazard_breakdown"] = json.loads(data["hazard_breakdown"] or "{}")
    data["top_hazards"] = [
        {"type": label, "count": count}
        for label, count in sorted(data["hazard_breakdown"].items(), key=lambda item: (-item[1], item[0]))[:3]
    ]
    data["source_location"] = (data.get("source_location") or data["location_label"]).strip()
    data["confidence"] = round(float(data["confidence"]), 2)
    data["latitude"] = round(float(data["latitude"]), 6)
    data["longitude"] = round(float(data["longitude"]), 6)
    return data


def get_recent_hazards(limit=100):
    cache_key = int(limit or 100)
    cached = cache_get(hazard_cache, cache_key)
    if cached is not None:
        return cached
    connection = get_db_connection()
    rows = connection.execute(
        """
        SELECT *
        FROM hazard_records
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()
    hazards = [serialize_hazard_row(row) for row in rows]
    cache_set(hazard_cache, cache_key, hazards, HAZARD_CACHE_TTL_SECONDS)
    return hazards


def get_hazard_by_id(hazard_id):
    connection = get_db_connection()
    row = connection.execute("SELECT * FROM hazard_records WHERE id = ?", (hazard_id,)).fetchone()
    connection.close()
    return serialize_hazard_row(row) if row else None


def insert_hazard_record(location_label, coordinates, summary, notes, source_location):
    source_location = (source_location or location_label).strip()
    connection = get_db_connection()
    with connection:
        cursor = connection.execute(
            """
            INSERT INTO hazard_records (
                created_at, location_label, latitude, longitude, hazard_type, severity,
                confidence, risk_score, hazard_count, frames_analyzed, notes, source_location,
                severity_breakdown, hazard_breakdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                location_label,
                coordinates[0],
                coordinates[1],
                summary["dominant_hazard"],
                summary["severity"],
                summary["average_confidence"],
                summary["risk_score"],
                summary["hazard_count"],
                summary["frames_analyzed"],
                notes or "",
                source_location,
                json.dumps(summary["severity_breakdown"]),
                json.dumps(summary["hazard_breakdown"]),
            ),
        )
        record_id = cursor.lastrowid
    row = connection.execute("SELECT * FROM hazard_records WHERE id = ?", (record_id,)).fetchone()
    connection.close()
    cache_clear(hazard_cache)
    return serialize_hazard_row(row)


def serialize_emergency_row(row):
    data = dict(row)
    data["services_notified"] = json.loads(data["services_notified"] or "[]")
    data["recipient_numbers"] = json.loads(data.get("recipient_numbers") or "[]")
    data["requested_channels"] = json.loads(data.get("requested_channels") or "[]")
    data["notification_results"] = json.loads(data.get("notification_results") or "[]")
    data["latitude"] = round(float(data["latitude"]), 6)
    data["longitude"] = round(float(data["longitude"]), 6)
    data["source_location"] = (data.get("source_location") or data["road_location"]).strip()
    return data


def insert_emergency_alert(
    road_location,
    source_location,
    coordinates,
    incident_type,
    severity,
    trigger_mode,
    services_notified,
    emergency_contact_name,
    emergency_contact_phone,
    notes,
    linked_hazard_id,
    recipient_numbers,
    requested_channels,
    provider,
    notification_results,
    alert_message,
    dispatch_status,
):
    source_location = (source_location or road_location).strip()
    connection = get_db_connection()
    with connection:
        cursor = connection.execute(
            """
            INSERT INTO emergency_alerts (
                created_at, road_location, source_location, latitude, longitude, incident_type,
                severity, trigger_mode, dispatch_status, services_notified,
                emergency_contact_name, emergency_contact_phone, notes, linked_hazard_id,
                alert_message, recipient_numbers, requested_channels, provider, notification_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                road_location,
                source_location,
                coordinates[0],
                coordinates[1],
                incident_type,
                severity,
                trigger_mode,
                dispatch_status,
                json.dumps(services_notified),
                emergency_contact_name or "",
                emergency_contact_phone or "",
                notes or "",
                linked_hazard_id,
                alert_message,
                json.dumps(recipient_numbers),
                json.dumps(requested_channels),
                provider,
                json.dumps(notification_results),
            ),
        )
        alert_id = cursor.lastrowid
    row = connection.execute("SELECT * FROM emergency_alerts WHERE id = ?", (alert_id,)).fetchone()
    connection.close()
    return serialize_emergency_row(row)


def get_recent_emergency_alerts(limit=20):
    connection = get_db_connection()
    rows = connection.execute(
        """
        SELECT *
        FROM emergency_alerts
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()
    return [serialize_emergency_row(row) for row in rows]
