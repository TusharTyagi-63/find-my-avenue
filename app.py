import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import polyline
import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

load_dotenv()

os.environ.setdefault(
    "YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_FOLDER = os.path.join(BASE_DIR, "instance")
DATABASE_PATH = os.path.join(INSTANCE_FOLDER, "find_my_avenue.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

HAZARD_INFLUENCE_KM = 0.4
ROUTE_COLORS = ["#38bdf8", "#6366f1", "#94a3b8"]
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
SEVERITY_WEIGHTS = {"low": 1.0, "medium": 2.4, "high": 4.0}
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
CUSTOM_HAZARD_MODEL_PATH = os.getenv("ROAD_HAZARD_MODEL_PATH")
INCIDENT_SERVICE_PROFILES = {
    "accident": [
        "Simulated Ambulance Dispatch",
        "Simulated Police Control Room",
        "Simulated Hospital Desk",
    ],
    "road blockage": [
        "Simulated Police Control Room",
        "Simulated Road Maintenance Cell",
        "Simulated Ambulance Dispatch",
    ],
    "flooding": [
        "Simulated Disaster Response Cell",
        "Simulated Police Control Room",
        "Simulated Ambulance Dispatch",
    ],
    "fire": [
        "Simulated Fire Response Desk",
        "Simulated Ambulance Dispatch",
        "Simulated Police Control Room",
    ],
    "medical emergency": [
        "Simulated Ambulance Dispatch",
        "Simulated Hospital Desk",
        "Simulated Police Control Room",
    ],
}
ALERT_STATUS_BY_SEVERITY = {
    "low": "Monitoring and advisory sent",
    "medium": "Priority dispatch simulated",
    "high": "Critical dispatch simulated",
}

os.makedirs(INSTANCE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ORS_API_KEY = os.getenv("ORS_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_SMS_FROM_NUMBER = os.getenv("TWILIO_SMS_FROM_NUMBER")
TWILIO_VOICE_FROM_NUMBER = os.getenv("TWILIO_VOICE_FROM_NUMBER") or TWILIO_SMS_FROM_NUMBER

model = None
analysis_jobs = {}
analysis_lock = threading.Lock()


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
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
            row["name"]
            for row in connection.execute("PRAGMA table_info(hazard_records)").fetchall()
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
            row["name"]
            for row in connection.execute("PRAGMA table_info(emergency_alerts)").fetchall()
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


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize_hazard_type(label):
    return label.replace("_", " ").strip().lower()


def allowed_video_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def resize_frame(frame, max_width=720):
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_height = max(1, int(height * scale))

    return cv2.resize(frame, (max_width, new_height))


def get_model():
    global model

    if not CUSTOM_HAZARD_MODEL_PATH:
        return None

    if model is None:
        from ultralytics import YOLO

        model = YOLO(CUSTOM_HAZARD_MODEL_PATH)

    return model


def set_analysis_job(job_id, **values):
    with analysis_lock:
        analysis_jobs.setdefault(job_id, {}).update(values)


def get_analysis_job(job_id):
    with analysis_lock:
        job = analysis_jobs.get(job_id)
        return dict(job) if job is not None else None


def insert_hazard_record(
    location_label,
    coordinates,
    summary,
    notes,
    source_location,
):
    source_location = (source_location or location_label).strip()
    connection = get_db_connection()

    with connection:
        cursor = connection.execute(
            """
            INSERT INTO hazard_records (
                created_at,
                location_label,
                latitude,
                longitude,
                hazard_type,
                severity,
                confidence,
                risk_score,
                hazard_count,
                frames_analyzed,
                notes,
                source_location,
                severity_breakdown,
                hazard_breakdown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    row = connection.execute(
        "SELECT * FROM hazard_records WHERE id = ?",
        (record_id,),
    ).fetchone()
    connection.close()

    return serialize_hazard_row(row)


def serialize_hazard_row(row):
    data = dict(row)
    data["severity_breakdown"] = json.loads(data["severity_breakdown"] or "{}")
    data["hazard_breakdown"] = json.loads(data["hazard_breakdown"] or "{}")
    data["source_location"] = (data.get("source_location") or data["location_label"]).strip()
    data["confidence"] = round(float(data["confidence"]), 2)
    data["latitude"] = round(float(data["latitude"]), 6)
    data["longitude"] = round(float(data["longitude"]), 6)
    return data


def get_recent_hazards(limit=100):
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
    return [serialize_hazard_row(row) for row in rows]


def get_hazard_by_id(hazard_id):
    connection = get_db_connection()
    row = connection.execute(
        "SELECT * FROM hazard_records WHERE id = ?",
        (hazard_id,),
    ).fetchone()
    connection.close()
    return serialize_hazard_row(row) if row else None


def parse_recipient_numbers(raw_value):
    if not raw_value:
        return []

    recipients = []

    for part in re.split(r"[,\n;]+", raw_value):
        cleaned = re.sub(r"[^\d+]", "", part.strip())

        if not cleaned:
            continue

        if cleaned.startswith("00"):
            cleaned = f"+{cleaned[2:]}"

        if cleaned.startswith("+") and re.fullmatch(r"\+[1-9]\d{7,14}", cleaned):
            recipients.append(cleaned)
            continue

        if re.fullmatch(r"[1-9]\d{7,14}", cleaned):
            recipients.append(f"+{cleaned}")
            continue

        raise ValueError(
            f"Use full mobile numbers with country code, like +919876543210. Invalid value: {part.strip()}"
        )

    return list(dict.fromkeys(recipients))


def get_twilio_client():
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError(
            "Twilio is not configured. Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
        )

    try:
        from twilio.rest import Client
    except ImportError as exc:
        raise RuntimeError(
            "Twilio dependency is missing on the server. Redeploy after installing requirements."
        ) from exc

    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def build_alert_message_text(
    road_location,
    source_location,
    incident_type,
    severity,
    notes,
):
    message = (
        f"Find My Avenue alert: {severity.title()} {incident_type} reported near "
        f"{road_location}. Source: {source_location}."
    )

    if notes:
        message = f"{message} Notes: {notes[:120]}"

    return message


def build_call_twiml(alert_message):
    try:
        from twilio.twiml.voice_response import VoiceResponse
    except ImportError as exc:
        raise RuntimeError(
            "Twilio voice helper is missing on the server. Redeploy after installing requirements."
        ) from exc

    response = VoiceResponse()
    response.pause(length=1)
    response.say(
        "This is a Find My Avenue emergency test alert.",
        voice="alice",
    )
    response.say(alert_message, voice="alice")
    response.pause(length=1)
    response.say(
        "This was a manual test call from the project prototype.",
        voice="alice",
    )
    return str(response)


def build_emergency_services(incident_type, severity):
    services = []

    for index, service_name in enumerate(
        INCIDENT_SERVICE_PROFILES.get(
            incident_type,
            INCIDENT_SERVICE_PROFILES["accident"],
        ),
        start=1,
    ):
        services.append(
            {
                "service": service_name,
                "channel": f"Simulated API channel #{index}",
                "status": "notified",
                "priority": severity,
            }
        )

    return services


def deliver_real_notifications(
    alert_message,
    recipient_numbers,
    send_sms,
    send_call,
):
    client = get_twilio_client()
    results = []

    if send_sms and not TWILIO_SMS_FROM_NUMBER:
        raise RuntimeError(
            "Twilio SMS sender is missing. Add TWILIO_SMS_FROM_NUMBER."
        )

    if send_call and not TWILIO_VOICE_FROM_NUMBER:
        raise RuntimeError(
            "Twilio voice sender is missing. Add TWILIO_VOICE_FROM_NUMBER or TWILIO_SMS_FROM_NUMBER."
        )

    call_twiml = build_call_twiml(alert_message) if send_call else None

    for number in recipient_numbers:
        if send_sms:
            try:
                message = client.messages.create(
                    body=alert_message,
                    from_=TWILIO_SMS_FROM_NUMBER,
                    to=number,
                )
                results.append(
                    {
                        "channel": "sms",
                        "to": number,
                        "provider": "twilio",
                        "status": message.status or "queued",
                        "sid": message.sid,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "channel": "sms",
                        "to": number,
                        "provider": "twilio",
                        "status": "failed",
                        "error": str(exc)[:240],
                    }
                )

        if send_call:
            try:
                call = client.calls.create(
                    to=number,
                    from_=TWILIO_VOICE_FROM_NUMBER,
                    twiml=call_twiml,
                )
                results.append(
                    {
                        "channel": "call",
                        "to": number,
                        "provider": "twilio",
                        "status": call.status or "queued",
                        "sid": call.sid,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "channel": "call",
                        "to": number,
                        "provider": "twilio",
                        "status": "failed",
                        "error": str(exc)[:240],
                    }
                )

    return results


def summarize_notification_results(results, send_sms, send_call):
    if not send_sms and not send_call:
        return "Simulation only. No real SMS or call was requested."

    if not results:
        return "No real notifications were attempted."

    delivered = [
        item for item in results if item.get("status") not in {"failed", "canceled"}
    ]
    failed = [item for item in results if item.get("status") == "failed"]
    channels = []

    if send_sms:
        sms_count = sum(1 for item in delivered if item["channel"] == "sms")
        channels.append(f"SMS attempted: {sms_count}")

    if send_call:
        call_count = sum(1 for item in delivered if item["channel"] == "call")
        channels.append(f"Calls attempted: {call_count}")

    if failed:
        channels.append(f"Failures: {len(failed)}")

    return " | ".join(channels)


def serialize_emergency_row(row):
    data = dict(row)
    data["services_notified"] = json.loads(data["services_notified"] or "[]")
    data["recipient_numbers"] = json.loads(data.get("recipient_numbers") or "[]")
    data["requested_channels"] = json.loads(data.get("requested_channels") or "[]")
    data["notification_results"] = json.loads(
        data.get("notification_results") or "[]"
    )
    data["latitude"] = round(float(data["latitude"]), 6)
    data["longitude"] = round(float(data["longitude"]), 6)
    data["source_location"] = (
        data.get("source_location") or data["road_location"]
    ).strip()
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
):
    source_location = (source_location or road_location).strip()
    dispatch_status = ALERT_STATUS_BY_SEVERITY[severity]
    alert_message = build_alert_message_text(
        road_location,
        source_location,
        incident_type,
        severity,
        notes,
    )
    alert_message = f"{alert_message} {dispatch_status}."

    connection = get_db_connection()

    with connection:
        cursor = connection.execute(
            """
            INSERT INTO emergency_alerts (
                created_at,
                road_location,
                source_location,
                latitude,
                longitude,
                incident_type,
                severity,
                trigger_mode,
                dispatch_status,
                services_notified,
                emergency_contact_name,
                emergency_contact_phone,
                notes,
                linked_hazard_id,
                alert_message,
                recipient_numbers,
                requested_channels,
                provider,
                notification_results
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    row = connection.execute(
        "SELECT * FROM emergency_alerts WHERE id = ?",
        (alert_id,),
    ).fetchone()
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


def parse_location(text):
    if not text:
        return None

    text = text.strip()

    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]

        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None

    return geocode_location(text)


def geocode_location(place):
    if not place:
        return None

    if ORS_API_KEY:
        try:
            response = requests.get(
                "https://api.openrouteservice.org/geocode/search",
                params={"api_key": ORS_API_KEY, "text": place, "size": 1},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])

            if features:
                coords = features[0]["geometry"]["coordinates"]
                return coords[1], coords[0]
        except requests.RequestException:
            pass
        except (KeyError, IndexError, TypeError, ValueError):
            pass

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "find-my-avenue/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            return None

        return float(data[0]["lat"]), float(data[0]["lon"])
    except requests.RequestException:
        return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def extract_service_error(response, fallback):
    if response is None:
        return fallback

    try:
        payload = response.json()
    except ValueError:
        return fallback

    error = payload.get("error") or payload.get("message")

    if isinstance(error, dict):
        return str(error.get("message") or fallback)

    if isinstance(error, list):
        return "; ".join(str(item) for item in error) or fallback

    return str(error or fallback)


def build_route_results(data):
    routes = []

    for route_data in data.get("routes", []):
        decoded = polyline.decode(route_data["geometry"])
        summary = route_data.get("summary", {})

        routes.append(
            {
                "coords": decoded,
                "distance": round(summary.get("distance", 0) / 1000, 2),
                "duration": round(summary.get("duration", 0) / 60, 2),
            }
        )

    return routes


def get_routes(start, end):
    if not ORS_API_KEY:
        raise RuntimeError("OpenRouteService API key is missing.")

    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    base_body = {"coordinates": [[start[1], start[0]], [end[1], end[0]]]}
    attempts = [
        {
            **base_body,
            "alternative_routes": {"target_count": 3, "share_factor": 0.6},
        },
        base_body,
    ]
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    last_error = "Route service could not generate a route."

    for body in attempts:
        try:
            response = requests.post(url, json=body, headers=headers, timeout=30)
            response.raise_for_status()
            routes = build_route_results(response.json())

            if routes:
                return routes

            last_error = "No routes were returned for that trip."
        except requests.HTTPError as exc:
            last_error = extract_service_error(
                exc.response,
                "Route service rejected the request.",
            )
        except requests.RequestException:
            app.logger.exception("Route lookup request failed.")
            last_error = "Route service could not be reached."

    raise RuntimeError(last_error)


def severity_from_score(score):
    if score >= 105:
        return "high"
    if score >= 78:
        return "medium"
    return "low"


def detect_road_anomalies(frame):
    height, width = frame.shape[:2]
    roi_top = int(height * 0.45)
    roi = frame[roi_top:, :]

    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gradient = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )

    threshold_value = int(max(35, min(110, cv2.mean(enhanced)[0] * 0.8)))
    _, dark_mask = cv2.threshold(
        enhanced,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV,
    )
    _, edge_mask = cv2.threshold(gradient, 18, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(
        dark_mask,
        cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1),
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2,
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    roi_area = roi.shape[0] * roi.shape[1]
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 180 or area > roi_area * 0.12:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = box_width / max(box_height, 1)

        if aspect_ratio < 0.35 or aspect_ratio > 4.5:
            continue

        fill_ratio = area / max(box_width * box_height, 1)

        if fill_ratio < 0.18 or fill_ratio > 0.95:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        center_bias = (y + (box_height / 2)) / max(roi.shape[0], 1)
        contour_mask = np.zeros_like(enhanced)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        darkness = 255 - cv2.mean(enhanced, mask=contour_mask)[0]
        edge_strength = cv2.mean(gradient, mask=contour_mask)[0]
        area_ratio = area / roi_area
        score = darkness * 0.55 + edge_strength * 0.3 + center_bias * 20
        score += min(25, area / 150)

        if score < 55:
            continue

        confidence = min(0.94, 0.30 + (score / 140) + (area_ratio * 6))
        severity = severity_from_score(score + (area_ratio * 1500))
        hazard_type = "pothole" if circularity > 0.18 else "road anomaly"

        detections.append(
            {
                "type": hazard_type,
                "severity": severity,
                "confidence": round(float(confidence), 2),
            }
        )

    detections.sort(
        key=lambda item: (
            SEVERITY_ORDER[item["severity"]],
            item["confidence"],
        ),
        reverse=True,
    )

    return detections[:5]


def detect_with_custom_model(frame, detector):
    height, width = frame.shape[:2]
    results = detector.predict(frame, imgsz=640, conf=0.25, verbose=False, device="cpu")
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            if isinstance(detector.names, dict):
                label = detector.names.get(cls, str(cls))
            else:
                label = detector.names[cls]

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / max(height * width, 1))
            severity_metric = confidence * 90 + area_ratio * 1200

            detections.append(
                {
                    "type": normalize_hazard_type(label),
                    "severity": severity_from_score(severity_metric),
                    "confidence": round(confidence, 2),
                }
            )

    return detections


def summarize_detections(all_detections, sampled_frames):
    severity_breakdown = {"low": 0, "medium": 0, "high": 0}
    hazard_breakdown = {}
    weighted_score = 0.0
    confidence_total = 0.0

    for detection in all_detections:
        severity_breakdown[detection["severity"]] += 1
        hazard_breakdown[detection["type"]] = hazard_breakdown.get(detection["type"], 0) + 1
        weighted_score += SEVERITY_WEIGHTS[detection["severity"]]
        confidence_total += detection["confidence"]

    hazard_count = len(all_detections)
    average_confidence = round(confidence_total / hazard_count, 2) if hazard_count else 0.0
    average_weight = weighted_score / max(sampled_frames, 1)
    risk_score = min(100, int(round(average_weight * 18)))

    if hazard_count == 0:
        status = "No major road anomalies detected"
        severity = "low"
        dominant_hazard = "road anomaly"
    elif risk_score < 30:
        status = "Minor surface anomalies detected"
        severity = "low"
        dominant_hazard = max(hazard_breakdown, key=hazard_breakdown.get)
    elif risk_score < 60:
        status = "Caution: uneven road conditions"
        severity = "medium"
        dominant_hazard = max(hazard_breakdown, key=hazard_breakdown.get)
    else:
        status = "High hazard exposure on this road segment"
        severity = "high"
        dominant_hazard = max(hazard_breakdown, key=hazard_breakdown.get)

    return {
        "hazard_count": hazard_count,
        "severity_breakdown": severity_breakdown,
        "hazard_breakdown": hazard_breakdown,
        "risk_score": risk_score,
        "status": status,
        "severity": severity,
        "dominant_hazard": dominant_hazard,
        "average_confidence": average_confidence,
        "frames_analyzed": sampled_frames,
    }


def analyze_saved_video(job_id, path, location_label, source_location, notes):
    cap = None

    try:
        coordinates = parse_location(location_label)
        source_location = (source_location or location_label).strip()

        if coordinates is None:
            raise ValueError("Could not understand that road location.")

        set_analysis_job(
            job_id,
            status="processing",
            message="Opening the uploaded road clip...",
        )

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise ValueError("The server could not read that video file.")

        detector = get_model()
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 24
        frame_interval = max(int(round(fps)), 12)
        max_frames = 18
        frame_index = 0
        sampled_frames = 0
        all_detections = []

        set_analysis_job(
            job_id,
            status="processing",
            message="Scanning the road surface for potholes and anomalies...",
        )

        while sampled_frames < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if not ret:
                break

            sampled_frames += 1
            frame_index += frame_interval
            frame = resize_frame(frame)

            if detector is None:
                detections = detect_road_anomalies(frame)
            else:
                detections = detect_with_custom_model(frame, detector)

            all_detections.extend(detections)

            if sampled_frames % 4 == 0:
                set_analysis_job(
                    job_id,
                    status="processing",
                    message=f"Analyzed {sampled_frames} frames so far...",
                )

        if sampled_frames == 0:
            raise ValueError("No readable frames were found in the uploaded video.")

        summary = summarize_detections(all_detections, sampled_frames)
        record = None

        if summary["hazard_count"] > 0:
            record = insert_hazard_record(
                location_label,
                coordinates,
                summary,
                notes,
                source_location,
            )

        set_analysis_job(
            job_id,
            status="completed",
            message="Analysis complete.",
            result={
                **summary,
                "location": {
                    "label": location_label,
                    "source_label": source_location,
                    "latitude": round(coordinates[0], 6),
                    "longitude": round(coordinates[1], 6),
                },
                "record": record,
            },
        )
    except ValueError as exc:
        set_analysis_job(job_id, status="failed", error=str(exc))
    except Exception as exc:
        app.logger.exception("Video analysis failed.")
        message = str(exc).strip() or exc.__class__.__name__
        set_analysis_job(
            job_id,
            status="failed",
            error=f"Server error during video analysis: {message[:240]}",
        )
    finally:
        if cap is not None:
            cap.release()

        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                app.logger.warning("Could not remove uploaded file: %s", path)


def project_point_to_xy(point, reference_lat):
    lat, lon = point
    x = math.radians(lon) * 6371.0088 * math.cos(math.radians(reference_lat))
    y = math.radians(lat) * 6371.0088
    return x, y


def point_to_segment_distance_km(point, start, end):
    reference_lat = (point[0] + start[0] + end[0]) / 3
    point_x, point_y = project_point_to_xy(point, reference_lat)
    start_x, start_y = project_point_to_xy(start, reference_lat)
    end_x, end_y = project_point_to_xy(end, reference_lat)

    dx = end_x - start_x
    dy = end_y - start_y

    if dx == 0 and dy == 0:
        return math.hypot(point_x - start_x, point_y - start_y)

    t = ((point_x - start_x) * dx + (point_y - start_y) * dy) / ((dx * dx) + (dy * dy))
    t = max(0.0, min(1.0, t))
    nearest_x = start_x + (t * dx)
    nearest_y = start_y + (t * dy)

    return math.hypot(point_x - nearest_x, point_y - nearest_y)


def simplify_route_coords(coords, max_points=90):
    if len(coords) <= max_points:
        return coords

    step = max(1, len(coords) // max_points)
    simplified = coords[::step]

    if simplified[-1] != coords[-1]:
        simplified.append(coords[-1])

    return simplified


def score_route_against_hazards(route_coords, hazards):
    simplified = simplify_route_coords(route_coords)

    if len(simplified) < 2:
        return {
            "safety_score": 100.0,
            "hazard_count": 0,
            "safety_label": "Safer route",
            "nearby_hazards": [],
            "penalty": 0.0,
        }

    impacts = []
    penalty = 0.0

    for hazard in hazards:
        hazard_point = (hazard["latitude"], hazard["longitude"])
        distance = min(
            point_to_segment_distance_km(hazard_point, start, end)
            for start, end in zip(simplified[:-1], simplified[1:])
        )

        if distance > HAZARD_INFLUENCE_KM:
            continue

        proximity = 1 - (distance / HAZARD_INFLUENCE_KM)
        severity_factor = {"low": 0.55, "medium": 1.0, "high": 1.45}[hazard["severity"]]
        risk_factor = max(0.45, hazard["risk_score"] / 100)
        impact = 9.0 * severity_factor * (0.45 + proximity) * risk_factor
        penalty += impact

        impacts.append(
            {
                "id": hazard["id"],
                "location_label": hazard["location_label"],
                "severity": hazard["severity"],
                "distance_km": round(distance, 2),
                "risk_score": hazard["risk_score"],
                "hazard_type": hazard["hazard_type"],
            }
        )

    impacts.sort(key=lambda item: item["distance_km"])
    safety_score = max(5, round(100 - min(85, penalty * 4), 1))

    if safety_score >= 75:
        safety_label = "Safer route"
    elif safety_score >= 50:
        safety_label = "Use caution"
    else:
        safety_label = "Avoid if possible"

    return {
        "safety_score": safety_score,
        "hazard_count": len(impacts),
        "safety_label": safety_label,
        "nearby_hazards": impacts[:5],
        "penalty": round(penalty, 2),
    }


def enrich_routes_with_hazards(routes):
    hazards = get_recent_hazards(limit=200)

    for route in routes:
        safety = score_route_against_hazards(route["coords"], hazards)
        route.update(safety)
        route["recommended_score"] = round(
            route["duration"] + ((100 - route["safety_score"]) / 6),
            2,
        )

    ranking = sorted(
        range(len(routes)),
        key=lambda index: routes[index]["recommended_score"],
    )

    for rank, route_index in enumerate(ranking):
        routes[route_index]["route_rank"] = rank + 1
        routes[route_index]["recommended"] = rank == 0
        routes[route_index]["color"] = ROUTE_COLORS[min(rank, len(ROUTE_COLORS) - 1)]

    return routes


@app.route("/api/traffic/tile/<int:z>/<int:x>/<int:y>.png")
def traffic_tile(z, x, y):
    if not TOMTOM_API_KEY:
        return ("Traffic service is not configured.", 503)

    try:
        response = requests.get(
            f"https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png",
            params={"key": TOMTOM_API_KEY, "tileSize": 256},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        app.logger.exception("Traffic tile request failed for %s/%s/%s", z, x, y)
        return ("Traffic tile unavailable.", 502)

    return Response(
        response.content,
        mimetype=response.headers.get("Content-Type", "image/png"),
        headers={"Cache-Control": response.headers.get("Cache-Control", "no-store")},
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/hazard")
def hazard():
    return render_template("hazard.html")


@app.route("/emergency")
def emergency():
    return render_template("emergency.html")


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Video is too large. Upload a clip under 50 MB."}), 413


@app.route("/api/hazards", methods=["GET"])
def hazards_api():
    limit = request.args.get("limit", default=100, type=int)
    if limit is None:
        limit = 100
    limit = max(1, min(limit, 250))
    return jsonify({"hazards": get_recent_hazards(limit=limit)})


@app.route("/api/emergency", methods=["GET", "POST"])
def emergency_api():
    if request.method == "GET":
        limit = request.args.get("limit", default=20, type=int)
        if limit is None:
            limit = 20
        limit = max(1, min(limit, 100))
        return jsonify({"alerts": get_recent_emergency_alerts(limit=limit)})

    data = request.get_json(silent=True) or {}
    road_location = (data.get("road_location") or "").strip()
    source_location = (data.get("source_location") or "").strip() or road_location
    incident_type = (data.get("incident_type") or "").strip().lower()
    severity = (data.get("severity") or "").strip().lower()
    trigger_mode = (data.get("trigger_mode") or "manual report").strip()
    notes = (data.get("notes") or "").strip()
    emergency_contact_name = (data.get("emergency_contact_name") or "").strip()
    emergency_contact_phone = (data.get("emergency_contact_phone") or "").strip()
    raw_recipient_numbers = data.get("recipient_numbers") or ""
    send_sms = bool(data.get("send_sms"))
    send_call = bool(data.get("send_call"))
    linked_hazard_id = data.get("linked_hazard_id")

    if not road_location:
        return jsonify({"error": "Add the emergency location first."}), 400

    if incident_type not in INCIDENT_SERVICE_PROFILES:
        return jsonify({"error": "Choose a supported incident type."}), 400

    if severity not in ALERT_STATUS_BY_SEVERITY:
        return jsonify({"error": "Choose a valid severity."}), 400

    coordinates = parse_location(road_location)

    if coordinates is None:
        return jsonify({"error": "Could not understand that emergency location."}), 400

    hazard = None
    normalized_hazard_id = None
    recipient_numbers = []

    if linked_hazard_id not in (None, "", "null"):
        try:
            normalized_hazard_id = int(linked_hazard_id)
        except (TypeError, ValueError):
            return jsonify({"error": "Linked hazard id is invalid."}), 400

        hazard = get_hazard_by_id(normalized_hazard_id)

        if hazard is None:
            return jsonify({"error": "The selected hazard report was not found."}), 404

    if send_sms or send_call:
        try:
            recipient_numbers = parse_recipient_numbers(raw_recipient_numbers)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not recipient_numbers:
            return jsonify(
                {
                    "error": "Add at least one test mobile number to send a real SMS or call."
                }
            ), 400

    services_notified = build_emergency_services(incident_type, severity)
    requested_channels = []

    if send_sms:
        requested_channels.append("sms")
    if send_call:
        requested_channels.append("call")

    provider = "simulation"
    notification_results = []

    if requested_channels:
        try:
            notification_results = deliver_real_notifications(
                build_alert_message_text(
                    road_location,
                    source_location,
                    incident_type,
                    severity,
                    notes,
                ),
                recipient_numbers,
                send_sms,
                send_call,
            )
            provider = "twilio"
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400

    alert = insert_emergency_alert(
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
        normalized_hazard_id,
        recipient_numbers,
        requested_channels,
        provider,
        notification_results,
    )

    return jsonify(
        {
            "message": "Emergency alert created. Dispatch simulation was saved and any requested real notifications were attempted.",
            "alert": alert,
            "linked_hazard": hazard,
            "delivery_summary": summarize_notification_results(
                notification_results,
                send_sms,
                send_call,
            ),
        }
    ), 201


@app.route("/api/route", methods=["POST"])
def route_api():
    data = request.get_json(silent=True) or {}
    start = parse_location(data.get("start"))
    end = parse_location(data.get("end"))

    if start is None or end is None:
        return jsonify({"error": "Enter valid start and end locations."}), 400

    try:
        routes = get_routes(start, end)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except (KeyError, TypeError, ValueError):
        app.logger.exception("Unexpected route lookup error.")
        return jsonify({"error": "Could not generate routes right now."}), 500

    if not routes:
        return jsonify({"error": "No routes were found for that trip."}), 404

    return jsonify({"routes": enrich_routes_with_hazards(routes)})


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    location_label = (request.form.get("location") or "").strip()
    source_location = (request.form.get("source_location") or "").strip()
    notes = (request.form.get("notes") or "").strip()

    if file is None or not file.filename:
        return jsonify({"error": "Choose a road video before analyzing."}), 400

    if not allowed_video_file(file.filename):
        return jsonify(
            {"error": "Upload an mp4, mov, avi, mkv, webm, or m4v video file."}
        ), 400

    if not location_label:
        return jsonify(
            {"error": "Add the road location so the result can be saved to the hazard map."}
        ), 400

    filename = secure_filename(file.filename)
    extension = os.path.splitext(filename)[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{extension}")

    try:
        file.save(path)
        job_id = uuid.uuid4().hex
        set_analysis_job(
            job_id,
            status="processing",
            message="Upload complete. Analysis started.",
        )
        thread = threading.Thread(
            target=analyze_saved_video,
            args=(job_id, path, location_label, source_location, notes),
            daemon=True,
        )
        thread.start()
        return jsonify(
            {
                "job_id": job_id,
                "status": "processing",
                "message": "Upload complete. Analysis started.",
            }
        ), 202
    except OSError:
        app.logger.exception("Could not save uploaded video.")
        return jsonify({"error": "The server could not save that video file."}), 500


@app.route("/analyze_video/<job_id>", methods=["GET"])
def analyze_video_status(job_id):
    job = get_analysis_job(job_id)

    if job is None:
        return jsonify({"error": "Analysis job not found."}), 404

    return jsonify(job)


init_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
