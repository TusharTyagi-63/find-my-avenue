import os
import threading
import uuid

import requests
from flask import Blueprint, Response, jsonify, request
from werkzeug.utils import secure_filename

from config import (
    ALERT_STATUS_BY_SEVERITY,
    INCIDENT_SERVICE_PROFILES,
    ROUTE_MODES,
    TOMTOM_API_KEY,
    TRAFFIC_TILE_CACHE_SECONDS,
    UPLOAD_FOLDER,
)
from db import (
    get_hazard_by_id,
    get_recent_emergency_alerts,
    get_recent_hazards,
    insert_emergency_alert,
)
from services.emergency import (
    build_alert_message_text,
    build_dispatch_status,
    build_emergency_services,
    deliver_real_notifications,
    parse_recipient_numbers,
    summarize_notification_results,
)
from services.geocoding import normalize_place_query, parse_coordinate_pair, parse_location, search_location_candidates
from services.hazards import allowed_video_file, analyze_saved_video, get_analysis_job, set_analysis_job
from services.http_clients import get_http_session
from services.routing import enrich_routes_with_hazards, get_routes, normalize_route_mode

api_bp = Blueprint("api", __name__)


@api_bp.route("/api/traffic/tile/<int:z>/<int:x>/<int:y>.png")
def traffic_tile(z, x, y):
    if not TOMTOM_API_KEY:
        return ("Traffic service is not configured.", 503)
    session = get_http_session()
    try:
        response = session.get(
            f"https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png",
            params={"key": TOMTOM_API_KEY, "tileSize": 256},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ("Traffic tile unavailable.", 502)
    return Response(
        response.content,
        mimetype=response.headers.get("Content-Type", "image/png"),
        headers={"Cache-Control": f"public, max-age={TRAFFIC_TILE_CACHE_SECONDS}"},
    )


@api_bp.route("/api/hazards", methods=["GET"])
def hazards_api():
    limit = request.args.get("limit", default=100, type=int)
    if limit is None:
        limit = 100
    limit = max(1, min(limit, 250))
    return jsonify({"hazards": get_recent_hazards(limit=limit)})


@api_bp.route("/api/location-search", methods=["GET"])
def location_search_api():
    query = request.args.get("q", default="", type=str)
    limit = request.args.get("limit", default=5, type=int)
    focus = parse_coordinate_pair(request.args.get("focus", default="", type=str))
    limit = max(1, min(limit or 5, 8))
    if len(normalize_place_query(query)) < 3:
        return jsonify({"suggestions": []})
    suggestions = search_location_candidates(query, limit=limit, focus=focus)
    return jsonify(
        {
            "suggestions": [
                {"label": item["label"], "latitude": item["latitude"], "longitude": item["longitude"]}
                for item in suggestions
            ]
        }
    )


@api_bp.route("/api/emergency", methods=["GET", "POST"])
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
            return jsonify({"error": "Add at least one test mobile number to send a real SMS or call."}), 400

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
                build_alert_message_text(road_location, source_location, incident_type, severity, notes),
                recipient_numbers,
                send_sms,
                send_call,
            )
            provider = "twilio"
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400

    dispatch_status = build_dispatch_status(severity)
    alert_message = build_alert_message_text(road_location, source_location, incident_type, severity, notes)
    alert_message = f"{alert_message} {dispatch_status}."
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
        alert_message,
        dispatch_status,
    )
    return (
        jsonify(
            {
                "message": "Emergency alert created. Dispatch simulation was saved and any requested real notifications were attempted.",
                "alert": alert,
                "linked_hazard": hazard,
                "delivery_summary": summarize_notification_results(notification_results, send_sms, send_call),
            }
        ),
        201,
    )


@api_bp.route("/api/route", methods=["POST"])
def route_api():
    data = request.get_json(silent=True) or {}
    raw_start = data.get("start")
    raw_end = data.get("end")
    start_focus = parse_coordinate_pair(raw_end)
    end_focus = parse_coordinate_pair(raw_start)
    start = parse_location(raw_start, focus=start_focus)
    end = parse_location(raw_end, focus=start if start is not None else end_focus)
    mode = normalize_route_mode(data.get("mode"))
    if start is None or end is None:
        return jsonify({"error": "Enter valid start and end locations."}), 400
    if mode is None:
        return jsonify({"error": "Choose a valid travel mode."}), 400
    try:
        routes = get_routes(start, end, mode=mode)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Could not generate routes right now."}), 500
    if not routes:
        return jsonify({"error": "No routes were found for that trip."}), 404
    return jsonify({"routes": enrich_routes_with_hazards(routes), "mode": mode, "mode_label": ROUTE_MODES[mode]["label"]})


@api_bp.route("/analyze_video", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    location_label = (request.form.get("location") or "").strip()
    source_location = (request.form.get("source_location") or "").strip()
    notes = (request.form.get("notes") or "").strip()
    if file is None or not file.filename:
        return jsonify({"error": "Choose a road video before analyzing."}), 400
    if not allowed_video_file(file.filename):
        return jsonify({"error": "Upload an mp4, mov, avi, mkv, webm, or m4v video file."}), 400
    if not location_label:
        return jsonify({"error": "Add the road location so the result can be saved to the hazard map."}), 400
    filename = secure_filename(file.filename)
    extension = os.path.splitext(filename)[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{extension}")
    try:
        file.save(path)
        job_id = uuid.uuid4().hex
        set_analysis_job(job_id, status="processing", message="Upload complete. Analysis started.")
        thread = threading.Thread(
            target=analyze_saved_video,
            args=(job_id, path, location_label, source_location, notes),
            daemon=True,
        )
        thread.start()
        return jsonify({"job_id": job_id, "status": "processing", "message": "Upload complete. Analysis started."}), 202
    except OSError:
        return jsonify({"error": "The server could not save that video file."}), 500


@api_bp.route("/analyze_video/<job_id>", methods=["GET"])
def analyze_video_status(job_id):
    job = get_analysis_job(job_id)
    if job is None:
        return jsonify({"error": "Analysis job not found."}), 404
    return jsonify(job)
