import logging
import math
import os
import threading
from time import monotonic

from config import (
    ALLOWED_VIDEO_EXTENSIONS,
    ANALYSIS_JOB_TTL_SECONDS,
    CUSTOM_HAZARD_MODEL_PATH,
    DEFAULT_OBJECT_MODEL_NAME,
    MAX_OBJECT_DETECTIONS_PER_FRAME,
    MAX_SURFACE_DETECTIONS_PER_FRAME,
    OBJECT_DETECTION_ENABLED,
    OBJECT_HAZARD_RULES,
    OBJECT_SOURCE_LABEL,
    SEVERITY_ORDER,
    SEVERITY_WEIGHTS,
    SURFACE_SOURCE_LABEL,
)
from db import insert_hazard_record
from services.geocoding import parse_location
from services.http_clients import get_cv2, get_np


logger = logging.getLogger(__name__)
model = None
model_error = None
object_model = None
object_model_error = None
analysis_jobs = {}
analysis_lock = threading.Lock()
model_lock = threading.Lock()


def cleanup_analysis_jobs(now=None):
    now = monotonic() if now is None else now
    expired_ids = [job_id for job_id, payload in analysis_jobs.items() if now - payload.get("_updated_at", now) > ANALYSIS_JOB_TTL_SECONDS]
    for job_id in expired_ids:
        analysis_jobs.pop(job_id, None)


def set_analysis_job(job_id, **values):
    with analysis_lock:
        cleanup_analysis_jobs()
        analysis_jobs.setdefault(job_id, {}).update(values)
        analysis_jobs[job_id]["_updated_at"] = monotonic()


def get_analysis_job(job_id):
    with analysis_lock:
        cleanup_analysis_jobs()
        job = analysis_jobs.get(job_id)
        if job is None:
            return None
        data = dict(job)
        data.pop("_updated_at", None)
        return data


def normalize_hazard_type(label):
    return label.replace("_", " ").strip().lower()


def allowed_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def resize_frame(frame, max_width=720):
    cv2 = get_cv2()
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (max_width, new_height))


def get_model():
    global model, model_error
    if not CUSTOM_HAZARD_MODEL_PATH:
        return None
    if model is None:
        with model_lock:
            if model is None:
                if model_error:
                    return None
                try:
                    from ultralytics import YOLO

                    model = YOLO(CUSTOM_HAZARD_MODEL_PATH)
                except Exception as exc:
                    model_error = str(exc).strip() or exc.__class__.__name__
                    logger.warning("Custom hazard model could not be initialized: %s", model_error)
                    return None
    return model


def get_object_model():
    global object_model, object_model_error
    if not OBJECT_DETECTION_ENABLED:
        return None
    if object_model is not None:
        return object_model
    if object_model_error:
        return None
    with model_lock:
        if object_model is not None:
            return object_model
        if object_model_error:
            return None
        try:
            from ultralytics import YOLO

            object_model = YOLO(DEFAULT_OBJECT_MODEL_NAME)
        except Exception as exc:
            object_model_error = str(exc).strip() or exc.__class__.__name__
            logger.warning("Object detection model could not be initialized: %s", object_model_error)
            return None
    return object_model


def get_detection_engines(surface_detector, object_detector):
    engines = ["custom hazard model" if surface_detector is not None else "surface heuristic analysis"]
    if object_detector is not None:
        engines.append("YOLO object detection")
    elif OBJECT_DETECTION_ENABLED and object_model_error:
        engines.append("object detection unavailable")
    return engines


def severity_from_score(score):
    if score >= 105:
        return "high"
    if score >= 78:
        return "medium"
    return "low"


def sort_detections(detections, limit=None):
    ordered = sorted(detections, key=lambda item: (SEVERITY_ORDER[item["severity"]], item["confidence"]), reverse=True)
    return ordered if limit is None else ordered[:limit]


def infer_detection_source(label):
    normalized = normalize_hazard_type(label)
    object_keywords = ("person", "pedestrian", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "animal", "dog", "cat", "cow", "horse", "sheep", "obstruction")
    if any(keyword in normalized for keyword in object_keywords):
        return OBJECT_SOURCE_LABEL
    return SURFACE_SOURCE_LABEL


def classify_surface_hazard(circularity, aspect_ratio, fill_ratio):
    if circularity >= 0.22 and fill_ratio >= 0.34:
        return "pothole"
    if aspect_ratio >= 2.4 and fill_ratio <= 0.55:
        return "surface crack"
    return "road anomaly"


def detect_road_anomalies(frame):
    cv2 = get_cv2()
    np = get_np()
    height, width = frame.shape[:2]
    roi_top = int(height * 0.55)
    lateral_margin = int(width * 0.08)
    roi = frame[roi_top:, lateral_margin : max(width - lateral_margin, lateral_margin + 1)]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    threshold_value = int(max(35, min(110, cv2.mean(enhanced)[0] * 0.8)))
    _, dark_mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    _, edge_mask = cv2.threshold(gradient, 18, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(dark_mask, cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 260 or area > roi_area * 0.08:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = box_width / max(box_height, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 3.2:
            continue
        fill_ratio = area / max(box_width * box_height, 1)
        if fill_ratio < 0.24 or fill_ratio > 0.84:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.3 or solidity > 0.96:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        vertical_bias = (y + (box_height / 2)) / max(roi.shape[0], 1)
        lane_center = (x + (box_width / 2)) / max(roi.shape[1], 1)
        lane_bias = 1 - min(1.0, abs(lane_center - 0.5) * 2)
        contour_mask = np.zeros_like(enhanced)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        darkness = 255 - cv2.mean(enhanced, mask=contour_mask)[0]
        edge_strength = cv2.mean(gradient, mask=contour_mask)[0]
        area_ratio = area / roi_area
        score = darkness * 0.36 + edge_strength * 0.34 + vertical_bias * 18 + lane_bias * 14 + min(16, area / 220)
        if darkness < 40 or edge_strength < 16 or score < 78:
            continue
        confidence = min(0.9, 0.18 + (score / 180) + (area_ratio * 4.4))
        severity_metric = score + (area_ratio * 1200) + (lane_bias * 10)
        hazard_type = classify_surface_hazard(circularity, aspect_ratio, fill_ratio)
        if hazard_type == "road anomaly" and severity_metric < 92:
            continue
        detections.append({"type": hazard_type, "severity": severity_from_score(severity_metric), "confidence": round(float(confidence), 2), "source": SURFACE_SOURCE_LABEL})
    return sort_detections(detections, limit=MAX_SURFACE_DETECTIONS_PER_FRAME)


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
            label = detector.names.get(cls, str(cls)) if isinstance(detector.names, dict) else detector.names[cls]
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / max(height * width, 1))
            severity_metric = confidence * 90 + area_ratio * 1200
            detections.append({"type": normalize_hazard_type(label), "severity": severity_from_score(severity_metric), "confidence": round(confidence, 2), "source": infer_detection_source(label)})
    return sort_detections(detections, limit=8)


def detect_object_hazards(frame, detector):
    height, width = frame.shape[:2]
    results = detector.predict(frame, imgsz=640, conf=0.25, verbose=False, device="cpu")
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            label = normalize_hazard_type(detector.names.get(cls, str(cls)) if isinstance(detector.names, dict) else detector.names[cls])
            rule = OBJECT_HAZARD_RULES.get(label)
            if rule is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / max(height * width, 1))
            bottom_bias = max(0.0, min(1.0, y2 / max(height, 1)))
            lane_center = ((x1 + x2) / 2) / max(width, 1)
            lane_bias = 1 - min(1.0, abs(lane_center - 0.5) * 2)
            if bottom_bias < rule.get("min_bottom_bias", 0.45):
                continue
            if area_ratio < rule["min_area_ratio"] and bottom_bias < max(0.76, rule.get("min_bottom_bias", 0.45) + 0.08):
                continue
            if lane_bias < 0.2 and area_ratio < rule["min_area_ratio"] * 1.5:
                continue
            severity_metric = rule["base_score"] + (confidence * 22) + (area_ratio * 1350) + (bottom_bias * 16) + (lane_bias * 8)
            if severity_metric < 74:
                continue
            detections.append({"type": rule["type"], "severity": severity_from_score(severity_metric), "confidence": round(min(0.95, confidence + (area_ratio * 1.2)), 2), "source": OBJECT_SOURCE_LABEL})
    return sort_detections(detections, limit=MAX_OBJECT_DETECTIONS_PER_FRAME)


def summarize_detections(all_detections, sampled_frames, detection_engines=None):
    severity_breakdown = {"low": 0, "medium": 0, "high": 0}
    hazard_breakdown = {}
    source_breakdown = {SURFACE_SOURCE_LABEL: 0, OBJECT_SOURCE_LABEL: 0}
    weighted_score = 0.0
    confidence_total = 0.0
    for detection in all_detections:
        severity_breakdown[detection["severity"]] += 1
        hazard_breakdown[detection["type"]] = hazard_breakdown.get(detection["type"], 0) + 1
        source = detection.get("source") or SURFACE_SOURCE_LABEL
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
        weighted_score += SEVERITY_WEIGHTS[detection["severity"]]
        confidence_total += detection["confidence"]
    hazard_count = len(all_detections)
    average_confidence = round(confidence_total / hazard_count, 2) if hazard_count else 0.0
    average_weight = weighted_score / max(sampled_frames, 1)
    risk_score = min(100, int(round(average_weight * 18)))
    top_hazards = [{"type": label, "count": count} for label, count in sorted(hazard_breakdown.items(), key=lambda item: (-item[1], item[0]))[:3]]
    if hazard_count == 0:
        status = "No major surface damage or roadway obstacles detected"
        severity = "low"
        dominant_hazard = "road anomaly"
    else:
        dominant_hazard = max(hazard_breakdown, key=hazard_breakdown.get)
        object_count = source_breakdown.get(OBJECT_SOURCE_LABEL, 0)
        surface_count = source_breakdown.get(SURFACE_SOURCE_LABEL, 0)
        if object_count and surface_count:
            status, severity = ("Mixed surface and obstacle hazards detected", "low") if risk_score < 35 else (("Multiple surface and roadway object hazards detected", "medium") if risk_score < 65 else ("High surface damage and roadway obstacle exposure", "high"))
        elif object_count:
            status, severity = ("Minor roadway object hazards detected", "low") if risk_score < 35 else (("Caution: roadway obstacles detected", "medium") if risk_score < 65 else ("High roadway obstacle exposure", "high"))
        else:
            status, severity = ("Minor surface anomalies detected", "low") if risk_score < 30 else (("Caution: uneven road conditions", "medium") if risk_score < 60 else ("High hazard exposure on this road segment", "high"))
    return {
        "hazard_count": hazard_count,
        "surface_hazard_count": source_breakdown.get(SURFACE_SOURCE_LABEL, 0),
        "object_hazard_count": source_breakdown.get(OBJECT_SOURCE_LABEL, 0),
        "severity_breakdown": severity_breakdown,
        "hazard_breakdown": hazard_breakdown,
        "source_breakdown": source_breakdown,
        "risk_score": risk_score,
        "status": status,
        "severity": severity,
        "dominant_hazard": dominant_hazard,
        "top_hazards": top_hazards,
        "average_confidence": average_confidence,
        "frames_analyzed": sampled_frames,
        "detection_engines": detection_engines or [],
    }


def analyze_saved_video(job_id, path, location_label, source_location, notes):
    cv2 = get_cv2()
    cap = None
    try:
        coordinates = parse_location(location_label)
        source_location = (source_location or location_label).strip()
        if coordinates is None:
            raise ValueError("Could not understand that road location.")
        set_analysis_job(job_id, status="processing", message="Opening the uploaded road clip...")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("The server could not read that video file.")
        detector = get_model()
        object_detector = get_object_model()
        detection_engines = get_detection_engines(detector, object_detector)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 24
        frame_interval = max(int(round(fps)), 12)
        max_frames = 18
        frame_index = 0
        sampled_frames = 0
        all_detections = []
        set_analysis_job(job_id, status="processing", message="Scanning road surface damage and roadway obstacles...")
        while sampled_frames < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break
            sampled_frames += 1
            frame_index += frame_interval
            frame = resize_frame(frame)
            detections = detect_road_anomalies(frame) if detector is None else detect_with_custom_model(frame, detector)
            if object_detector is not None:
                detections.extend(detect_object_hazards(frame, object_detector))
            all_detections.extend(detections)
            if sampled_frames % 4 == 0:
                set_analysis_job(job_id, status="processing", message=f"Analyzed {sampled_frames} frames for surface damage and obstacles so far...")
        if sampled_frames == 0:
            raise ValueError("No readable frames were found in the uploaded video.")
        summary = summarize_detections(all_detections, sampled_frames, detection_engines=detection_engines)
        record = None
        if summary["hazard_count"] > 0:
            record = insert_hazard_record(location_label, coordinates, summary, notes, source_location)
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
                "surface_detection_status": "custom-model" if detector is not None else ("fallback-heuristic" if model_error else "heuristic"),
                "surface_detection_error": model_error if detector is None else "",
                "object_detection_status": "active" if object_detector is not None else ("unavailable" if OBJECT_DETECTION_ENABLED else "disabled"),
                "object_detection_error": object_model_error if object_detector is None else "",
                "record": record,
            },
        )
    except ValueError as exc:
        set_analysis_job(job_id, status="failed", error=str(exc))
    except Exception as exc:
        logger.exception("Video analysis failed.")
        message = str(exc).strip() or exc.__class__.__name__
        set_analysis_job(job_id, status="failed", error=f"Server error during video analysis: {message[:240]}")
    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Could not remove uploaded file: %s", path)
