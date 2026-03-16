import os
import tempfile
import threading
import uuid

import cv2
import polyline
import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

load_dotenv()

os.environ.setdefault(
    "YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics")
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
ROUTE_COLORS = ["#2563eb", "#6366f1", "#94a3b8"]
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ORS_API_KEY = os.getenv("ORS_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

model = None
analysis_jobs = {}
analysis_lock = threading.Lock()


def get_model():
    global model
    if model is None:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
    return model


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


def set_analysis_job(job_id, **values):
    with analysis_lock:
        analysis_jobs.setdefault(job_id, {}).update(values)


def get_analysis_job(job_id):
    with analysis_lock:
        job = analysis_jobs.get(job_id)
        return dict(job) if job is not None else None


def build_risk_status(hazards, sampled_frames):
    average_vehicle_density = hazards / sampled_frames
    risk_score = min(100, int(round(average_vehicle_density * 20)))

    if risk_score < 30:
        status = "Safe Route"
    elif risk_score < 60:
        status = "Moderate Risk"
    else:
        status = "High Risk Route"

    return risk_score, status


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
    if not ORS_API_KEY:
        return None

    try:
        res = requests.get(
            "https://api.openrouteservice.org/geocode/search",
            params={"api_key": ORS_API_KEY, "text": place, "size": 1},
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        features = data.get("features", [])

        if not features:
            return None

        coords = features[0]["geometry"]["coordinates"]
        return coords[1], coords[0]
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

    for index, route_data in enumerate(data.get("routes", [])):
        decoded = polyline.decode(route_data["geometry"])
        summary = route_data.get("summary", {})

        routes.append(
            {
                "coords": decoded,
                "distance": round(summary.get("distance", 0) / 1000, 2),
                "duration": round(summary.get("duration", 0) / 60, 2),
                "color": ROUTE_COLORS[index % len(ROUTE_COLORS)],
            }
        )

    return routes


def get_routes(start, end):
    if not ORS_API_KEY:
        raise RuntimeError("OpenRouteService API key is missing.")

    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    base_body = {
        "coordinates": [
            [start[1], start[0]],
            [end[1], end[0]],
        ]
    }
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
            res = requests.post(url, json=body, headers=headers, timeout=30)
            res.raise_for_status()
            routes = build_route_results(res.json())

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


def analyze_saved_video(job_id, path):
    cap = None

    try:
        set_analysis_job(
            job_id,
            status="processing",
            message="Opening the uploaded video...",
        )

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise ValueError("The server could not read that video file.")

        detector = get_model()
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 24
        frame_interval = max(int(round(fps)), 12)
        max_frames = 12
        frame_index = 0
        sampled_frames = 0
        hazards = 0
        vehicle_classes = {"car", "truck", "bus", "motorcycle"}

        set_analysis_job(
            job_id,
            status="processing",
            message="Scanning the video for vehicles...",
        )

        while sampled_frames < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if not ret:
                break

            sampled_frames += 1
            frame_index += frame_interval
            frame = resize_frame(frame)
            results = detector.predict(
                frame, imgsz=640, conf=0.25, verbose=False, device="cpu"
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(box.cls[0])

                    if isinstance(detector.names, dict):
                        label = detector.names.get(cls, str(cls))
                    else:
                        label = detector.names[cls]

                    if label in vehicle_classes:
                        hazards += 1

        if sampled_frames == 0:
            raise ValueError("No readable frames were found in the uploaded video.")

        risk_score, status = build_risk_status(hazards, sampled_frames)

        set_analysis_job(
            job_id,
            status="completed",
            message="Analysis complete.",
            result={
                "hazards": hazards,
                "risk_score": risk_score,
                "status": status,
                "frames_analyzed": sampled_frames,
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


@app.route("/api/traffic/tile/<int:z>/<int:x>/<int:y>.png")
def traffic_tile(z, x, y):
    if not TOMTOM_API_KEY:
        return ("Traffic service is not configured.", 503)

    try:
        res = requests.get(
            f"https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png",
            params={"key": TOMTOM_API_KEY, "tileSize": 256},
            timeout=20,
        )
        res.raise_for_status()
    except requests.RequestException:
        app.logger.exception("Traffic tile request failed for %s/%s/%s", z, x, y)
        return ("Traffic tile unavailable.", 502)

    return Response(
        res.content,
        mimetype=res.headers.get("Content-Type", "image/png"),
        headers={"Cache-Control": res.headers.get("Cache-Control", "no-store")},
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


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Video is too large. Upload a clip under 50 MB."}), 413


@app.route("/api/route",methods=["POST"])
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

    return jsonify({"routes": routes})


@app.route("/analyze_video",methods=["POST"])
def analyze_video():
    file = request.files.get("video")

    if file is None or not file.filename:
        return jsonify({"error": "Choose a video file before analyzing."}), 400

    if not allowed_video_file(file.filename):
        return jsonify(
            {"error": "Upload an mp4, mov, avi, mkv, webm, or m4v video file."}
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
            args=(job_id, path),
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
        return jsonify(
            {"error": "The server could not save that video file."}
        ), 500


@app.route("/analyze_video/<job_id>", methods=["GET"])
def analyze_video_status(job_id):
    job = get_analysis_job(job_id)

    if job is None:
        return jsonify({"error": "Analysis job not found."}), 404

    return jsonify(job)


if __name__=="__main__":

    port=int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0",port=port)
