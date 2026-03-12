import os
import requests
import polyline
import cv2
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ultralytics import YOLO

# load environment variables
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ORS_API_KEY = os.getenv("ORS_API_KEY")

# global model variable
model = None


# ---------------------------
# Lazy load YOLO model
# ---------------------------
def get_model():
    global model
    if model is None:
        print("Loading YOLOv8 model...")
        model = YOLO("yolov8n")   # automatic download
    return model


# ---------------------------
# Parse location input
# ---------------------------
def parse_location(text):

    if not text:
        return None

    # coordinate format
    if "," in text:
        try:
            lat, lon = text.split(",")
            return float(lat), float(lon)
        except:
            return None

    # place name
    return geocode_location(text)


# ---------------------------
# Geocode location
# ---------------------------
def geocode_location(place):

    url = f"https://api.openrouteservice.org/geocode/search?api_key={ORS_API_KEY}&text={place}&size=1"

    try:
        res = requests.get(url)
        data = res.json()

        if len(data["features"]) == 0:
            return None

        coords = data["features"][0]["geometry"]["coordinates"]

        return coords[1], coords[0]

    except:
        return None


# ---------------------------
# Get route
# ---------------------------
def get_route(start, end):

    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    body = {
        "coordinates": [
            [start[1], start[0]],
            [end[1], end[0]]
        ]
    }

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    try:

        res = requests.post(url, json=body, headers=headers)

        if res.status_code != 200:
            return None

        data = res.json()

        geometry = data["routes"][0]["geometry"]
        decoded = polyline.decode(geometry)

        summary = data["routes"][0]["summary"]

        return {
            "coords": decoded,
            "distance": round(summary["distance"] / 1000, 2),
            "duration": round(summary["duration"] / 60, 2)
        }

    except:
        return None


# ---------------------------
# Pages
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/hazard")
def hazard():
    return render_template("hazard.html")


# ---------------------------
# Route API
# ---------------------------

@app.route("/api/route", methods=["POST"])
def route_api():

    data = request.json

    start_text = data.get("start")
    end_text = data.get("end")

    start = parse_location(start_text)
    end = parse_location(end_text)

    if not start or not end:
        return jsonify({"error": "Location not found"})

    route = get_route(start, end)

    if not route:
        return jsonify({"error": "Route generation failed"})

    return jsonify(route)


# ---------------------------
# Hazard detection
# ---------------------------

@app.route("/analyze_video", methods=["POST"])
def analyze_video():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"})

    file = request.files["video"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    cap = cv2.VideoCapture(path)

    model = get_model()

    hazards = 0
    frame_count = 0

    vehicle_classes = ["car", "truck", "bus", "motorcycle"]

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # analyze every 10th frame for speed
        if frame_count % 10 != 0:
            continue

        results = model(frame)

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                if label in vehicle_classes:
                    hazards += 1

    cap.release()

    risk_score = min(100, hazards * 2)

    if risk_score < 30:
        status = "Safe Route"
    elif risk_score < 60:
        status = "Moderate Risk"
    else:
        status = "High Risk Route"

    return jsonify({
        "hazards": hazards,
        "risk_score": risk_score,
        "status": status
    })


# ---------------------------
# Run server
# ---------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)