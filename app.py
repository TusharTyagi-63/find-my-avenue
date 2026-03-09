import os
import requests
import polyline
import cv2
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ORS_API_KEY = os.getenv("ORS_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

# Lazy YOLO model
model = None


def get_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")
    return model


# ---------------------------
# Geocode location
# ---------------------------

def geocode_location(place):
    url = f"https://api.openrouteservice.org/geocode/search?api_key={ORS_API_KEY}&text={place}&size=1"
    res = requests.get(url)

    if res.status_code != 200:
        return None

    data = res.json()

    if len(data["features"]) == 0:
        return None

    coords = data["features"][0]["geometry"]["coordinates"]
    return coords[1], coords[0]


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

    res = requests.post(url, json=body, headers=headers)

    if res.status_code != 200:
        return None

    data = res.json()

    geometry = data["routes"][0]["geometry"]
    decoded = polyline.decode(geometry)

    summary = data["routes"][0]["summary"]

    return {
        "route": decoded,
        "distance": summary["distance"] / 1000,
        "duration": summary["duration"] / 60
    }


# ---------------------------
# Home
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# Route page
# ---------------------------

@app.route("/route")
def route_page():
    return render_template("dashboard.html")


# ---------------------------
# Hazard page
# ---------------------------

@app.route("/hazard")
def hazard_page():
    return render_template("hazard.html")


# ---------------------------
# Route API
# ---------------------------

@app.route("/get_route", methods=["POST"])
def route_api():

    data = request.json

    start_place = data.get("start")
    end_place = data.get("end")

    start = geocode_location(start_place)
    end = geocode_location(end_place)

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

    file = request.files["video"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    cap = cv2.VideoCapture(path)

    model = get_model()

    hazards = 0
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 5 != 0:
            continue

        results = model(frame)

        for r in results:
            for box in r.boxes:
                hazards += 1

    cap.release()

    risk_score = min(100, hazards // 10)

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
    app.run(debug=True)