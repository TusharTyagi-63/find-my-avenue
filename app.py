from flask import Flask, render_template, request, jsonify
import os
import requests
import polyline
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2

load_dotenv()

app = Flask(__name__)

ORS_API_KEY = os.getenv("ORS_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

# Load YOLO once
model = YOLO("yolov8n.pt")


# -------------------------
# PAGE ROUTES
# -------------------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/hazard")
def hazard_page():
    return render_template("hazard.html")


# -------------------------
# ROUTE API (REAL ROUTE TRAFFIC)
# -------------------------

@app.route("/api/route", methods=["POST"])
def api_route():

    data = request.json
    start = data.get("start")
    end = data.get("end")

    start_coords = parse_or_geocode(start)
    end_coords = parse_or_geocode(end)

    if not start_coords or not end_coords:
        return jsonify({"error": "Geocoding failed"}), 400

    routes = get_routes(start_coords, end_coords)

    processed = []

    for route in routes:

        # Decode polyline (lat, lon)
        decoded = polyline.decode(route["geometry"])

        # 🔥 Real route traffic score
        traffic_score = get_route_traffic_score(decoded)

        # Stable hazard score for dashboard
        hazard_score = 10

        # Weighted risk model
        final_risk = int(0.6 * traffic_score + 0.4 * hazard_score)

        processed.append({
            "coords": decoded,
            "distance": round(route["summary"]["distance"] / 1000, 2),
            "duration": round(route["summary"]["duration"] / 60, 2),
            "traffic": traffic_score,
            "hazard": hazard_score,
            "risk": final_risk
        })

    # Sort safest first
    processed.sort(key=lambda x: x["risk"])

    return jsonify({"routes": processed})


# -------------------------
# REAL ROUTE TRAFFIC LOGIC
# -------------------------

def get_route_traffic_score(coords):

    if not coords:
        return 0

    # Sample every 20th coordinate to avoid rate limits
    sample_points = coords[::20]
    traffic_values = []

    for lat, lon in sample_points:

        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={TOMTOM_API_KEY}"

        try:
            res = requests.get(url, timeout=3)
            data = res.json()

            free = data["flowSegmentData"]["freeFlowSpeed"]
            current = data["flowSegmentData"]["currentSpeed"]

            if free == 0:
                continue

            congestion = ((free - current) / free) * 100
            traffic_values.append(congestion)

        except:
            continue

    if not traffic_values:
        return 0

    return int(sum(traffic_values) / len(traffic_values))


# -------------------------
# HAZARD VIDEO API (Optimized)
# -------------------------

@app.route("/api/hazard", methods=["POST"])
def analyze_video():

    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    path = "temp.mp4"
    file.save(path)

    cap = cv2.VideoCapture(path)

    hazard_count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analyze only every 15th frame
        if frame_count % 15 != 0:
            continue

        results = model(frame, verbose=False)
        hazard_count += len(results[0].boxes)

    cap.release()

    density = min(int((hazard_count / 150) * 100), 100)

    return jsonify({
        "hazards_detected": hazard_count,
        "hazard_density": density
    })


# -------------------------
# UTILITY FUNCTIONS
# -------------------------

def parse_or_geocode(text):

    # If coordinates entered manually
    if "," in text:
        try:
            lat, lon = map(float, text.split(","))
            return [lon, lat]
        except:
            pass

    # Otherwise geocode normally
    url = "https://api.openrouteservice.org/geocode/search"
    headers = {"Authorization": ORS_API_KEY}
    params = {"text": text, "size": 1}

    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        return data["features"][0]["geometry"]["coordinates"]
    except:
        return None


def get_routes(start, end):

    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "coordinates": [start, end],
        "alternative_routes": {
            "target_count": 2,
            "share_factor": 0.6
        }
    }

    try:
        res = requests.post(url, json=body, headers=headers)
        return res.json().get("routes", [])
    except:
        return []


# -------------------------

if __name__ == "__main__":
    app.run(debug=True)