import os
import requests
import polyline
import cv2
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

os.environ["YOLO_CONFIG_DIR"] = "/tmp"

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ORS_API_KEY = os.getenv("ORS_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

model = None


def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
    return model


def parse_location(text):

    if "," in text:
        lat, lon = text.split(",")
        return float(lat), float(lon)

    return geocode_location(text)


def geocode_location(place):

    url = f"https://api.openrouteservice.org/geocode/search?api_key={ORS_API_KEY}&text={place}&size=1"

    try:
        res = requests.get(url)
        data = res.json()

        coords = data["features"][0]["geometry"]["coordinates"]

        return coords[1], coords[0]

    except:
        return None


def get_traffic(lat, lon):

    try:

        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={TOMTOM_API_KEY}"

        res = requests.get(url)
        data = res.json()

        current = data["flowSegmentData"]["currentSpeed"]
        free = data["flowSegmentData"]["freeFlowSpeed"]

        congestion = max(0, min(100, int((1 - current/free) * 100)))

        return congestion

    except:
        return 0


def get_routes(start, end):

    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    body = {
        "coordinates":[
            [start[1],start[0]],
            [end[1],end[0]]
        ],
        "alternative_routes":{
            "target_count":3,
            "share_factor":0.6
        }
    }

    headers={
        "Authorization":ORS_API_KEY,
        "Content-Type":"application/json"
    }

    res=requests.post(url,json=body,headers=headers)

    data=res.json()

    routes=[]

    for r in data["routes"]:

        decoded=polyline.decode(r["geometry"])

        segments=[]

        step=max(1,len(decoded)//25)

        for i in range(0,len(decoded)-step,step):

            lat=decoded[i][0]
            lon=decoded[i][1]

            traffic=get_traffic(lat,lon)

            if traffic<30:
                color="green"
            elif traffic<60:
                color="yellow"
            else:
                color="red"

            segments.append({
                "coords":[decoded[i],decoded[i+step]],
                "traffic":traffic,
                "color":color
            })

        summary=r["summary"]

        routes.append({
            "segments":segments,
            "distance":round(summary["distance"]/1000,2),
            "duration":round(summary["duration"]/60,2)
        })

    return routes


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/hazard")
def hazard():
    return render_template("hazard.html")


@app.route("/api/route",methods=["POST"])
def route_api():

    data=request.json

    start=parse_location(data["start"])
    end=parse_location(data["end"])

    routes=get_routes(start,end)

    return jsonify({"routes":routes})


@app.route("/analyze_video",methods=["POST"])
def analyze_video():

    file=request.files["video"]

    path=os.path.join(UPLOAD_FOLDER,file.filename)

    file.save(path)

    cap=cv2.VideoCapture(path)

    model=get_model()

    hazards=0
    frame_count=0

    vehicle_classes=["car","truck","bus","motorcycle"]

    while True:

        ret,frame=cap.read()

        if not ret:
            break

        frame_count+=1

        if frame_count%10!=0:
            continue

        results=model(frame)

        for r in results:
            for box in r.boxes:

                cls=int(box.cls[0])

                label=model.names[cls]

                if label in vehicle_classes:
                    hazards+=1

    cap.release()

    risk_score=min(100,hazards*2)

    if risk_score<30:
        status="Safe Route"
    elif risk_score<60:
        status="Moderate Risk"
    else:
        status="High Risk Route"

    return jsonify({
        "hazards":hazards,
        "risk_score":risk_score,
        "status":status
    })


if __name__=="__main__":

    port=int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0",port=port)