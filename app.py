import os
import requests
from flask import Flask,render_template,request,jsonify
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

app=Flask(__name__)

ORS_KEY=os.getenv("ORS_API_KEY")
TOMTOM_KEY=os.getenv("TOMTOM_API_KEY")

model=YOLO("yolov8n.pt")

@app.route("/")
def index():

    return render_template(
        "index.html",
        traffic_key=TOMTOM_KEY
    )

@app.route("/hazard")
def hazard():

    return render_template("hazard.html")

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")

@app.route("/route",methods=["POST"])
def route():

    data=request.json

    start=data["start"]
    end=data["end"]
    mode=data.get("mode","driving-car")

    start_lat,start_lon=map(float,start.split(","))
    end_lat,end_lon=map(float,end.split(","))

    coords=[
        [start_lon,start_lat],
        [end_lon,end_lat]
    ]

    url=f"https://api.openrouteservice.org/v2/directions/{mode}/geojson"

    body={
        "coordinates":coords,
        "alternative_routes":{
            "target_count":3,
            "weight_factor":1.4
        }
    }

    headers={
        "Authorization":ORS_KEY,
        "Content-Type":"application/json"
    }

    r=requests.post(url,json=body,headers=headers)

    if r.status_code!=200:
        return jsonify({"error":"route failed"}),500

    data=r.json()

    routes=[]

    for f in data["features"]:

        routes.append({

            "geometry":f["geometry"],

            "distance":f["properties"]["summary"]["distance"],

            "duration":f["properties"]["summary"]["duration"]

        })

    return jsonify({"routes":routes})

@app.route("/detect",methods=["POST"])
def detect():

    file=request.files["image"]

    path="static/uploads/"+file.filename

    file.save(path)

    results=model(path)

    return "Hazard detection completed"

if __name__=="__main__":

    app.run(debug=True)