import os
import tempfile
from dotenv import load_dotenv

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
ROUTE_MODES = {
    "drive": {"profile": "driving-car", "label": "Drive"},
    "bike": {"profile": "driving-car", "label": "Bike"},
    "bicycle": {"profile": "cycling-regular", "label": "Bicycle"},
    "walk": {"profile": "foot-walking", "label": "Walk"},
}
DEFAULT_COUNTRY_HINT = os.getenv("GEOCODE_COUNTRY_HINT", "India")
PLACE_RESULT_LIMIT = 6
SPECIFIC_PLACE_TAGS = {
    "amenity",
    "building",
    "education",
    "emergency",
    "healthcare",
    "historic",
    "landuse",
    "leisure",
    "office",
    "railway",
    "shop",
    "tourism",
    "university",
}
SPECIFIC_PLACE_TYPES = {
    "airport",
    "bank",
    "bus_station",
    "cafe",
    "clinic",
    "college",
    "hospital",
    "hostel",
    "hotel",
    "library",
    "mall",
    "marketplace",
    "museum",
    "park",
    "school",
    "station",
    "stadium",
    "supermarket",
    "temple",
    "train_station",
    "university",
}
BROAD_PLACE_TYPES = {
    "administrative",
    "city",
    "country",
    "county",
    "district",
    "hamlet",
    "locality",
    "neighbourhood",
    "region",
    "state",
    "suburb",
    "town",
    "village",
}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
SEVERITY_WEIGHTS = {"low": 1.0, "medium": 2.4, "high": 4.0}
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
CUSTOM_HAZARD_MODEL_PATH = os.getenv("ROAD_HAZARD_MODEL_PATH")
DEFAULT_OBJECT_MODEL_NAME = os.getenv("ROAD_OBJECT_MODEL_NAME", "yolov8n.pt")
OBJECT_DETECTION_ENABLED = os.getenv(
    "ROAD_OBJECT_DETECTION_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
OBJECT_HAZARD_RULES = {
    "person": {
        "type": "pedestrian obstruction",
        "base_score": 70,
        "min_area_ratio": 0.006,
        "min_bottom_bias": 0.52,
    },
    "bicycle": {
        "type": "cycle obstruction",
        "base_score": 56,
        "min_area_ratio": 0.012,
        "min_bottom_bias": 0.62,
    },
    "motorcycle": {
        "type": "two wheeler obstruction",
        "base_score": 60,
        "min_area_ratio": 0.014,
        "min_bottom_bias": 0.64,
    },
    "car": {
        "type": "vehicle obstruction",
        "base_score": 48,
        "min_area_ratio": 0.04,
        "min_bottom_bias": 0.7,
    },
    "bus": {
        "type": "heavy vehicle obstruction",
        "base_score": 58,
        "min_area_ratio": 0.03,
        "min_bottom_bias": 0.68,
    },
    "truck": {
        "type": "heavy vehicle obstruction",
        "base_score": 62,
        "min_area_ratio": 0.03,
        "min_bottom_bias": 0.68,
    },
    "dog": {
        "type": "stray animal",
        "base_score": 74,
        "min_area_ratio": 0.003,
        "min_bottom_bias": 0.46,
    },
    "cat": {
        "type": "stray animal",
        "base_score": 64,
        "min_area_ratio": 0.002,
        "min_bottom_bias": 0.46,
    },
    "cow": {
        "type": "stray animal",
        "base_score": 82,
        "min_area_ratio": 0.008,
        "min_bottom_bias": 0.48,
    },
    "horse": {
        "type": "stray animal",
        "base_score": 80,
        "min_area_ratio": 0.008,
        "min_bottom_bias": 0.48,
    },
    "sheep": {
        "type": "stray animal",
        "base_score": 68,
        "min_area_ratio": 0.003,
        "min_bottom_bias": 0.46,
    },
}
SURFACE_SOURCE_LABEL = "surface"
OBJECT_SOURCE_LABEL = "object"
MAX_SURFACE_DETECTIONS_PER_FRAME = 3
MAX_OBJECT_DETECTIONS_PER_FRAME = 2
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

LOCATION_CACHE_TTL_SECONDS = 15 * 60
ROUTE_CACHE_TTL_SECONDS = 5 * 60
HAZARD_CACHE_TTL_SECONDS = 20
ANALYSIS_JOB_TTL_SECONDS = 60 * 60
TRAFFIC_TILE_CACHE_SECONDS = 45

# Hazard analysis performance knobs.
# Speed: lower HAZARD_MAX_FRAMES, raise HAZARD_MIN_FRAME_INTERVAL, lower HAZARD_YOLO_IMGSZ,
# raise HAZARD_OBJECT_EVERY_N_FRAMES (skip object YOLO on some frames), set HAZARD_YOLO_DEVICE=cuda:0 if available.
HAZARD_MAX_FRAMES = max(6, int(os.getenv("HAZARD_MAX_FRAMES", "12")))
HAZARD_MIN_FRAME_INTERVAL = max(6, int(os.getenv("HAZARD_MIN_FRAME_INTERVAL", "12")))
HAZARD_RESIZE_WIDTH = max(480, int(os.getenv("HAZARD_RESIZE_WIDTH", "640")))
HAZARD_MIN_BRIGHTNESS_STD = max(
    2.0,
    float(os.getenv("HAZARD_MIN_BRIGHTNESS_STD", "10.0")),
)
HAZARD_MIN_FRAME_DIFF = max(
    0.1,
    float(os.getenv("HAZARD_MIN_FRAME_DIFF", "2.0")),
)
# Ultralytics: smaller imgsz is faster; 512 is a good speed/quality tradeoff on CPU/GPU.
HAZARD_YOLO_IMGSZ = max(320, min(1280, int(os.getenv("HAZARD_YOLO_IMGSZ", "512"))))
# Run COCO/object YOLO every N-th analyzed frame (1 = every frame). Surface/custom model still runs every frame.
HAZARD_OBJECT_EVERY_N_FRAMES = max(1, int(os.getenv("HAZARD_OBJECT_EVERY_N_FRAMES", "2")))
# Empty = auto (CUDA if available, else Apple MPS, else CPU). Examples: "cpu", "0", "cuda:0", "mps"
HAZARD_YOLO_DEVICE = os.getenv("HAZARD_YOLO_DEVICE", "").strip()
# FP16 on CUDA only; set HAZARD_YOLO_HALF=1 when using a GPU for a further speedup.
HAZARD_YOLO_HALF = os.getenv("HAZARD_YOLO_HALF", "0").strip().lower() in {"1", "true", "yes", "on"}

ORS_API_KEY = os.getenv("ORS_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_SMS_FROM_NUMBER = os.getenv("TWILIO_SMS_FROM_NUMBER")
TWILIO_VOICE_FROM_NUMBER = os.getenv("TWILIO_VOICE_FROM_NUMBER") or TWILIO_SMS_FROM_NUMBER
