from flask import Blueprint, jsonify, render_template

web_bp = Blueprint("web", __name__)


@web_bp.route("/")
def home():
    return render_template("index.html")


@web_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@web_bp.route("/hazard")
def hazard():
    return render_template("hazard.html")


@web_bp.route("/emergency")
def emergency():
    return render_template("emergency.html")


@web_bp.app_errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Video is too large. Upload a clip under 50 MB."}), 413
