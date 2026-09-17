import os

from flask import Flask

from config import UPLOAD_FOLDER
from db import init_db
from routes.api import api_bp
from routes.web import web_bp


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_db()
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
