import requests
from requests.adapters import HTTPAdapter

http_session = None
cv2_module = None
np_module = None


def get_http_session():
    global http_session
    if http_session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        http_session = session
    return http_session


def get_cv2():
    global cv2_module
    if cv2_module is None:
        import cv2

        cv2_module = cv2
    return cv2_module


def get_np():
    global np_module
    if np_module is None:
        import numpy as np

        np_module = np
    return np_module
