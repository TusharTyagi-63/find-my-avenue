import logging
import math
from datetime import datetime, timezone

import polyline
import requests

from config import HAZARD_INFLUENCE_KM, ORS_API_KEY, ROUTE_COLORS, ROUTE_MODES, ROUTE_CACHE_TTL_SECONDS
from db import get_recent_hazards
from services.cache_store import cache_get, cache_set, route_cache
from services.http_clients import get_http_session


logger = logging.getLogger(__name__)


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
    for route_data in data.get("routes", []):
        decoded = polyline.decode(route_data["geometry"])
        summary = route_data.get("summary", {})
        routes.append(
            {
                "coords": decoded,
                "distance": round(summary.get("distance", 0) / 1000, 2),
                "duration": round(summary.get("duration", 0) / 60, 2),
            }
        )
    return routes


def normalize_route_mode(mode):
    if not mode:
        return "drive"
    normalized = str(mode).strip().lower()
    legacy_aliases = {
        "ride": "bicycle",
        "cycle": "bicycle",
        "motorbike": "bike",
        "motorcycle": "bike",
        "scooter": "bike",
    }
    normalized = legacy_aliases.get(normalized, normalized)
    return normalized if normalized in ROUTE_MODES else None


def get_routes(start, end, mode="drive"):
    if not ORS_API_KEY:
        raise RuntimeError("OpenRouteService API key is missing.")
    cache_key = (round(float(start[0]), 5), round(float(start[1]), 5), round(float(end[0]), 5), round(float(end[1]), 5), mode)
    cached = cache_get(route_cache, cache_key)
    if cached is not None:
        return cached
    route_mode = ROUTE_MODES[mode]
    session = get_http_session()
    url = f"https://api.openrouteservice.org/v2/directions/{route_mode['profile']}"
    base_body = {"coordinates": [[start[1], start[0]], [end[1], end[0]]]}
    attempts = [{**base_body, "alternative_routes": {"target_count": 3, "share_factor": 0.6}}, base_body]
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    last_error = "Route service could not generate a route."
    for body in attempts:
        try:
            response = session.post(url, json=body, headers=headers, timeout=30)
            response.raise_for_status()
            routes = build_route_results(response.json())
            if routes:
                cache_set(route_cache, cache_key, routes, ROUTE_CACHE_TTL_SECONDS)
                return routes
            last_error = "No routes were returned for that trip."
        except requests.HTTPError as exc:
            last_error = extract_service_error(exc.response, "Route service rejected the request.")
        except requests.RequestException:
            logger.exception("Route lookup request failed.")
            last_error = "Route service could not be reached."
    raise RuntimeError(last_error)


def hazard_recency_factor(created_at):
    if not created_at:
        return 0.65
    try:
        reported_at = datetime.fromisoformat(str(created_at))
    except ValueError:
        return 0.65
    if reported_at.tzinfo is None:
        reported_at = reported_at.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (datetime.now(timezone.utc) - reported_at).total_seconds() / 3600)
    if age_hours <= 12:
        return 1.0
    if age_hours <= 72:
        return 0.88
    if age_hours <= 24 * 14:
        return 0.7
    if age_hours <= 24 * 60:
        return 0.48
    return 0.3


def project_point_to_xy(point, reference_lat):
    lat, lon = point
    x = math.radians(lon) * 6371.0088 * math.cos(math.radians(reference_lat))
    y = math.radians(lat) * 6371.0088
    return x, y


def point_to_segment_distance_km(point, start, end):
    reference_lat = (point[0] + start[0] + end[0]) / 3
    point_x, point_y = project_point_to_xy(point, reference_lat)
    start_x, start_y = project_point_to_xy(start, reference_lat)
    end_x, end_y = project_point_to_xy(end, reference_lat)
    dx = end_x - start_x
    dy = end_y - start_y
    if dx == 0 and dy == 0:
        return math.hypot(point_x - start_x, point_y - start_y)
    t = ((point_x - start_x) * dx + (point_y - start_y) * dy) / ((dx * dx) + (dy * dy))
    t = max(0.0, min(1.0, t))
    nearest_x = start_x + (t * dx)
    nearest_y = start_y + (t * dy)
    return math.hypot(point_x - nearest_x, point_y - nearest_y)


def simplify_route_coords(coords, max_points=90):
    if len(coords) <= max_points:
        return coords
    step = max(1, len(coords) // max_points)
    simplified = coords[::step]
    if simplified[-1] != coords[-1]:
        simplified.append(coords[-1])
    return simplified


def score_route_against_hazards(route_coords, hazards):
    simplified = simplify_route_coords(route_coords)
    if len(simplified) < 2:
        return {"safety_score": 100.0, "hazard_count": 0, "safety_label": "No saved hazards nearby", "nearby_hazards": [], "penalty": 0.0}
    impacts = []
    penalty = 0.0
    for hazard in hazards:
        hazard_point = (hazard["latitude"], hazard["longitude"])
        distance = min(point_to_segment_distance_km(hazard_point, start, end) for start, end in zip(simplified[:-1], simplified[1:]))
        if distance > HAZARD_INFLUENCE_KM:
            continue
        proximity = 1 - (distance / HAZARD_INFLUENCE_KM)
        severity_factor = {"low": 0.55, "medium": 1.0, "high": 1.45}[hazard["severity"]]
        risk_factor = max(0.45, hazard["risk_score"] / 100)
        recency_factor = hazard_recency_factor(hazard.get("created_at"))
        impact = 9.0 * severity_factor * (0.45 + proximity) * risk_factor * recency_factor
        penalty += impact
        impacts.append(
            {
                "id": hazard["id"],
                "location_label": hazard["location_label"],
                "severity": hazard["severity"],
                "distance_km": round(distance, 2),
                "risk_score": hazard["risk_score"],
                "hazard_type": hazard["hazard_type"],
                "recency_factor": round(recency_factor, 2),
            }
        )
    impacts.sort(key=lambda item: item["distance_km"])
    if not impacts:
        return {
            "safety_score": 100.0,
            "hazard_count": 0,
            "safety_label": "No saved hazards nearby" if hazards else "No saved hazard reports yet",
            "nearby_hazards": [],
            "penalty": 0.0,
        }
    safety_score = max(5, round(100 - min(85, penalty * 4), 1))
    if safety_score >= 75:
        safety_label = "Safer route"
    elif safety_score >= 50:
        safety_label = "Use caution"
    else:
        safety_label = "Avoid if possible"
    return {"safety_score": safety_score, "hazard_count": len(impacts), "safety_label": safety_label, "nearby_hazards": impacts[:5], "penalty": round(penalty, 2)}


def enrich_routes_with_hazards(routes):
    hazards = get_recent_hazards(limit=200)
    for route in routes:
        safety = score_route_against_hazards(route["coords"], hazards)
        route.update(safety)
        route["recommended_score"] = round(route["duration"] + ((100 - route["safety_score"]) / 6), 2)
    ranking = sorted(range(len(routes)), key=lambda index: routes[index]["recommended_score"])
    for rank, route_index in enumerate(ranking):
        routes[route_index]["route_rank"] = rank + 1
        routes[route_index]["recommended"] = rank == 0
        routes[route_index]["color"] = ROUTE_COLORS[min(rank, len(ROUTE_COLORS) - 1)]
    return routes
