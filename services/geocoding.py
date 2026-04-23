import math
import re

import requests

from config import (
    BROAD_PLACE_TYPES,
    DEFAULT_COUNTRY_HINT,
    ORS_API_KEY,
    PLACE_RESULT_LIMIT,
    SPECIFIC_PLACE_TAGS,
    SPECIFIC_PLACE_TYPES,
)
from services.cache_store import cache_get, cache_set, location_cache
from services.http_clients import get_http_session
from config import LOCATION_CACHE_TTL_SECONDS


def parse_coordinate_pair(text):
    if not text:
        return None
    normalized = str(text).strip()
    if "," not in normalized:
        return None
    parts = [part.strip() for part in normalized.split(",", 1)]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def normalize_place_query(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def build_place_query_variants(place):
    normalized = normalize_place_query(place)
    if not normalized:
        return []
    variants = [normalized]
    lower_text = normalized.lower()
    if DEFAULT_COUNTRY_HINT and DEFAULT_COUNTRY_HINT.lower() not in lower_text and "india" not in lower_text:
        variants.append(f"{normalized}, {DEFAULT_COUNTRY_HINT}")
    return variants


def distance_between_points_km(point_a, point_b):
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371.0088 * c


def score_location_candidate(query, candidate, focus=None):
    normalized_query = normalize_place_query(query).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized_query) if token]
    search_blob = " ".join(
        [
            candidate.get("label", ""),
            candidate.get("name", ""),
            candidate.get("display_name", ""),
            candidate.get("category", ""),
            candidate.get("type", ""),
            candidate.get("layer", ""),
        ]
    ).lower()
    matched_tokens = sum(1 for token in tokens if token in search_blob)
    coverage = matched_tokens / max(len(tokens), 1)
    score = coverage * 7
    if normalized_query and normalized_query in search_blob:
        score += 2.8
    if candidate.get("category") in SPECIFIC_PLACE_TAGS:
        score += 1.8
    if candidate.get("type") in SPECIFIC_PLACE_TYPES:
        score += 2.2
    if candidate.get("layer") in {"venue", "address", "street"}:
        score += 1.4
    if candidate.get("type") in BROAD_PLACE_TYPES and len(tokens) > 1:
        score -= 1.8
    if candidate.get("source") == "ors":
        score += float(candidate.get("confidence", 0)) * 1.6
    else:
        score += float(candidate.get("importance", 0)) * 1.8
    if focus is not None:
        distance_km = distance_between_points_km(focus, (candidate["latitude"], candidate["longitude"]))
        if distance_km <= 5:
            score += 4.6
        elif distance_km <= 25:
            score += 3.0
        elif distance_km <= 80:
            score += 1.6
        elif distance_km <= 200:
            score += 0.7
        elif len(tokens) > 1 and distance_km > 300:
            score -= 1.6
    return round(score, 4)


def search_ors_candidates(place, limit=PLACE_RESULT_LIMIT, focus=None):
    if not ORS_API_KEY:
        return []
    session = get_http_session()
    candidates = []
    for query in build_place_query_variants(place):
        try:
            params = {"api_key": ORS_API_KEY, "text": query, "size": limit}
            if focus is not None:
                params["focus.point.lat"] = focus[0]
                params["focus.point.lon"] = focus[1]
            response = session.get("https://api.openrouteservice.org/geocode/search", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            continue
        for feature in data.get("features", []):
            try:
                properties = feature.get("properties", {})
                coordinates = feature["geometry"]["coordinates"]
                candidate = {
                    "label": properties.get("label") or properties.get("name") or query,
                    "name": properties.get("name") or "",
                    "display_name": properties.get("label") or properties.get("name") or query,
                    "latitude": float(coordinates[1]),
                    "longitude": float(coordinates[0]),
                    "category": properties.get("category") or "",
                    "type": properties.get("type") or "",
                    "layer": properties.get("layer") or "",
                    "confidence": float(properties.get("confidence") or 0.0),
                    "source": "ors",
                }
                candidate["score"] = score_location_candidate(place, candidate, focus=focus)
                candidates.append(candidate)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        if candidates:
            break
    return candidates


def search_nominatim_candidates(place, limit=PLACE_RESULT_LIMIT, focus=None):
    session = get_http_session()
    candidates = []
    for query in build_place_query_variants(place):
        try:
            response = session.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query,
                    "format": "jsonv2",
                    "limit": limit,
                    "addressdetails": 1,
                    "namedetails": 1,
                    "extratags": 1,
                },
                headers={"User-Agent": "find-my-avenue/1.0"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            continue
        for item in data:
            try:
                candidate = {
                    "label": item.get("display_name") or query,
                    "name": item.get("name") or item.get("namedetails", {}).get("name") or "",
                    "display_name": item.get("display_name") or query,
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lon"]),
                    "category": item.get("class") or item.get("category") or "",
                    "type": item.get("type") or "",
                    "layer": "",
                    "importance": float(item.get("importance") or 0.0),
                    "source": "nominatim",
                }
                candidate["score"] = score_location_candidate(place, candidate, focus=focus)
                candidates.append(candidate)
            except (KeyError, TypeError, ValueError):
                continue
        if candidates:
            break
    return candidates


def dedupe_location_candidates(candidates, limit=PLACE_RESULT_LIMIT):
    deduped = []
    seen = set()
    for candidate in sorted(candidates, key=lambda item: item.get("score", 0), reverse=True):
        key = (round(candidate["latitude"], 5), round(candidate["longitude"], 5), candidate["label"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped


def normalize_focus_key(focus):
    if focus is None:
        return None
    return (round(float(focus[0]), 4), round(float(focus[1]), 4))


def search_location_candidates(place, limit=PLACE_RESULT_LIMIT, focus=None):
    normalized = normalize_place_query(place)
    if not normalized:
        return []
    cache_key = (normalized, int(limit), normalize_focus_key(focus))
    cached = cache_get(location_cache, cache_key)
    if cached is not None:
        return cached
    candidates = search_ors_candidates(normalized, limit=limit, focus=focus)
    candidates.extend(search_nominatim_candidates(normalized, limit=limit, focus=focus))
    deduped = dedupe_location_candidates(candidates, limit=limit)
    cache_set(location_cache, cache_key, deduped, LOCATION_CACHE_TTL_SECONDS)
    return deduped


def geocode_location(place, focus=None):
    candidates = search_location_candidates(place, limit=1, focus=focus)
    if not candidates:
        return None
    best_match = candidates[0]
    return best_match["latitude"], best_match["longitude"]


def parse_location(text, focus=None):
    if not text:
        return None
    text = text.strip()
    coordinate_pair = parse_coordinate_pair(text)
    if coordinate_pair is not None:
        return coordinate_pair
    return geocode_location(text, focus=focus)
