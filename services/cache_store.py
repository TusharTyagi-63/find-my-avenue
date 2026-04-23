import copy
import threading
from time import monotonic


cache_lock = threading.Lock()
location_cache = {}
route_cache = {}
hazard_cache = {}


def cache_get(cache_store, key):
    now = monotonic()
    with cache_lock:
        payload = cache_store.get(key)
        if not payload:
            return None
        if payload["expires_at"] <= now:
            cache_store.pop(key, None)
            return None
        return copy.deepcopy(payload["value"])


def cache_set(cache_store, key, value, ttl_seconds):
    with cache_lock:
        cache_store[key] = {
            "value": copy.deepcopy(value),
            "expires_at": monotonic() + ttl_seconds,
        }


def cache_clear(cache_store):
    with cache_lock:
        cache_store.clear()
