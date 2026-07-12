"""
Project Argos – In-memory incident store
REST endpoints: GET /incidents  |  DELETE /incidents
"""
from fastapi import APIRouter
from typing import List, Dict, Any
import threading

router = APIRouter()

_lock = threading.Lock()
_incidents: List[Dict[str, Any]] = []


def add_incident(incident: Dict[str, Any]):
    """Called by the WebSocket handler to persist a new alert."""
    with _lock:
        _incidents.append(incident)
        # Keep only the last 200 incidents to avoid memory growth
        if len(_incidents) > 200:
            _incidents.pop(0)


@router.get("/incidents")
def get_incidents():
    with _lock:
        return list(reversed(_incidents))   # newest first


@router.delete("/incidents")
def clear_incidents():
    with _lock:
        _incidents.clear()
    return {"status": "cleared"}
