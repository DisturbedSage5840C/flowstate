"""FastAPI app: JSON API for the Flow State UI, plus the built frontend when ``frontend/dist`` exists.

Run:  uvicorn src.api.server:app --port 8000
Dev:  the Vite dev server (frontend/) proxies /api to this process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.service import ROOT, Store

app = FastAPI(title="Flow State API", version="1.0")
DIST = ROOT / "frontend" / "dist"


@lru_cache(maxsize=1)
def store() -> Store:
    return Store()


class AskBody(BaseModel):
    query: str


@app.get("/api/overview")
def overview(state: str | None = None):
    return store().overview(state)


@app.get("/api/stations")
def stations(year: int | None = None, state: str | None = None):
    return store().list_stations(year=year, state=state)


@app.get("/api/stations/{sid}")
def station(sid: str):
    try:
        return store().station(sid)
    except KeyError:
        raise HTTPException(404, f"unknown station {sid}")


@app.get("/api/compare")
def compare(a: str, b: str):
    try:
        return store().compare(a, b)
    except KeyError as e:
        raise HTTPException(404, f"unknown station {e.args[0]}")


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    return store().search(q)


@app.get("/api/cities")
def cities():
    return store().cities()


@app.get("/api/priorities")
def priorities(n: int = 8, state: str | None = None):
    return store().priorities(n=n, state=state)


@app.get("/api/recommendations")
def recommendations():
    return store().area_recommendations()


@app.get("/api/alerts")
def alerts(n: int = 8):
    return store().alerts(n=n)


@app.get("/api/scene")
def scene(lat: float, lon: float):
    return store().scene_near(lat, lon)


@app.post("/api/ask")
def ask(body: AskBody):
    return store().ask(body.query)


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
