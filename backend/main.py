"""
Project Argos – FastAPI entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import guardian_ws, incidents_router

app = FastAPI(title="Project Argos – AI Campus Guardian")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(guardian_ws.router)
app.include_router(incidents_router.router)


@app.get("/")
def root():
    return {"project": "Project Argos", "status": "online"}
