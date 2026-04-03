"""
AegisVision – DualShield Interface
FastAPI Backend with OpenEnv integration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from env import AegisVisionEnv, Action

app = FastAPI(
    title="AegisVision API",
    description="DualShield Interface – AI-powered women's digital safety platform",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = AegisVisionEnv()


# ---------- Models ----------

class StepRequest(BaseModel):
    action: str


class ProtectRequest(BaseModel):
    image_name: Optional[str] = "uploaded_image"


class CompareRequest(BaseModel):
    original_name: Optional[str] = "original"
    suspect_name: Optional[str] = "suspect"


# ---------- OpenEnv Endpoints ----------

@app.get("/reset", tags=["OpenEnv"])
async def reset():
    result = env.reset()
    return JSONResponse(content=result)


@app.get("/state", tags=["OpenEnv"])
async def state():
    result = env.state()
    return JSONResponse(content=result)


@app.post("/step/{action}", tags=["OpenEnv"])
async def step(action: str):
    valid_actions = [a.value for a in Action]
    if action.upper() not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action}'. Must be one of: {valid_actions}"
        )
    result = env.step(action.upper())
    return JSONResponse(content=result)


# ---------- Protection Endpoint ----------

@app.post("/protect", tags=["Shield"])
async def protect(request: ProtectRequest):
    result = env.protect_image(image_name=request.image_name or "image")
    return JSONResponse(content=result)


@app.post("/protect/upload", tags=["Shield"])
async def protect_upload(file: UploadFile = File(...)):
    result = env.protect_image(image_name=file.filename or "uploaded_image")
    return JSONResponse(content=result)


# ---------- Compare Endpoint ----------

@app.post("/compare", tags=["Shield"])
async def compare(request: CompareRequest):
    result = env.compare_images(
        original_name=request.original_name or "original",
        suspect_name=request.suspect_name or "suspect"
    )
    return JSONResponse(content=result)


@app.post("/compare/upload", tags=["Shield"])
async def compare_upload(
    original: UploadFile = File(...),
    suspect: UploadFile = File(...)
):
    result = env.compare_images(
        original_name=original.filename or "original",
        suspect_name=suspect.filename or "suspect"
    )
    return JSONResponse(content=result)


# ---------- Health Check ----------

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "env_initialized": env._state.scenario is not None}


# ---------- Serve Frontend ----------
# This must be LAST — it catches all remaining routes
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {
            "system": "AegisVision DualShield Interface",
            "status": "ONLINE",
            "version": "1.0.0",
            "note": "Frontend not found. Place index.html in /frontend folder.",
            "endpoints": ["/reset", "/state", "/step/{action}", "/protect", "/compare"]
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)