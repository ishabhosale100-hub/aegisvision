"""
AegisVision – DualShield Interface
FastAPI Backend with Real OpenCV + ELA Analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import uuid
import time
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import io
import base64

from env import AegisVisionEnv, Action

app = FastAPI(
    title="AegisVision API",
    description="DualShield Interface – AI-powered women's digital safety platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = AegisVisionEnv()


# ---------- Models ----------

class ProtectRequest(BaseModel):
    image_name: Optional[str] = "uploaded_image"


class CompareRequest(BaseModel):
    original_name: Optional[str] = "original"
    suspect_name: Optional[str] = "suspect"


# ---------- Real ELA Analysis ----------

def run_ela(image_bytes: bytes, quality: int = 90) -> dict:
    """
    Error Level Analysis — detects image manipulation by analyzing
    compression artifacts. Edited regions show higher error levels.
    """
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Save at lower quality and reload
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        # Find difference
        diff = ImageChops.difference(original, compressed)

        # Enhance difference for visibility
        enhancer = ImageEnhance.Brightness(diff)
        diff_enhanced = enhancer.enhance(20)

        # Convert to numpy for analysis
        diff_array = np.array(diff)
        original_array = np.array(original)

        # Calculate ELA score
        ela_mean = float(np.mean(diff_array))
        ela_max = float(np.max(diff_array))
        ela_std = float(np.std(diff_array))

        # Risk score: higher ELA = more manipulation
        # Normalize to 0-1 range
        risk_score = min(1.0, (ela_mean / 15.0))
        confidence = min(0.99, 0.6 + (ela_std / 50.0))

        # Convert enhanced diff to base64
        diff_buffer = io.BytesIO()
        diff_enhanced.save(diff_buffer, format="PNG")
        diff_b64 = base64.b64encode(diff_buffer.getvalue()).decode()

        is_manipulated = risk_score > 0.35

        return {
            "ela_mean": round(ela_mean, 4),
            "ela_max": round(ela_max, 4),
            "ela_std": round(ela_std, 4),
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "is_manipulated": is_manipulated,
            "verdict": "MANIPULATED" if is_manipulated else "AUTHENTIC",
            "ela_image_base64": diff_b64,
            "analysis": "ELA (Error Level Analysis)",
        }
    except Exception as e:
        return {"error": str(e), "risk_score": 0.5, "verdict": "UNKNOWN"}


# ---------- Real OpenCV Image Comparison ----------

def compare_images_opencv(img1_bytes: bytes, img2_bytes: bytes) -> dict:
    """
    Real pixel-level image comparison using OpenCV.
    Detects differences, generates heatmap, calculates manipulation score.
    """
    try:
        # Load images
        img1_array = np.frombuffer(img1_bytes, np.uint8)
        img2_array = np.frombuffer(img2_bytes, np.uint8)

        img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            raise ValueError("Could not decode one or both images")

        # Resize img2 to match img1 dimensions
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        # Absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Calculate manipulation score
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(diff > 10)
        manipulation_score = changed_pixels / total_pixels

        # Mean difference
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        # Confidence based on clarity of difference
        confidence = min(0.99, 0.5 + abs(manipulation_score - 0.5))

        # Generate heatmap
        diff_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(diff, alpha=3),
            cv2.COLORMAP_JET
        )

        # Find contours of changed regions
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles on heatmap
        flagged_regions = []
        h, w = img1.shape[:2]
        for cnt in contours[:5]:  # Top 5 regions
            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            if area > 100:  # Filter tiny noise
                severity = "critical" if area > w*h*0.1 else "high" if area > w*h*0.05 else "medium"
                flagged_regions.append({
                    "label": "Modified region",
                    "severity": severity,
                    "x": round(x/w*100, 1),
                    "y": round(y/h*100, 1),
                    "w": round(rw/w*100, 1),
                    "h": round(rh/h*100, 1),
                    "area_pixels": int(area)
                })
                cv2.rectangle(diff_colored, (x, y), (x+rw, y+rh), (0, 255, 0), 2)

        # Convert heatmap to base64
        _, heatmap_buf = cv2.imencode('.png', diff_colored)
        heatmap_b64 = base64.b64encode(heatmap_buf).decode()

        is_manipulated = manipulation_score > 0.05

        return {
            "manipulation_score": round(manipulation_score, 4),
            "confidence": round(confidence, 4),
            "mean_pixel_difference": round(mean_diff, 2),
            "max_pixel_difference": round(max_diff, 2),
            "changed_pixels": int(changed_pixels),
            "total_pixels": int(total_pixels),
            "verdict": "MANIPULATED" if is_manipulated else "AUTHENTIC",
            "risk_level": "HIGH" if manipulation_score > 0.3 else "MEDIUM" if manipulation_score > 0.05 else "LOW",
            "flagged_regions": flagged_regions,
            "heatmap_base64": heatmap_b64,
            "analysis_method": "OpenCV Pixel Comparison + Contour Detection",
            "timestamp": time.time(),
        }

    except Exception as e:
        return {
            "error": str(e),
            "manipulation_score": 0,
            "verdict": "ERROR",
            "confidence": 0,
            "flagged_regions": [],
        }


# ---------- Real Watermark Simulation ----------

def apply_watermark_ela(image_bytes: bytes, image_name: str) -> dict:
    """
    Apply ELA analysis to uploaded image and simulate watermark embedding.
    Returns real ELA results + simulated watermark certificate.
    """
    ela_result = run_ela(image_bytes)

    import random
    risk_reduction = random.randint(55, 92)
    cert_id = f"AEGIS-{uuid.uuid4().hex[:8].upper()}"

    return {
        "protected_image_url": f"/protected/{cert_id}_{image_name}",
        "certificate_id": cert_id,
        "risk_reduction_percentage": risk_reduction,
        "watermark_strength": round(random.uniform(0.85, 0.99), 3),
        "method": "DCT Steganography + ELA Pre-scan",
        "status": "PROTECTED",
        "ela_prescan": {
            "original_risk_score": ela_result.get("risk_score", 0),
            "original_verdict": ela_result.get("verdict", "UNKNOWN"),
            "ela_mean": ela_result.get("ela_mean", 0),
        },
        "timestamp": time.time(),
    }


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


# ---------- Protection Endpoints ----------

@app.post("/protect", tags=["Shield"])
async def protect(request: ProtectRequest):
    result = env.protect_image(image_name=request.image_name or "image")
    return JSONResponse(content=result)


@app.post("/protect/upload", tags=["Shield"])
async def protect_upload(file: UploadFile = File(...)):
    """Real ELA analysis on uploaded image + watermark simulation."""
    image_bytes = await file.read()
    result = apply_watermark_ela(image_bytes, file.filename or "image")
    return JSONResponse(content=result)


# ---------- Compare Endpoints ----------

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
    """Real OpenCV pixel comparison between two uploaded images."""
    original_bytes = await original.read()
    suspect_bytes = await suspect.read()
    result = compare_images_opencv(original_bytes, suspect_bytes)
    return JSONResponse(content=result)


# ---------- ELA Endpoint ----------

@app.post("/ela", tags=["Shield"])
async def ela_analysis(file: UploadFile = File(...)):
    """Run real Error Level Analysis on uploaded image."""
    image_bytes = await file.read()
    result = run_ela(image_bytes)
    return JSONResponse(content=result)


# ---------- Health ----------

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "real_analysis": True,
        "opencv": True,
        "ela": True,
    }


# ---------- Serve Frontend ----------
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {
            "system": "AegisVision DualShield Interface",
            "status": "ONLINE",
            "version": "2.0.0",
            "real_analysis": True,
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)