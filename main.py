import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from digit_modifier import DateDigitModifier
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
app = FastAPI(
    title="Date Year Digit Modifier API",
    description="High-precision API for modifying year digits in [200x-yy-zz] date fields on document images."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)
modifier = DateDigitModifier()
@app.post("/api/modify")
async def modify_image(
    file: UploadFile = File(...),
    digit: str = Form(...),
    platform: str = Form("auto")
):
    try:
        if not digit.isdigit() or len(digit) != 1:
            raise HTTPException(status_code=400, detail="Digit must be a single number (0-9).")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        result = modifier.process_image(img, digit, platform=platform)
        _, encoded_img = cv2.imencode(".png", result['image'])
        return Response(
            content=encoded_img.tobytes(),
            media_type="image/png",
            headers={"X-Method-Used": result['method']}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")