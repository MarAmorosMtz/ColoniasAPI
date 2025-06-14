import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os
import cv2
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/")
def index():
    return {"message" : "Jorge choooto"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        image.save(file_path)
        return {"filename": file.filename, "saved_to": file_path}
    except Exception as e:
        return {"error": str(e)}

