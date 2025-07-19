import io
import logging
import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from ultralytics import YOLO
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Object Detection API")

model = None
MODEL_URL = "https://drive.google.com/uc?export=download&id=13sDjGcLhDUjTM8hvKlASYgkQWIlkgcMK"
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.1

def download_file_from_google_drive(url, destination):
    """Downloads a file from a Google Drive link."""
    logger.info(f"Attempting to download model from {url} to {destination}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Model downloaded successfully to {destination}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download model: {e}")
        return False

def download_yolov8_model():
    """Checks for the model file and downloads it if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found locally. Starting download...")
        download_file_from_google_drive(MODEL_URL, MODEL_PATH)
    else:
        logger.info(f"Model already exists at {MODEL_PATH}. Skipping download.")

@app.on_event("startup")
async def load_model():
    """Load the model on startup after ensuring it's downloaded."""
    global model
    logger.info("Ensuring YOLOv8 model is available...")
    try:
        download_yolov8_model()
        logger.info("Model check/download complete.")
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}")
        raise RuntimeError(f"Failed to initialize the application: {e}")

def process_image_and_predict(image_bytes):
    """Receives image bytes, performs prediction, and returns results."""
    assert model is not None, "Model has not been loaded."
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(np.array(image), conf=CONFIDENCE_THRESHOLD)
        
        detections = []
        logger.info(f"Processing {len(results)} result sets.")
        for result in results:
            boxes = result.boxes
            logger.info(f"Model returned {len(boxes)} boxes.")
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                xyxy = boxes.xyxy[i].tolist()
                cls = int(boxes.cls[i].item())
                class_name = model.names[cls]
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bounding_box": xyxy
                })
        logger.info(f"Returning {len(detections)} detections.")
        return detections
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.post("/predict/")
async def create_prediction(file: UploadFile = File(...)):
    """Endpoint to receive an image and return predictions."""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    contents = await file.read()
    detections = process_image_and_predict(contents)
    return {"detections": detections}

@app.get("/summary/")
async def get_summary():
    """Endpoint to get a summary of the loaded model."""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    # The str(model) gives a good summary of the model architecture.
    return {"model_summary": str(model)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
