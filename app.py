import io
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from ultralytics import YOLO
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Object Detection API")

# Load the model directly from the Google Drive URL as you requested.
# Note: The YOLO library might not support Google Drive links directly.
# If this fails, we will need to download the file first.
MODEL_URL = "https://drive.google.com/uc?export=download&id=13sDjGcLhDUjTM8hvKlASYgkQWIlkgcMK"
model = None

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    logger.info("Loading YOLOv8 model...")
    try:
        # We will try to load the model from a local path first.
        # If it doesn't exist, we'll download it. This is more robust.
        local_model_path = "best.pt"
        if not __import__('os').path.exists(local_model_path):
            logger.info("Model not found locally, downloading from Google Drive...")
            import requests
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(local_model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("Model downloaded successfully.")
        model = YOLO(local_model_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Fatal error loading model: {e}")
        # We raise a runtime error to stop the app if the model can't be loaded.
        raise RuntimeError(f"Failed to load model: {e}")

def process_image_and_predict(image_bytes):
    """Receives image bytes, performs prediction, and returns results."""
    assert model is not None, "Model has not been loaded."
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(np.array(image))
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = model.names[cls]
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bounding_box": xyxy
                })
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
