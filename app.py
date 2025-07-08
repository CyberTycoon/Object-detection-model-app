import os
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI(title="YOLOv8 Object Detection API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Update this path to where your yolov8n.pt is located
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

def detect_objects(image, confidence_threshold=0.25):
    if image is None:
        return None, [], "Please upload an image first."
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    results = model.predict(img_array, conf=confidence_threshold)
    result = results[0]
    annotated_img = result.plot()
    annotated_img = Image.fromarray(annotated_img)
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            confidence = box.conf[0].item()
            detections.append({
                "class": class_name,
                "confidence": confidence
            })
    detection_summary = generate_summary(result)
    return annotated_img, detections, detection_summary

def generate_summary(result):
    if result.boxes is None or len(result.boxes) == 0:
        return "No objects detected with the current confidence threshold."
    summary_lines = [f"Detected {len(result.boxes)} objects:"]
    class_counts = {}
    detections_detail = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = box.conf[0].item()
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        detections_detail.append(f"  • {class_name}: {confidence:.2%} confidence")
    summary_lines.append("Summary by Object Type:")
    for class_name, count in sorted(class_counts.items()):
        summary_lines.append(f"  • {class_name}: {count}")
    summary_lines.append("Detailed Detections:")
    summary_lines.extend(detections_detail)
    summary_lines.append("Model Information:")
    summary_lines.append("  • Model: YOLOv8n (nano variant)")
    summary_lines.append("  • Classes: 80 COCO dataset classes")
    summary_lines.append("  • Framework: Ultralytics YOLOv8")
    return "\n".join(summary_lines)

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLOv8 Object Detection API!"}

@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    annotated_img, detections, summary = detect_objects(image, confidence)
    img_base64 = image_to_base64(annotated_img)
    return JSONResponse(content={
        "image_base64": img_base64,
        "detections": detections,
        "summary": summary
    })

@app.post("/detect-json/")
async def detect_json(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    _, detections, summary = detect_objects(image, confidence)
    return JSONResponse(content={
        "detections": detections,
        "summary": summary
    })

@app.get("/test")
def test():
    return {"status": "ok"}