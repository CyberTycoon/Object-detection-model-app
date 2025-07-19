import os
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Object Detection API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Create directories if they don't exist
Path(STATIC_DIR).mkdir(exist_ok=True)
Path(TEMPLATES_DIR).mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

model = None  # Global cache

@app.on_event("startup")
async def load_model():
    global model
    try:
        logger.info("Starting model loading process...")
        
        # Check if ultralytics is properly installed
        try:
            import ultralytics
            logger.info(f"Ultralytics version: {ultralytics.__version__}")
        except ImportError as e:
            logger.error("Ultralytics not found. Install with: pip install ultralytics")
            raise e
        
        # Check internet connectivity for model download
        logger.info("Loading YOLO model...")
        
        # Try different model loading approaches
        model_options = ["yolov8n.pt", "yolov8s.pt"]
        
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load {model_name}...")
                model = YOLO(model_name)
                logger.info(f"Model {model_name} loaded successfully!")
                
                # Test the model with a dummy prediction to ensure it's working
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = model.predict(dummy_image, verbose=False)
                logger.info("Model test prediction successful!")
                break
                
            except Exception as model_error:
                logger.error(f"Failed to load {model_name}: {model_error}")
                model = None
                continue
        
        if model is None:
            raise RuntimeError("All model loading attempts failed")
            
    except Exception as e:
        logger.error(f"Critical error during model loading: {e}", exc_info=True)
        model = None
        # Don't raise here to keep the app running, but log the error clearly
        logger.error("APPLICATION WILL NOT FUNCTION PROPERLY - MODEL LOADING FAILED")

def get_model():
    if model is None:
        logger.error("Model is not available. Check startup logs for errors.")
        raise HTTPException(
            status_code=503, 
            detail="Model service unavailable. Please check server logs and try again later."
        )
    return model

def detect_objects(image, confidence_threshold=0.25):
    """Detect objects in an image using YOLOv8"""
    if image is None:
        return None, [], "Please upload an image first."
    
    try:
        # Ensure image is in RGB format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for YOLO
        img_array = np.array(image)
        
        # Get model and run prediction
        current_model = get_model()
        results = current_model.predict(img_array, conf=confidence_threshold, verbose=False)
        result = results[0]
        
        # Create annotated image
        annotated_img = result.plot()
        annotated_img = Image.fromarray(annotated_img)
        
        # Extract detections
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
        
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

def generate_summary(result):
    """Generate a human-readable summary of detections"""
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
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Serve the upload form"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        current_model = get_model()
        return {"status": "healthy", "model_loaded": True}
    except:
        return {"status": "unhealthy", "model_loaded": False}

@app.post("/detect", response_class=JSONResponse)
async def detect_objects_endpoint(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """Main detection endpoint"""
    try:
        # Validate confidence threshold
        if not 0.01 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.01 and 1.0")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Run object detection
        annotated_img, detections, summary = detect_objects(image, confidence)
        
        if annotated_img is None:
            raise HTTPException(status_code=500, detail="Failed to process image")

        img_base64 = image_to_base64(annotated_img)
        
        return JSONResponse(content={
            "annotated_image": img_base64,
            "detections": detections,
            "detection_summary": summary,
            "total_detections": len(detections),
            "confidence_threshold": confidence
        })
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in /detect endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# Add a simple diagnostic endpoint
@app.get("/diagnostics")
async def diagnostics():
    """Diagnostic information"""
    import torch
    import platform
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_available": True,
        "cuda_available": False,
        "model_loaded": model is not None
    }
    
    try:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except:
        info["torch_available"] = False
    
    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except:
        info["ultralytics_available"] = False
    
    return info