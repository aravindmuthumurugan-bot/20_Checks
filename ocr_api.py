from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import os
import shutil
from pathlib import Path
import asyncio
from datetime import datetime
import json

# Import the previous PII detection modules
import cv2
import numpy as np
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from PIL import Image
import easyocr
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# GPU Configuration (from previous code)
# ============================================================================

def configure_gpu():
    """Configure TensorFlow and ONNX Runtime to use GPU efficiently"""
    print("="*60)
    print("GPU CONFIGURATION")
    print("="*60)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs Available: {len(gpus)}")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Device: {gpus[0]}")
            print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")
            print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (float16)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("WARNING: No GPU detected for TensorFlow. Running on CPU.")

    print("="*60 + "\n")
    return len(gpus) > 0

TF_GPU_AVAILABLE = configure_gpu()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    ONNX_GPU_AVAILABLE = 'CUDAExecutionProvider' in providers
except:
    ONNX_GPU_AVAILABLE = False

GPU_AVAILABLE = TF_GPU_AVAILABLE or ONNX_GPU_AVAILABLE

# ============================================================================
# Core Classes (from previous code)
# ============================================================================

class PIIType(Enum):
    """Enumeration of PII types"""
    MOBILE_NUMBER = "Mobile Number"
    EMAIL = "Email Address"
    INSTAGRAM_ID = "Instagram ID"
    TWITTER_ID = "Twitter/X ID"
    FACEBOOK_ID = "Facebook ID"
    AADHAR_NUMBER = "Aadhar Number"
    PAN_CARD = "PAN Card"
    CREDIT_CARD = "Credit Card"
    SSN = "Social Security Number"
    ADDRESS = "Physical Address"
    UNKNOWN = "Unknown PII"


@dataclass
class PIIDetectionResult:
    """Data class to store PII detection results"""
    pii_type: str
    matched_text: str
    confidence: float


@dataclass
class ImageProcessingResult:
    """Data class to store image processing results"""
    image_path: str
    image_index: int
    text_detected: bool
    detected_text: str
    contains_pii: bool
    pii_details: List[Dict[str, Any]]
    status: str
    processing_time: float


class PIIDetector:
    """Class to detect various types of PII in text"""
    
    PATTERNS = {
        PIIType.MOBILE_NUMBER: [
            r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',
            r'\+?91[\s-]?[6-9]\d{9}',
            r'\+?[0-9]{1,4}[\s-]?[0-9]{10,}',
            r'[6-9]\d{9}',
        ],
        PIIType.EMAIL: [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ],
        PIIType.INSTAGRAM_ID: [
            r'@[A-Za-z0-9._]{1,30}',
            r'instagram\.com/[A-Za-z0-9._]+',
            r'ig:\s*@?[A-Za-z0-9._]+',
        ],
        PIIType.TWITTER_ID: [
            r'@[A-Za-z0-9_]{1,15}(?:\s|$)',
            r'twitter\.com/[A-Za-z0-9_]+',
            r'x\.com/[A-Za-z0-9_]+',
        ],
        PIIType.FACEBOOK_ID: [
            r'facebook\.com/[A-Za-z0-9.]+',
            r'fb\.com/[A-Za-z0-9.]+',
        ],
        PIIType.AADHAR_NUMBER: [
            r'\d{4}\s?\d{4}\s?\d{4}',
        ],
        PIIType.PAN_CARD: [
            r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
        ],
        PIIType.CREDIT_CARD: [
            r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
        ],
        PIIType.SSN: [
            r'\d{3}-\d{2}-\d{4}',
        ],
    }

    @staticmethod
    def detect_pii(text: str) -> tuple:
        """Detect PII in the given text"""
        pii_results = []
        
        for pii_type, patterns in PIIDetector.PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()
                    if PIIDetector._validate_match(pii_type, matched_text):
                        pii_results.append({
                            "pii_type": pii_type.value,
                            "matched_text": matched_text,
                            "confidence": 0.95
                        })
        
        return len(pii_results) > 0, pii_results

    @staticmethod
    def _validate_match(pii_type: PIIType, text: str) -> bool:
        """Additional validation for matched PII"""
        if pii_type == PIIType.EMAIL:
            return '@' in text and '.' in text.split('@')[1]
        elif pii_type == PIIType.MOBILE_NUMBER:
            digits = re.sub(r'\D', '', text)
            return len(digits) >= 10
        elif pii_type == PIIType.CREDIT_CARD:
            digits = re.sub(r'\D', '', text)
            if len(digits) < 13 or len(digits) > 19:
                return False
            return PIIDetector._luhn_check(digits)
        return True

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """Luhn algorithm for credit card validation"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10 == 0


class TextDetector:
    """Class to perform text detection using GPU-accelerated OCR"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("EasyOCR initialized with GPU support")
        else:
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized with CPU")
    
    def detect_text(self, image_path: str) -> tuple:
        """Detect text in image using GPU-accelerated OCR"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, ""
            
            results = self.reader.readtext(img)
            detected_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    detected_texts.append(text)
            
            combined_text = ' '.join(detected_texts)
            return len(combined_text.strip()) > 0, combined_text
            
        except Exception as e:
            print(f"Error in text detection: {e}")
            return False, ""


def process_single_image(
    image_path: str,
    image_index: int,
    text_detector: TextDetector,
    pii_detector: PIIDetector
) -> Dict[str, Any]:
    """Process a single image for text detection and PII check"""
    import time
    start_time = time.time()
    
    text_detected, extracted_text = text_detector.detect_text(image_path)
    
    if not text_detected:
        return {
            "image_path": image_path,
            "image_index": image_index,
            "text_detected": False,
            "detected_text": "",
            "contains_pii": False,
            "pii_details": [],
            "status": "APPROVED",
            "processing_time": time.time() - start_time
        }
    
    contains_pii, pii_details = pii_detector.detect_pii(extracted_text)
    status = "REJECTED" if contains_pii else "APPROVED"
    
    return {
        "image_path": image_path,
        "image_index": image_index,
        "text_detected": True,
        "detected_text": extracted_text,
        "contains_pii": contains_pii,
        "pii_details": pii_details,
        "status": status,
        "processing_time": time.time() - start_time
    }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PII Detection API",
    description="API for detecting PII in images using GPU-accelerated OCR",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize detectors (singleton)
text_detector = TextDetector(use_gpu=True)
pii_detector = PIIDetector()

# Job storage (in production, use Redis or database)
job_storage: Dict[str, Dict] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class PIICheckResponse(BaseModel):
    """Response model for PII check"""
    image_path: str
    image_index: int
    text_detected: bool
    detected_text: str
    contains_pii: bool
    pii_details: List[Dict[str, Any]]
    status: str
    processing_time: float


class BatchPIICheckResponse(BaseModel):
    """Response model for batch PII check"""
    job_id: str
    total_images: int
    results: List[PIICheckResponse]
    summary: Dict[str, Any]
    total_processing_time: float


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float
    total_images: int
    processed_images: int
    results: Optional[List[Dict[str, Any]]] = None
    summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    gpu_available: bool
    tensorflow_gpu: bool
    onnx_gpu: bool
    timestamp: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "PII Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "check_single_image": "/api/v1/check-image",
            "check_multiple_images": "/api/v1/check-images",
            "check_images_async": "/api/v1/check-images-async",
            "job_status": "/api/v1/job/{job_id}",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        gpu_available=GPU_AVAILABLE,
        tensorflow_gpu=TF_GPU_AVAILABLE,
        onnx_gpu=ONNX_GPU_AVAILABLE,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/check-image", response_model=PIICheckResponse, tags=["PII Detection"])
async def check_single_image(
    file: UploadFile = File(..., description="Image file to check for PII")
):
    """
    Check a single image for PII
    
    - Detects text in the image using GPU-accelerated OCR
    - Checks for PII (mobile numbers, emails, social media IDs, etc.)
    - Returns APPROVED if no PII found, REJECTED if PII detected
    """
    try:
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        result = process_single_image(
            str(file_path),
            0,
            text_detector,
            pii_detector
        )
        
        # Clean up
        os.remove(file_path)
        
        return PIICheckResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/check-images", response_model=BatchPIICheckResponse, tags=["PII Detection"])
async def check_multiple_images(
    files: List[UploadFile] = File(..., description="Multiple image files to check for PII"),
    max_workers: int = 4
):
    """
    Check multiple images for PII (synchronous)
    
    - Processes multiple images in parallel using threading
    - Uses GPU acceleration for OCR
    - Returns results for all images with summary statistics
    """
    try:
        import time
        start_time = time.time()
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 images allowed per request")
        
        # Save all uploaded files
        file_paths = []
        for file in files:
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(str(file_path))
        
        # Process images in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    process_single_image,
                    file_path,
                    idx,
                    text_detector,
                    pii_detector
                ): (file_path, idx)
                for idx, file_path in enumerate(file_paths)
            }
            
            for future in as_completed(future_to_image):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing image: {e}")
        
        # Sort results by index
        results.sort(key=lambda x: x["image_index"])
        
        # Calculate summary
        approved = sum(1 for r in results if r["status"] == "APPROVED")
        rejected = sum(1 for r in results if r["status"] == "REJECTED")
        total_pii_found = sum(len(r["pii_details"]) for r in results)
        
        summary = {
            "total_images": len(files),
            "approved": approved,
            "rejected": rejected,
            "total_pii_instances": total_pii_found,
            "text_detected_count": sum(1 for r in results if r["text_detected"])
        }
        
        # Clean up files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        
        total_time = time.time() - start_time
        
        return BatchPIICheckResponse(
            job_id=str(uuid.uuid4()),
            total_images=len(files),
            results=[PIICheckResponse(**r) for r in results],
            summary=summary,
            total_processing_time=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/api/v1/check-images-async", tags=["PII Detection"])
async def check_multiple_images_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple image files to check for PII"),
    max_workers: int = 4
):
    """
    Check multiple images for PII (asynchronous)
    
    - Accepts multiple images and processes them in the background
    - Returns a job_id immediately
    - Use /api/v1/job/{job_id} to check status and get results
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 images allowed per request")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save all uploaded files
        file_paths = []
        for file in files:
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(str(file_path))
        
        # Initialize job status
        job_storage[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "total_images": len(files),
            "processed_images": 0,
            "results": None,
            "summary": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Add background task
        background_tasks.add_task(
            process_images_background,
            job_id,
            file_paths,
            max_workers
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Images are being processed in the background",
            "total_images": len(files),
            "check_status_at": f"/api/v1/job/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting job: {str(e)}")


@app.get("/api/v1/job/{job_id}", response_model=JobStatusResponse, tags=["Job Management"])
async def get_job_status(job_id: str):
    """
    Get the status of an async job
    
    - Returns job status (processing/completed/failed)
    - Progress percentage
    - Results when completed
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        total_images=job_data["total_images"],
        processed_images=job_data["processed_images"],
        results=job_data.get("results"),
        summary=job_data.get("summary")
    )


@app.delete("/api/v1/job/{job_id}", tags=["Job Management"])
async def delete_job(job_id: str):
    """Delete a completed job and its results"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_storage[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


# ============================================================================
# Background Task Functions
# ============================================================================

def process_images_background(job_id: str, file_paths: List[str], max_workers: int):
    """Background task to process images"""
    try:
        import time
        
        results = []
        total = len(file_paths)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    process_single_image,
                    file_path,
                    idx,
                    text_detector,
                    pii_detector
                ): (file_path, idx)
                for idx, file_path in enumerate(file_paths)
            }
            
            for future in as_completed(future_to_image):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress
                    job_storage[job_id]["processed_images"] = len(results)
                    job_storage[job_id]["progress"] = (len(results) / total) * 100
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
        
        # Sort results
        results.sort(key=lambda x: x["image_index"])
        
        # Calculate summary
        approved = sum(1 for r in results if r["status"] == "APPROVED")
        rejected = sum(1 for r in results if r["status"] == "REJECTED")
        total_pii_found = sum(len(r["pii_details"]) for r in results)
        
        summary = {
            "total_images": total,
            "approved": approved,
            "rejected": rejected,
            "total_pii_instances": total_pii_found,
            "text_detected_count": sum(1 for r in results if r["text_detected"])
        }
        
        # Update job status
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["progress"] = 100.0
        job_storage[job_id]["results"] = results
        job_storage[job_id]["summary"] = summary
        job_storage[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Clean up files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
                
    except Exception as e:
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        print(f"Background task error: {e}")


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("\n" + "="*80)
    print("PII DETECTION API STARTUP")
    print("="*80)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"TensorFlow GPU: {TF_GPU_AVAILABLE}")
    print(f"ONNX GPU: {ONNX_GPU_AVAILABLE}")
    print(f"Upload Directory: {UPLOAD_DIR.absolute()}")
    print("="*80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nShutting down PII Detection API...")
    # Clean up upload directory
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


# ============================================================================
# Run the application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",  # Change "main" to your filename
        host="0.0.0.0",
        port=8001,
        reload=True,
        workers=1  # Use 1 worker for GPU
    )