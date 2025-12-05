"""
FastAPI Backend for DIP Project
REST API for image processing operations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional, List
import sys
import os
import zipfile
import json
import asyncio
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basic_operations import (
    resize_image, rotate_image, flip_image, negative_image,
    adjust_brightness, adjust_contrast, gamma_correction,
    histogram_equalization, translate_image, shear_image
)
from src.filters import (
    average_filter, gaussian_filter, median_filter, bilateral_filter,
    sharpen_filter, unsharp_mask, non_local_means_denoise, emboss_filter,
    motion_blur, add_gaussian_noise, add_salt_pepper_noise
)
from src.edge_detection import (
    sobel_edge_detection, canny_edge_detection, laplacian_edge_detection,
    prewitt_edge_detection, roberts_edge_detection, scharr_edge_detection,
    auto_canny
)
from src.segmentation import (
    simple_threshold, otsu_threshold, adaptive_threshold,
    kmeans_segmentation, contour_detection, draw_contours
)
from src.morphology import (
    erosion, dilation, opening, closing, morphological_gradient,
    top_hat, black_hat, skeletonization, boundary_extraction
)
from src.frequency_domain import (
    compute_dft, get_magnitude_spectrum, ideal_lowpass_filter,
    ideal_highpass_filter, butterworth_lowpass_filter, gaussian_lowpass_filter,
    homomorphic_filter
)
from src.feature_detection import (
    detect_harris_corners, detect_shi_tomasi_corners, detect_orb_features,
    detect_lines_houghp, detect_circles_hough
)
from src.deep_learning import (
    detect_faces_haar, detect_eyes, neural_style_transfer_simple,
    pencil_sketch, stylization, detail_enhance
)

# Create FastAPI app
app = FastAPI(
    title="DIP Project API",
    description="Digital Image Processing REST API",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def encode_image(img: np.ndarray, format: str = "png") -> bytes:
    """Encode numpy array to image bytes."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    success, encoded = cv2.imencode(f".{format}", img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return encoded.tobytes()


def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    encoded = encode_image(img)
    return base64.b64encode(encoded).decode('utf-8')


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "DIP Project API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/operations")
async def get_operations():
    """Get list of all available operations."""
    return {
        "categories": {
            "basic": [
                "resize", "rotate", "flip", "brightness", "contrast",
                "gamma", "negative", "histogram_equalization"
            ],
            "filters": [
                "average", "gaussian", "median", "bilateral", "sharpen",
                "unsharp_mask", "emboss", "motion_blur", "denoise",
                "add_gaussian_noise", "add_salt_pepper_noise"
            ],
            "edge_detection": [
                "sobel", "canny", "laplacian", "prewitt", "roberts",
                "scharr", "auto_canny"
            ],
            "segmentation": [
                "simple_threshold", "otsu_threshold", "adaptive_threshold",
                "kmeans", "contours"
            ],
            "morphology": [
                "erosion", "dilation", "opening", "closing", "gradient",
                "top_hat", "black_hat", "skeleton", "boundary"
            ],
            "frequency": [
                "magnitude_spectrum", "ideal_lowpass", "ideal_highpass",
                "butterworth_lowpass", "gaussian_lowpass", "homomorphic"
            ],
            "features": [
                "harris_corners", "shi_tomasi", "orb", "hough_lines", "hough_circles"
            ],
            "deep_learning": [
                "face_detection", "eye_detection", "pencil_sketch",
                "cartoon", "stylization", "detail_enhance"
            ]
        }
    }


# ==================== BASIC OPERATIONS ====================

@app.post("/api/basic/resize")
async def api_resize(
    file: UploadFile = File(...),
    scale: float = Query(1.0, ge=0.1, le=5.0)
):
    """Resize image by scale factor."""
    img = decode_image(await file.read())
    result = resize_image(img, scale=scale)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/rotate")
async def api_rotate(
    file: UploadFile = File(...),
    angle: float = Query(0, ge=-180, le=180)
):
    """Rotate image by angle in degrees."""
    img = decode_image(await file.read())
    result = rotate_image(img, angle)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/flip")
async def api_flip(
    file: UploadFile = File(...),
    direction: str = Query("horizontal", regex="^(horizontal|vertical|both)$")
):
    """Flip image horizontally, vertically, or both."""
    img = decode_image(await file.read())
    result = flip_image(img, direction)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/brightness")
async def api_brightness(
    file: UploadFile = File(...),
    value: int = Query(0, ge=-100, le=100)
):
    """Adjust image brightness."""
    img = decode_image(await file.read())
    result = adjust_brightness(img, value)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/contrast")
async def api_contrast(
    file: UploadFile = File(...),
    factor: float = Query(1.0, ge=0.1, le=3.0)
):
    """Adjust image contrast."""
    img = decode_image(await file.read())
    result = adjust_contrast(img, factor)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/gamma")
async def api_gamma(
    file: UploadFile = File(...),
    gamma: float = Query(1.0, ge=0.1, le=3.0)
):
    """Apply gamma correction."""
    img = decode_image(await file.read())
    result = gamma_correction(img, gamma)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/negative")
async def api_negative(file: UploadFile = File(...)):
    """Create negative of image."""
    img = decode_image(await file.read())
    result = negative_image(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/basic/histogram")
async def api_histogram(
    file: UploadFile = File(...),
    use_clahe: bool = Query(False)
):
    """Apply histogram equalization."""
    img = decode_image(await file.read())
    result = histogram_equalization(img, use_clahe=use_clahe)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== FILTERS ====================

@app.post("/api/filters/average")
async def api_average_filter(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21)
):
    """Apply average filter."""
    img = decode_image(await file.read())
    result = average_filter(img, kernel_size)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/filters/gaussian")
async def api_gaussian_filter(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21),
    sigma: float = Query(0, ge=0, le=10)
):
    """Apply Gaussian filter."""
    img = decode_image(await file.read())
    result = gaussian_filter(img, kernel_size, sigma)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/filters/median")
async def api_median_filter(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21)
):
    """Apply median filter."""
    img = decode_image(await file.read())
    result = median_filter(img, kernel_size)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/filters/bilateral")
async def api_bilateral_filter(
    file: UploadFile = File(...),
    d: int = Query(9, ge=3, le=15),
    sigma_color: float = Query(75, ge=10, le=200),
    sigma_space: float = Query(75, ge=10, le=200)
):
    """Apply bilateral filter."""
    img = decode_image(await file.read())
    result = bilateral_filter(img, d, sigma_color, sigma_space)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/filters/sharpen")
async def api_sharpen_filter(
    file: UploadFile = File(...),
    strength: float = Query(1.0, ge=0.1, le=3.0)
):
    """Apply sharpening filter."""
    img = decode_image(await file.read())
    result = sharpen_filter(img, strength)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/filters/emboss")
async def api_emboss_filter(file: UploadFile = File(...)):
    """Apply emboss filter."""
    img = decode_image(await file.read())
    result = emboss_filter(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/filters/denoise")
async def api_denoise(
    file: UploadFile = File(...),
    strength: int = Query(10, ge=1, le=30)
):
    """Apply non-local means denoising."""
    img = decode_image(await file.read())
    result = non_local_means_denoise(img, h=strength)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== EDGE DETECTION ====================

@app.post("/api/edge/sobel")
async def api_sobel(
    file: UploadFile = File(...),
    ksize: int = Query(3, ge=1, le=7)
):
    """Apply Sobel edge detection."""
    img = decode_image(await file.read())
    result = sobel_edge_detection(img, ksize=ksize)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/edge/canny")
async def api_canny(
    file: UploadFile = File(...),
    low_threshold: int = Query(50, ge=0, le=200),
    high_threshold: int = Query(150, ge=50, le=300)
):
    """Apply Canny edge detection."""
    img = decode_image(await file.read())
    result = canny_edge_detection(img, low_threshold, high_threshold)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/edge/laplacian")
async def api_laplacian(
    file: UploadFile = File(...),
    ksize: int = Query(3, ge=1, le=7)
):
    """Apply Laplacian edge detection."""
    img = decode_image(await file.read())
    result = laplacian_edge_detection(img, ksize)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/edge/prewitt")
async def api_prewitt(file: UploadFile = File(...)):
    """Apply Prewitt edge detection."""
    img = decode_image(await file.read())
    result = prewitt_edge_detection(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/edge/roberts")
async def api_roberts(file: UploadFile = File(...)):
    """Apply Roberts edge detection."""
    img = decode_image(await file.read())
    result = roberts_edge_detection(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/edge/auto_canny")
async def api_auto_canny(
    file: UploadFile = File(...),
    sigma: float = Query(0.33, ge=0.1, le=1.0)
):
    """Apply auto Canny edge detection."""
    img = decode_image(await file.read())
    result = auto_canny(img, sigma)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== SEGMENTATION ====================

@app.post("/api/segment/threshold")
async def api_threshold(
    file: UploadFile = File(...),
    threshold: int = Query(127, ge=0, le=255)
):
    """Apply simple thresholding."""
    img = decode_image(await file.read())
    result = simple_threshold(img, threshold)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/segment/otsu")
async def api_otsu(file: UploadFile = File(...)):
    """Apply Otsu thresholding."""
    img = decode_image(await file.read())
    result, thresh = otsu_threshold(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Threshold": str(thresh)}
    )


@app.post("/api/segment/adaptive")
async def api_adaptive_threshold(
    file: UploadFile = File(...),
    method: str = Query("gaussian", regex="^(gaussian|mean)$"),
    block_size: int = Query(11, ge=3, le=51),
    c: int = Query(2, ge=-20, le=20)
):
    """Apply adaptive thresholding."""
    img = decode_image(await file.read())
    result = adaptive_threshold(img, method=method, block_size=block_size, c=c)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/segment/kmeans")
async def api_kmeans(
    file: UploadFile = File(...),
    k: int = Query(3, ge=2, le=10)
):
    """Apply K-means segmentation."""
    img = decode_image(await file.read())
    result = kmeans_segmentation(img, k=k)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/segment/contours")
async def api_contours(
    file: UploadFile = File(...),
    threshold: int = Query(127, ge=0, le=255)
):
    """Detect and draw contours."""
    img = decode_image(await file.read())
    contours = contour_detection(img, threshold=threshold)
    result = draw_contours(img, contours)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Contour-Count": str(len(contours))}
    )


# ==================== MORPHOLOGY ====================

@app.post("/api/morph/erosion")
async def api_erosion(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21),
    iterations: int = Query(1, ge=1, le=10)
):
    """Apply erosion."""
    img = decode_image(await file.read())
    result = erosion(img, kernel_size, iterations)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/morph/dilation")
async def api_dilation(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21),
    iterations: int = Query(1, ge=1, le=10)
):
    """Apply dilation."""
    img = decode_image(await file.read())
    result = dilation(img, kernel_size, iterations)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/morph/opening")
async def api_opening(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21)
):
    """Apply opening."""
    img = decode_image(await file.read())
    result = opening(img, kernel_size)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/morph/closing")
async def api_closing(
    file: UploadFile = File(...),
    kernel_size: int = Query(5, ge=3, le=21)
):
    """Apply closing."""
    img = decode_image(await file.read())
    result = closing(img, kernel_size)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/morph/skeleton")
async def api_skeleton(file: UploadFile = File(...)):
    """Apply skeletonization."""
    img = decode_image(await file.read())
    result = skeletonization(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== FREQUENCY DOMAIN ====================

@app.post("/api/freq/magnitude")
async def api_magnitude_spectrum(file: UploadFile = File(...)):
    """Get magnitude spectrum."""
    img = decode_image(await file.read())
    dft = compute_dft(img)
    result = get_magnitude_spectrum(dft)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/freq/lowpass")
async def api_lowpass(
    file: UploadFile = File(...),
    cutoff: int = Query(30, ge=5, le=200),
    filter_type: str = Query("ideal", regex="^(ideal|butterworth|gaussian)$")
):
    """Apply low-pass filter."""
    img = decode_image(await file.read())
    
    if filter_type == "ideal":
        result = ideal_lowpass_filter(img, cutoff)
    elif filter_type == "butterworth":
        result = butterworth_lowpass_filter(img, cutoff)
    else:
        result = gaussian_lowpass_filter(img, cutoff)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/freq/highpass")
async def api_highpass(
    file: UploadFile = File(...),
    cutoff: int = Query(30, ge=5, le=200)
):
    """Apply high-pass filter."""
    img = decode_image(await file.read())
    result = ideal_highpass_filter(img, cutoff)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/freq/homomorphic")
async def api_homomorphic(
    file: UploadFile = File(...),
    gamma_l: float = Query(0.5, ge=0.1, le=1.0),
    gamma_h: float = Query(2.0, ge=1.0, le=5.0),
    cutoff: int = Query(30, ge=5, le=100)
):
    """Apply homomorphic filter."""
    img = decode_image(await file.read())
    result = homomorphic_filter(img, gamma_l, gamma_h, cutoff)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== FEATURE DETECTION ====================

@app.post("/api/features/harris")
async def api_harris(
    file: UploadFile = File(...),
    threshold: float = Query(0.01, ge=0.001, le=0.1)
):
    """Detect Harris corners."""
    img = decode_image(await file.read())
    result = detect_harris_corners(img, threshold=threshold)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/features/orb")
async def api_orb(
    file: UploadFile = File(...),
    n_features: int = Query(500, ge=100, le=2000)
):
    """Detect ORB features."""
    img = decode_image(await file.read())
    result, keypoints, _ = detect_orb_features(img, n_features)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Keypoint-Count": str(len(keypoints))}
    )


@app.post("/api/features/hough_lines")
async def api_hough_lines(
    file: UploadFile = File(...),
    threshold: int = Query(50, ge=10, le=200),
    min_line_length: int = Query(50, ge=10, le=200)
):
    """Detect lines using Hough transform."""
    img = decode_image(await file.read())
    result, lines = detect_lines_houghp(img, threshold=threshold, min_line_length=min_line_length)
    line_count = len(lines) if lines is not None else 0
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Line-Count": str(line_count)}
    )


@app.post("/api/features/hough_circles")
async def api_hough_circles(
    file: UploadFile = File(...),
    min_dist: int = Query(50, ge=10, le=200),
    param2: int = Query(30, ge=10, le=100)
):
    """Detect circles using Hough transform."""
    img = decode_image(await file.read())
    result, circles = detect_circles_hough(img, min_dist=min_dist, param2=param2)
    circle_count = len(circles[0]) if circles is not None else 0
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Circle-Count": str(circle_count)}
    )


# ==================== DEEP LEARNING ====================

@app.post("/api/dl/face_detection")
async def api_face_detection(file: UploadFile = File(...)):
    """Detect faces."""
    img = decode_image(await file.read())
    result, faces = detect_faces_haar(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Face-Count": str(len(faces))}
    )


@app.post("/api/dl/eye_detection")
async def api_eye_detection(file: UploadFile = File(...)):
    """Detect eyes."""
    img = decode_image(await file.read())
    result, eyes = detect_eyes(img)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png",
        headers={"X-Eye-Count": str(len(eyes))}
    )


@app.post("/api/dl/pencil_sketch")
async def api_pencil_sketch(
    file: UploadFile = File(...),
    shade_factor: float = Query(0.05, ge=0.01, le=0.1)
):
    """Create pencil sketch effect."""
    img = decode_image(await file.read())
    gray_sketch, color_sketch = pencil_sketch(img, shade_factor=shade_factor)
    return StreamingResponse(
        io.BytesIO(encode_image(color_sketch)),
        media_type="image/png"
    )


@app.post("/api/dl/cartoon")
async def api_cartoon(file: UploadFile = File(...)):
    """Apply cartoon effect."""
    img = decode_image(await file.read())
    result = neural_style_transfer_simple(img, style='cartoon')
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/dl/stylization")
async def api_stylization(
    file: UploadFile = File(...),
    sigma_s: float = Query(60, ge=10, le=200),
    sigma_r: float = Query(0.45, ge=0.1, le=1.0)
):
    """Apply stylization effect."""
    img = decode_image(await file.read())
    result = stylization(img, sigma_s, sigma_r)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/dl/detail_enhance")
async def api_detail_enhance(
    file: UploadFile = File(...),
    sigma_s: float = Query(10, ge=1, le=50),
    sigma_r: float = Query(0.15, ge=0.05, le=0.5)
):
    """Enhance image details."""
    img = decode_image(await file.read())
    result = detail_enhance(img, sigma_s, sigma_r)
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ==================== NEW FEATURES ====================

# ==================== BATCH PROCESSING ====================

@app.post("/api/batch/process")
async def batch_process(
    files: List[UploadFile] = File(...),
    operation: str = Query(...),
    params: str = Query("{}")  # JSON string of parameters
):
    """Process multiple images with the same operation and return as ZIP."""
    try:
        operation_params = json.loads(params)
    except json.JSONDecodeError:
        operation_params = {}
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, file in enumerate(files):
            img = decode_image(await file.read())
            result = apply_operation(img, operation, operation_params)
            img_bytes = encode_image(result)
            filename = f"processed_{i+1}_{file.filename or 'image.png'}"
            zip_file.writestr(filename, img_bytes)
    
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=processed_images.zip"}
    )


def apply_operation(img: np.ndarray, operation: str, params: dict) -> np.ndarray:
    """Apply a specific operation to an image."""
    operations_map = {
        'grayscale': lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        'negative': lambda: negative_image(img),
        'flip_h': lambda: flip_image(img, 'horizontal'),
        'flip_v': lambda: flip_image(img, 'vertical'),
        'rotate': lambda: rotate_image(img, params.get('angle', 90)),
        'brightness': lambda: adjust_brightness(img, params.get('value', 50)),
        'contrast': lambda: adjust_contrast(img, params.get('factor', 1.5)),
        'gaussian': lambda: gaussian_filter(img, params.get('kernel_size', 5)),
        'median': lambda: median_filter(img, params.get('kernel_size', 5)),
        'sharpen': lambda: sharpen_filter(img),
        'sobel': lambda: sobel_edge_detection(img),
        'canny': lambda: canny_edge_detection(img),
        'laplacian': lambda: laplacian_edge_detection(img),
        'erosion': lambda: erosion(img),
        'dilation': lambda: dilation(img),
    }
    
    if operation in operations_map:
        return operations_map[operation]()
    return img


# ==================== BACKGROUND REMOVAL ====================

@app.post("/api/ai/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Remove background from image using GrabCut algorithm."""
    img = decode_image(await file.read())
    
    # Use GrabCut algorithm
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define rectangle (with some margin)
    h, w = img.shape[:2]
    margin = 10
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create mask where foreground is
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask to image
    result = img * mask2[:, :, np.newaxis]
    
    # Add alpha channel
    b, g, r = cv2.split(result)
    alpha = (mask2 * 255).astype(np.uint8)
    result_rgba = cv2.merge([b, g, r, alpha])
    
    # Encode as PNG to preserve transparency
    success, encoded = cv2.imencode(".png", result_rgba)
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/png"
    )


# ==================== OBJECT DETECTION ====================

@app.post("/api/ai/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects using edge-based detection with contours."""
    img = decode_image(await file.read())
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes for significant contours
    result = img.copy()
    objects_found = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, f"Object {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            objects_found.append({"id": i+1, "x": x, "y": y, "width": w, "height": h, "area": area})
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== OCR TEXT EXTRACTION ====================

@app.post("/api/ai/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """Extract text regions from image (prepare for OCR)."""
    img = decode_image(await file.read())
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply MSER for text region detection
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    result = img.copy()
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    
    for hull in hulls:
        x, y, w, h = cv2.boundingRect(hull)
        # Filter by aspect ratio (text regions are usually horizontal)
        if w > h and w > 10 and h > 5:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== IMAGE INPAINTING ====================

@app.post("/api/ai/inpaint")
async def inpaint_image(
    file: UploadFile = File(...),
    mask_data: str = Form(None),  # Base64 encoded mask
    x: int = Query(None),
    y: int = Query(None),
    width: int = Query(50),
    height: int = Query(50)
):
    """Inpaint (remove and fill) a region of the image."""
    img = decode_image(await file.read())
    
    # Create mask - either from provided mask or from coordinates
    if mask_data:
        # Decode base64 mask
        mask_bytes = base64.b64decode(mask_data)
        mask_arr = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
    else:
        # Create mask from coordinates
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if x is not None and y is not None:
            mask[y:y+height, x:x+width] = 255
        else:
            # Default: center region
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            mask[cy-25:cy+25, cx-25:cx+25] = 255
    
    # Apply inpainting
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== PRESET FILTERS (Instagram-style) ====================

@app.post("/api/presets/vintage")
async def preset_vintage(file: UploadFile = File(...)):
    """Apply vintage filter."""
    img = decode_image(await file.read())
    
    # Sepia tone
    sepia_kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    result = cv2.transform(img, sepia_kernel)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Add vignette
    rows, cols = result.shape[:2]
    X = cv2.getGaussianKernel(cols, cols/2)
    Y = cv2.getGaussianKernel(rows, rows/2)
    kernel = Y * X.T
    mask = kernel / kernel.max()
    for i in range(3):
        result[:, :, i] = result[:, :, i] * mask
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/presets/noir")
async def preset_noir(file: UploadFile = File(...)):
    """Apply noir (high contrast B&W) filter."""
    img = decode_image(await file.read())
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    
    # Add slight vignette
    rows, cols = result.shape
    X = cv2.getGaussianKernel(cols, cols/1.5)
    Y = cv2.getGaussianKernel(rows, rows/1.5)
    kernel = Y * X.T
    mask = kernel / kernel.max()
    result = (result * mask).astype(np.uint8)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/presets/warm")
async def preset_warm(file: UploadFile = File(...)):
    """Apply warm color filter."""
    img = decode_image(await file.read())
    
    # Increase red and yellow
    result = img.copy().astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255)  # Red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.1, 0, 255)  # Green (for yellow)
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)  # Reduce blue
    
    return StreamingResponse(
        io.BytesIO(encode_image(result.astype(np.uint8))),
        media_type="image/png"
    )


@app.post("/api/presets/cool")
async def preset_cool(file: UploadFile = File(...)):
    """Apply cool color filter."""
    img = decode_image(await file.read())
    
    # Increase blue
    result = img.copy().astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.2, 0, 255)  # Blue
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)  # Slight green
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.9, 0, 255)  # Reduce red
    
    return StreamingResponse(
        io.BytesIO(encode_image(result.astype(np.uint8))),
        media_type="image/png"
    )


@app.post("/api/presets/dramatic")
async def preset_dramatic(file: UploadFile = File(...)):
    """Apply dramatic filter with high contrast and saturation."""
    img = decode_image(await file.read())
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Increase saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    
    # Convert back
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Increase contrast
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/presets/fade")
async def preset_fade(file: UploadFile = File(...)):
    """Apply faded/washed out filter."""
    img = decode_image(await file.read())
    
    # Reduce contrast and increase brightness
    result = cv2.convertScaleAbs(img, alpha=0.7, beta=60)
    
    # Desaturate slightly
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.7
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== IMAGE COLORIZATION ====================

@app.post("/api/ai/colorize")
async def colorize_image(file: UploadFile = File(...)):
    """Colorize grayscale image using histogram matching."""
    img = decode_image(await file.read())
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply color map for colorization effect
    result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== CUSTOM FILTER BUILDER ====================

@app.post("/api/custom/apply-kernel")
async def apply_custom_kernel(
    file: UploadFile = File(...),
    kernel: str = Query(...)  # JSON array for kernel matrix
):
    """Apply a custom convolution kernel."""
    img = decode_image(await file.read())
    
    try:
        kernel_matrix = np.array(json.loads(kernel), dtype=np.float32)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid kernel format: {e}")
    
    # Ensure kernel is properly shaped
    if len(kernel_matrix.shape) != 2:
        raise HTTPException(status_code=400, detail="Kernel must be a 2D matrix")
    
    # Apply kernel
    result = cv2.filter2D(img, -1, kernel_matrix)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== IMAGE ANNOTATIONS ====================

@app.post("/api/annotate/draw")
async def draw_annotation(
    file: UploadFile = File(...),
    shapes: str = Query(...)  # JSON array of shapes
):
    """Draw annotations on image."""
    img = decode_image(await file.read())
    
    try:
        shapes_list = json.loads(shapes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid shapes JSON")
    
    result = img.copy()
    
    for shape in shapes_list:
        shape_type = shape.get('type', 'rectangle')
        color = tuple(shape.get('color', [0, 255, 0]))
        thickness = shape.get('thickness', 2)
        
        if shape_type == 'rectangle':
            x, y = shape.get('x', 0), shape.get('y', 0)
            w, h = shape.get('width', 50), shape.get('height', 50)
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
        
        elif shape_type == 'circle':
            cx, cy = shape.get('cx', 50), shape.get('cy', 50)
            radius = shape.get('radius', 25)
            cv2.circle(result, (cx, cy), radius, color, thickness)
        
        elif shape_type == 'line':
            x1, y1 = shape.get('x1', 0), shape.get('y1', 0)
            x2, y2 = shape.get('x2', 100), shape.get('y2', 100)
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        
        elif shape_type == 'text':
            text = shape.get('text', 'Text')
            x, y = shape.get('x', 50), shape.get('y', 50)
            font_scale = shape.get('font_scale', 1)
            cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


@app.post("/api/annotate/crop")
async def crop_image(
    file: UploadFile = File(...),
    x: int = Query(0),
    y: int = Query(0),
    width: int = Query(100),
    height: int = Query(100)
):
    """Crop image to specified region."""
    img = decode_image(await file.read())
    
    h, w = img.shape[:2]
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    width = min(width, w - x)
    height = min(height, h - y)
    
    result = img[y:y+height, x:x+width]
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== VIDEO PROCESSING ====================

@app.websocket("/ws/video")
async def video_processing_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing."""
    await websocket.accept()
    
    try:
        while True:
            # Receive frame data and operation
            data = await websocket.receive_json()
            
            frame_data = data.get('frame')
            operation = data.get('operation', 'none')
            params = data.get('params', {})
            
            if frame_data:
                # Decode base64 frame
                frame_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Apply operation
                    processed = apply_operation(frame, operation, params)
                    
                    # Encode and send back
                    _, encoded = cv2.imencode('.jpg', processed)
                    result_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
                    
                    await websocket.send_json({'frame': result_base64})
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ==================== IMAGE COMPARISON ====================

@app.post("/api/compare/diff")
async def image_diff(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """Calculate difference between two images."""
    img1 = decode_image(await file1.read())
    img2 = decode_image(await file2.read())
    
    # Resize to same dimensions if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Highlight differences
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Create colored diff overlay
    result = img1.copy()
    result[thresh > 0] = [0, 0, 255]  # Red for differences
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )


# ==================== HDR EFFECT ====================

@app.post("/api/ai/hdr-effect")
async def hdr_effect(file: UploadFile = File(...)):
    """Apply HDR-like effect."""
    img = decode_image(await file.read())
    
    # Apply detail enhancement
    result = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    
    # Increase local contrast
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Slight saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return StreamingResponse(
        io.BytesIO(encode_image(result)),
        media_type="image/png"
    )
