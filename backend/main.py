"""
FastAPI Backend for DIP Project
REST API for image processing operations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional
import sys
import os

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
