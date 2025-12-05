"""
Deep learning based image processing: face detection, object detection, etc.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import os


class FaceDetector:
    """Face detection using OpenCV's DNN module or Haar Cascades."""
    
    def __init__(self, method: str = 'haar'):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('haar', 'dnn')
        """
        self.method = method
        
        if method == 'haar':
            # Use Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            # Use DNN-based detector (requires model files)
            self.detector = None
    
    def detect(self, img: np.ndarray, scale_factor: float = 1.1,
               min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple]:
        """
        Detect faces in image.
        
        Args:
            img: Input image
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        if self.method == 'haar':
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            return list(faces)
        
        return []
    
    def draw_detections(self, img: np.ndarray, faces: List,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """
        Draw face detections on image.
        
        Args:
            img: Input image
            faces: List of face bounding boxes
            color: Rectangle color
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        return result


class EyeDetector:
    """Eye detection using Haar Cascades."""
    
    def __init__(self):
        """Initialize eye detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, img: np.ndarray, face_regions: List = None) -> List[Tuple]:
        """
        Detect eyes in image.
        
        Args:
            img: Input image
            face_regions: Optional list of face regions to search within
            
        Returns:
            List of eye bounding boxes (x, y, w, h)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        all_eyes = []
        
        if face_regions:
            for (fx, fy, fw, fh) in face_regions:
                roi = gray[fy:fy+fh, fx:fx+fw]
                eyes = self.detector.detectMultiScale(roi)
                for (ex, ey, ew, eh) in eyes:
                    all_eyes.append((fx + ex, fy + ey, ew, eh))
        else:
            eyes = self.detector.detectMultiScale(gray)
            all_eyes = list(eyes)
        
        return all_eyes


def detect_faces_haar(img: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Detect faces using Haar Cascade.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (image with faces marked, list of face regions)
    """
    detector = FaceDetector(method='haar')
    faces = detector.detect(img)
    result = detector.draw_detections(img, faces)
    return result, faces


def detect_eyes(img: np.ndarray, detect_faces_first: bool = True) -> Tuple[np.ndarray, List]:
    """
    Detect eyes in image.
    
    Args:
        img: Input image
        detect_faces_first: If True, detect faces first and search within
        
    Returns:
        Tuple of (image with eyes marked, list of eye regions)
    """
    result = img.copy()
    
    face_regions = None
    if detect_faces_first:
        face_detector = FaceDetector()
        face_regions = face_detector.detect(img)
    
    eye_detector = EyeDetector()
    eyes = eye_detector.detect(img, face_regions)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return result, eyes


def detect_smile(img: np.ndarray, face_regions: List = None) -> Tuple[np.ndarray, List]:
    """
    Detect smiles in image.
    
    Args:
        img: Input image
        face_regions: Optional list of face regions
        
    Returns:
        Tuple of (image with smiles marked, list of smile regions)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    if face_regions is None:
        face_detector = FaceDetector()
        face_regions = face_detector.detect(img)
    
    all_smiles = []
    for (fx, fy, fw, fh) in face_regions:
        roi = gray[fy:fy+fh, fx:fx+fw]
        smiles = smile_cascade.detectMultiScale(roi, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            all_smiles.append((fx + sx, fy + sy, sw, sh))
            cv2.rectangle(result, (fx + sx, fy + sy), (fx + sx + sw, fy + sy + sh), (0, 255, 255), 2)
    
    return result, all_smiles


def detect_pedestrians(img: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Detect pedestrians using HOG descriptor.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (image with pedestrians marked, list of detections)
    """
    result = img.copy()
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect pedestrians
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(4, 4), scale=1.05)
    
    for (x, y, w, h) in boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result, list(boxes)


def detect_cars(img: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Detect cars using Haar Cascade (if available).
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (image with cars marked, list of detections)
    """
    result = img.copy()
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Try to load car cascade (may not be available in all OpenCV builds)
    try:
        car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        if car_cascade.empty():
            # Fallback: use frontal face cascade as placeholder
            return result, []
        
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result, list(cars)
    except:
        return result, []


def neural_style_transfer_simple(content_img: np.ndarray, 
                                  style: str = 'pencil') -> np.ndarray:
    """
    Apply simple artistic style transfer effects.
    
    Args:
        content_img: Input image
        style: Style to apply ('pencil', 'cartoon', 'watercolor', 'oil')
        
    Returns:
        Styled image
    """
    if style == 'pencil':
        # Pencil sketch effect
        gray = cv2.cvtColor(content_img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    elif style == 'cartoon':
        # Cartoon effect
        gray = cv2.cvtColor(content_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(content_img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    elif style == 'watercolor':
        # Watercolor effect
        result = cv2.stylization(content_img, sigma_s=60, sigma_r=0.6)
        return result
    
    elif style == 'oil':
        # Oil painting effect using bilateral filter
        result = content_img.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        return result
    
    return content_img


def edge_preserving_filter(img: np.ndarray, sigma_s: float = 60, 
                            sigma_r: float = 0.4,
                            mode: str = 'recursive') -> np.ndarray:
    """
    Apply edge-preserving smoothing filter.
    
    Args:
        img: Input image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
        mode: Filter mode ('recursive', 'normalized')
        
    Returns:
        Filtered image
    """
    flags = cv2.RECURS_FILTER if mode == 'recursive' else cv2.NORMCONV_FILTER
    return cv2.edgePreservingFilter(img, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)


def detail_enhance(img: np.ndarray, sigma_s: float = 10, 
                    sigma_r: float = 0.15) -> np.ndarray:
    """
    Enhance image details.
    
    Args:
        img: Input image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
        
    Returns:
        Detail-enhanced image
    """
    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)


def pencil_sketch(img: np.ndarray, sigma_s: float = 60, 
                   sigma_r: float = 0.07, shade_factor: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pencil sketch effect.
    
    Args:
        img: Input image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
        shade_factor: Shade factor for grayscale sketch
        
    Returns:
        Tuple of (grayscale sketch, color sketch)
    """
    sketch_gray, sketch_color = cv2.pencilSketch(img, sigma_s=sigma_s, 
                                                  sigma_r=sigma_r, 
                                                  shade_factor=shade_factor)
    return sketch_gray, sketch_color


def stylization(img: np.ndarray, sigma_s: float = 60, 
                 sigma_r: float = 0.45) -> np.ndarray:
    """
    Apply stylization effect (like cartoon).
    
    Args:
        img: Input image
        sigma_s: Spatial sigma
        sigma_r: Range sigma
        
    Returns:
        Stylized image
    """
    return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)


def inpaint_image(img: np.ndarray, mask: np.ndarray, 
                   radius: int = 3, method: str = 'telea') -> np.ndarray:
    """
    Inpaint (fill in) regions of image.
    
    Args:
        img: Input image
        mask: Mask where non-zero pixels indicate regions to inpaint
        radius: Neighborhood radius for inpainting
        method: Inpainting method ('telea', 'ns')
        
    Returns:
        Inpainted image
    """
    if method == 'telea':
        flags = cv2.INPAINT_TELEA
    else:
        flags = cv2.INPAINT_NS
    
    return cv2.inpaint(img, mask, radius, flags)


def seamless_clone(src: np.ndarray, dst: np.ndarray, mask: np.ndarray,
                    center: Tuple[int, int], mode: str = 'normal') -> np.ndarray:
    """
    Seamlessly clone source into destination.
    
    Args:
        src: Source image
        dst: Destination image
        mask: Mask indicating region to clone
        center: Center point in destination
        mode: Cloning mode ('normal', 'mixed', 'monochrome')
        
    Returns:
        Cloned result
    """
    modes = {
        'normal': cv2.NORMAL_CLONE,
        'mixed': cv2.MIXED_CLONE,
        'monochrome': cv2.MONOCHROME_TRANSFER
    }
    clone_mode = modes.get(mode, cv2.NORMAL_CLONE)
    
    return cv2.seamlessClone(src, dst, mask, center, clone_mode)


def colorize_image(img: np.ndarray) -> np.ndarray:
    """
    Colorize a grayscale image using simple heuristics.
    Note: For better results, use deep learning models.
    
    Args:
        img: Input grayscale image
        
    Returns:
        Colorized image (simple approximation)
    """
    if len(img.shape) == 3:
        return img
    
    # Simple colorization by applying a color map
    colorized = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    
    return colorized


def super_resolution_simple(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Simple super-resolution using interpolation.
    For better results, use deep learning models.
    
    Args:
        img: Input image
        scale: Scale factor
        
    Returns:
        Upscaled image
    """
    # Use Lanczos interpolation for best quality
    h, w = img.shape[:2]
    new_size = (w * scale, h * scale)
    
    upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Apply slight sharpening to enhance details
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ], dtype=np.float32)
    
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise_deep(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Denoise image using advanced methods.
    
    Args:
        img: Input image
        strength: Denoising strength
        
    Returns:
        Denoised image
    """
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
