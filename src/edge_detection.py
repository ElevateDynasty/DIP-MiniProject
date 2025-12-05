"""
Edge detection algorithms: Sobel, Canny, Laplacian, Prewitt, Roberts.
"""

import cv2
import numpy as np
from typing import Tuple


def sobel_edge_detection(img: np.ndarray, ksize: int = 3, 
                          direction: str = 'both') -> np.ndarray:
    """
    Apply Sobel edge detection.
    
    Args:
        img: Input image
        ksize: Kernel size (1, 3, 5, or 7)
        direction: 'x', 'y', or 'both'
        
    Returns:
        Edge-detected image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if direction == 'x':
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    elif direction == 'y':
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)


def canny_edge_detection(img: np.ndarray, low_threshold: int = 50, 
                          high_threshold: int = 150, 
                          aperture_size: int = 3) -> np.ndarray:
    """
    Apply Canny edge detection.
    
    Args:
        img: Input image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        aperture_size: Aperture size for Sobel operator
        
    Returns:
        Edge-detected binary image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    return cv2.Canny(blurred, low_threshold, high_threshold, 
                     apertureSize=aperture_size)


def laplacian_edge_detection(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Laplacian edge detection.
    
    Args:
        img: Input image
        ksize: Kernel size for the Laplacian operator
        
    Returns:
        Edge-detected image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)


def laplacian_of_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Laplacian of Gaussian (LoG) edge detection.
    
    Args:
        img: Input image
        sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Edge-detected image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    # Apply Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.clip(np.abs(log), 0, 255).astype(np.uint8)


def prewitt_edge_detection(img: np.ndarray, direction: str = 'both') -> np.ndarray:
    """
    Apply Prewitt edge detection.
    
    Args:
        img: Input image
        direction: 'x', 'y', or 'both'
        
    Returns:
        Edge-detected image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)
    
    if direction == 'x':
        return cv2.filter2D(img, -1, kernel_x)
    elif direction == 'y':
        return cv2.filter2D(img, -1, kernel_y)
    else:
        prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
        magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)


def roberts_edge_detection(img: np.ndarray) -> np.ndarray:
    """
    Apply Roberts cross edge detection.
    
    Args:
        img: Input image
        
    Returns:
        Edge-detected image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)
    
    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)
    
    roberts_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    
    magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def scharr_edge_detection(img: np.ndarray, direction: str = 'both') -> np.ndarray:
    """
    Apply Scharr edge detection (more accurate than Sobel for 3x3 kernel).
    
    Args:
        img: Input image
        direction: 'x', 'y', or 'both'
        
    Returns:
        Edge-detected image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if direction == 'x':
        return cv2.Scharr(img, cv2.CV_64F, 1, 0)
    elif direction == 'y':
        return cv2.Scharr(img, cv2.CV_64F, 0, 1)
    else:
        scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)


def zero_crossing_detection(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Detect zero crossings in an image (typically after LoG).
    
    Args:
        img: Input image (usually LoG filtered)
        threshold: Minimum difference for zero crossing
        
    Returns:
        Binary image with zero crossings
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply LoG
    log = cv2.Laplacian(cv2.GaussianBlur(img, (5, 5), 1), cv2.CV_64F)
    
    # Find zero crossings
    rows, cols = log.shape
    zero_cross = np.zeros_like(log, dtype=np.uint8)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [log[i-1, j], log[i+1, j], log[i, j-1], log[i, j+1]]
            if log[i, j] > 0:
                if any(n < 0 and abs(log[i, j] - n) > threshold for n in neighbors):
                    zero_cross[i, j] = 255
            elif log[i, j] < 0:
                if any(n > 0 and abs(log[i, j] - n) > threshold for n in neighbors):
                    zero_cross[i, j] = 255
    
    return zero_cross


def edge_magnitude_direction(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate edge magnitude and direction using Sobel operators.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (magnitude, direction) arrays
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x)
    
    return magnitude, direction


def auto_canny(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Apply Canny edge detection with automatic threshold detection.
    
    Args:
        img: Input image
        sigma: Threshold calculation parameter
        
    Returns:
        Edge-detected binary image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute median of image
    v = np.median(img)
    
    # Compute thresholds
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    return cv2.Canny(img, lower, upper)


def structured_edge_detection(img: np.ndarray) -> np.ndarray:
    """
    Simple structured edge detection approximation.
    Combines multiple edge detectors for better results.
    
    Args:
        img: Input image
        
    Returns:
        Edge map
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply multiple edge detectors
    canny = cv2.Canny(gray, 50, 150)
    sobel = sobel_edge_detection(gray)
    laplacian = laplacian_edge_detection(gray)
    
    # Combine edge maps
    combined = np.maximum(canny, np.maximum(sobel, laplacian))
    
    return combined
