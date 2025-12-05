"""
Image segmentation algorithms: thresholding, watershed, k-means, region growing.
"""

import cv2
import numpy as np
from typing import Tuple, List


def simple_threshold(img: np.ndarray, threshold: int = 127, 
                      max_value: int = 255) -> np.ndarray:
    """
    Apply simple binary thresholding.
    
    Args:
        img: Input image
        threshold: Threshold value
        max_value: Value to assign when pixel > threshold
        
    Returns:
        Binary image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY)
    return binary


def inverse_threshold(img: np.ndarray, threshold: int = 127, 
                       max_value: int = 255) -> np.ndarray:
    """
    Apply inverse binary thresholding.
    
    Args:
        img: Input image
        threshold: Threshold value
        max_value: Value to assign when pixel <= threshold
        
    Returns:
        Binary image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(img, threshold, max_value, cv2.THRESH_BINARY_INV)
    return binary


def otsu_threshold(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Apply Otsu's automatic thresholding.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (binary image, optimal threshold value)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    threshold, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, int(threshold)


def adaptive_threshold(img: np.ndarray, max_value: int = 255, 
                        method: str = 'gaussian', block_size: int = 11, 
                        c: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding.
    
    Args:
        img: Input image
        max_value: Maximum value to assign
        method: 'mean' or 'gaussian'
        block_size: Size of neighborhood (must be odd)
        c: Constant subtracted from mean/gaussian
        
    Returns:
        Binary image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if block_size % 2 == 0:
        block_size += 1
    
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
    
    return cv2.adaptiveThreshold(img, max_value, adaptive_method, 
                                  cv2.THRESH_BINARY, block_size, c)


def multi_level_threshold(img: np.ndarray, levels: List[int]) -> np.ndarray:
    """
    Apply multi-level thresholding.
    
    Args:
        img: Input image
        levels: List of threshold values
        
    Returns:
        Multi-level segmented image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    result = np.zeros_like(img)
    levels = sorted(levels)
    
    for i, level in enumerate(levels):
        mask = img >= level
        result[mask] = int(255 * (i + 1) / len(levels))
    
    return result


def watershed_segmentation(img: np.ndarray, 
                            markers: np.ndarray = None) -> np.ndarray:
    """
    Apply watershed segmentation.
    
    Args:
        img: Input image (color)
        markers: Optional marker image for seeded watershed
        
    Returns:
        Segmented image with colored regions
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if markers is None:
        # Auto-generate markers using thresholding and morphology
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    
    # Create colored output
    result = img.copy()
    result[markers == -1] = [0, 0, 255]  # Mark boundaries in red
    
    return result


def kmeans_segmentation(img: np.ndarray, k: int = 3, 
                         max_iterations: int = 100) -> np.ndarray:
    """
    Apply K-means clustering for image segmentation.
    
    Args:
        img: Input image
        k: Number of clusters
        max_iterations: Maximum iterations for K-means
        
    Returns:
        Segmented image
    """
    # Reshape image for K-means
    if len(img.shape) == 3:
        pixel_values = img.reshape((-1, 3))
    else:
        pixel_values = img.reshape((-1, 1))
    
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                max_iterations, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 
                                     10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Map labels to centers
    segmented = centers[labels.flatten()]
    
    # Reshape back to original image shape
    return segmented.reshape(img.shape)


def mean_shift_segmentation(img: np.ndarray, spatial_radius: int = 20, 
                             color_radius: int = 40) -> np.ndarray:
    """
    Apply mean shift segmentation.
    
    Args:
        img: Input image (color)
        spatial_radius: Spatial window radius
        color_radius: Color window radius
        
    Returns:
        Segmented image
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius)


def region_growing(img: np.ndarray, seed: Tuple[int, int], 
                    threshold: int = 10) -> np.ndarray:
    """
    Apply region growing segmentation.
    
    Args:
        img: Input image (grayscale)
        seed: Starting point (x, y)
        threshold: Intensity difference threshold
        
    Returns:
        Binary mask of grown region
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    seed_value = int(img[seed[1], seed[0]])
    stack = [seed]
    
    while stack:
        x, y = stack.pop()
        
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if mask[y, x] != 0:
            continue
        
        pixel_value = int(img[y, x])
        if abs(pixel_value - seed_value) <= threshold:
            mask[y, x] = 255
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    
    return mask


def grabcut_segmentation(img: np.ndarray, rect: Tuple[int, int, int, int] = None,
                          iterations: int = 5) -> np.ndarray:
    """
    Apply GrabCut segmentation.
    
    Args:
        img: Input image (color)
        rect: Rectangle (x, y, width, height) containing foreground
        iterations: Number of iterations
        
    Returns:
        Segmented image with background removed
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    if rect is None:
        h, w = img.shape[:2]
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    # Create mask where foreground is 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    return img * mask2[:, :, np.newaxis]


def contour_detection(img: np.ndarray, threshold: int = 127, 
                       mode: str = 'external') -> List[np.ndarray]:
    """
    Detect contours in image.
    
    Args:
        img: Input image
        threshold: Threshold for binarization
        mode: Contour retrieval mode ('external', 'list', 'tree')
        
    Returns:
        List of contours
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    modes = {
        'external': cv2.RETR_EXTERNAL,
        'list': cv2.RETR_LIST,
        'tree': cv2.RETR_TREE
    }
    retrieval_mode = modes.get(mode, cv2.RETR_EXTERNAL)
    
    contours, _ = cv2.findContours(binary, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img: np.ndarray, contours: List[np.ndarray], 
                   color: Tuple[int, int, int] = (0, 255, 0), 
                   thickness: int = 2) -> np.ndarray:
    """
    Draw contours on image.
    
    Args:
        img: Input image
        contours: List of contours
        color: Contour color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn contours
    """
    result = img.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def connected_components(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find connected components in binary image.
    
    Args:
        img: Input binary image
        
    Returns:
        Tuple of (labeled image, number of components)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    num_labels, labels = cv2.connectedComponents(binary)
    
    return labels, num_labels


def flood_fill(img: np.ndarray, seed: Tuple[int, int], 
                new_value: int = 255, tolerance: int = 10) -> np.ndarray:
    """
    Apply flood fill algorithm.
    
    Args:
        img: Input image
        seed: Starting point (x, y)
        new_value: New value to fill
        tolerance: Tolerance for color matching
        
    Returns:
        Flood-filled image
    """
    result = img.copy()
    h, w = result.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    if len(result.shape) == 3:
        cv2.floodFill(result, mask, seed, (new_value, new_value, new_value),
                      (tolerance,) * 3, (tolerance,) * 3)
    else:
        cv2.floodFill(result, mask, seed, new_value, tolerance, tolerance)
    
    return result
