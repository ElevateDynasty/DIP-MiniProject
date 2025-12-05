"""
Morphological operations: erosion, dilation, opening, closing, skeleton.
"""

import cv2
import numpy as np
from typing import Tuple


def get_structuring_element(shape: str = 'rect', size: int = 5) -> np.ndarray:
    """
    Get a structuring element for morphological operations.
    
    Args:
        shape: Shape of the element ('rect', 'ellipse', 'cross')
        size: Size of the element
        
    Returns:
        Structuring element
    """
    shapes = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    morph_shape = shapes.get(shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (size, size))


def erosion(img: np.ndarray, kernel_size: int = 5, iterations: int = 1,
             kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply erosion operation.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        iterations: Number of times to apply erosion
        kernel_shape: Shape of kernel ('rect', 'ellipse', 'cross')
        
    Returns:
        Eroded image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.erode(img, kernel, iterations=iterations)


def dilation(img: np.ndarray, kernel_size: int = 5, iterations: int = 1,
              kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply dilation operation.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        iterations: Number of times to apply dilation
        kernel_shape: Shape of kernel ('rect', 'ellipse', 'cross')
        
    Returns:
        Dilated image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.dilate(img, kernel, iterations=iterations)


def opening(img: np.ndarray, kernel_size: int = 5, 
             kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply opening operation (erosion followed by dilation).
    Removes small bright spots (noise) from dark background.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel ('rect', 'ellipse', 'cross')
        
    Returns:
        Opened image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def closing(img: np.ndarray, kernel_size: int = 5, 
             kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply closing operation (dilation followed by erosion).
    Removes small dark spots (holes) from bright objects.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel ('rect', 'ellipse', 'cross')
        
    Returns:
        Closed image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def morphological_gradient(img: np.ndarray, kernel_size: int = 5,
                            kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply morphological gradient (difference between dilation and erosion).
    Produces an outline of the object.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel
        
    Returns:
        Gradient image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def top_hat(img: np.ndarray, kernel_size: int = 9, 
             kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply top hat transformation (difference between image and opening).
    Extracts small bright elements on dark background.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel
        
    Returns:
        Top hat image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def black_hat(img: np.ndarray, kernel_size: int = 9, 
               kernel_shape: str = 'rect') -> np.ndarray:
    """
    Apply black hat transformation (difference between closing and image).
    Extracts small dark elements on bright background.
    
    Args:
        img: Input image
        kernel_size: Size of the structuring element
        kernel_shape: Shape of kernel
        
    Returns:
        Black hat image
    """
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def hit_or_miss(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Apply hit-or-miss transform for pattern detection.
    
    Args:
        img: Input binary image
        kernel: Structuring element with 1, 0, -1 values
        
    Returns:
        Hit-or-miss result
    """
    if kernel is None:
        # Default kernel to detect isolated points
        kernel = np.array([
            [0, -1, 0],
            [-1, 1, -1],
            [0, -1, 0]
        ], dtype=np.int8)
    
    return cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)


def skeletonization(img: np.ndarray) -> np.ndarray:
    """
    Apply skeletonization to extract the skeleton of shapes.
    
    Args:
        img: Input binary image
        
    Returns:
        Skeleton image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        
        if cv2.countNonZero(binary) == 0:
            break
    
    return skeleton


def thinning(img: np.ndarray) -> np.ndarray:
    """
    Apply morphological thinning (Zhang-Suen algorithm via OpenCV).
    
    Args:
        img: Input binary image
        
    Returns:
        Thinned image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    return cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else skeletonization(binary)


def boundary_extraction(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Extract boundary of objects using morphological operations.
    
    Args:
        img: Input binary image
        kernel_size: Size of erosion kernel
        
    Returns:
        Boundary image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel = get_structuring_element('rect', kernel_size)
    eroded = cv2.erode(img, kernel)
    boundary = cv2.subtract(img, eroded)
    
    return boundary


def hole_filling(img: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary objects.
    
    Args:
        img: Input binary image
        
    Returns:
        Image with holes filled
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Create a mask slightly larger than the image
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from point (0, 0)
    filled = binary.copy()
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # Invert to get holes
    filled_inv = cv2.bitwise_not(filled)
    
    # Combine with original
    return binary | filled_inv


def remove_small_objects(img: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Remove small connected components.
    
    Args:
        img: Input binary image
        min_size: Minimum number of pixels to keep
        
    Returns:
        Image with small objects removed
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output image
    result = np.zeros_like(binary)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255
    
    return result


def remove_small_holes(img: np.ndarray, max_size: int = 100) -> np.ndarray:
    """
    Remove small holes (dark regions) from binary image.
    
    Args:
        img: Input binary image
        max_size: Maximum hole size to fill
        
    Returns:
        Image with small holes filled
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Invert to find holes
    inverted = cv2.bitwise_not(binary)
    
    # Find connected components in inverted image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    # Fill small holes
    result = binary.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_size:
            result[labels == i] = 255
    
    return result


def geodesic_dilation(marker: np.ndarray, mask: np.ndarray, 
                       iterations: int = -1) -> np.ndarray:
    """
    Perform geodesic dilation (dilation constrained by mask).
    
    Args:
        marker: Initial marker image
        mask: Constraint mask
        iterations: Number of iterations (-1 for until convergence)
        
    Returns:
        Result of geodesic dilation
    """
    kernel = get_structuring_element('rect', 3)
    
    if iterations == -1:
        # Iterate until convergence
        prev = np.zeros_like(marker)
        result = marker.copy()
        while not np.array_equal(prev, result):
            prev = result.copy()
            result = cv2.dilate(result, kernel)
            result = cv2.bitwise_and(result, mask)
        return result
    else:
        result = marker.copy()
        for _ in range(iterations):
            result = cv2.dilate(result, kernel)
            result = cv2.bitwise_and(result, mask)
        return result


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Perform morphological reconstruction by dilation.
    
    Args:
        marker: Marker image (must be <= mask everywhere)
        mask: Mask image
        
    Returns:
        Reconstructed image
    """
    return geodesic_dilation(marker, mask, iterations=-1)
