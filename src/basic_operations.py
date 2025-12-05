"""
Basic image operations: resize, rotate, flip, crop, transform.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(img: np.ndarray, scale: float = None, width: int = None, 
                  height: int = None, interpolation: str = 'bilinear') -> np.ndarray:
    """
    Resize image by scale or to specific dimensions.
    
    Args:
        img: Input image
        scale: Scale factor (e.g., 0.5 for half size, 2.0 for double)
        width: Target width in pixels
        height: Target height in pixels
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic', 'lanczos')
        
    Returns:
        Resized image
    """
    interp_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'area': cv2.INTER_AREA
    }
    interp = interp_methods.get(interpolation, cv2.INTER_LINEAR)
    
    if scale is not None:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    elif width is not None and height is not None:
        return cv2.resize(img, (width, height), interpolation=interp)
    elif width is not None:
        aspect_ratio = img.shape[0] / img.shape[1]
        new_height = int(width * aspect_ratio)
        return cv2.resize(img, (width, new_height), interpolation=interp)
    elif height is not None:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(height * aspect_ratio)
        return cv2.resize(img, (new_width, height), interpolation=interp)
    return img


def pixel_replication(img: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Enlarge image using pixel replication (nearest neighbor).
    
    Args:
        img: Input image
        factor: Enlargement factor
        
    Returns:
        Enlarged image
    """
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def bilinear_interpolation(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    Enlarge image using bilinear interpolation.
    
    Args:
        img: Input image
        factor: Enlargement factor
        
    Returns:
        Enlarged image
    """
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)


def bicubic_interpolation(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    Enlarge image using bicubic interpolation.
    
    Args:
        img: Input image
        factor: Enlargement factor
        
    Returns:
        Enlarged image
    """
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def rotate_image(img: np.ndarray, angle: float, center: Tuple[int, int] = None,
                  scale: float = 1.0, expand: bool = False) -> np.ndarray:
    """
    Rotate image by given angle.
    
    Args:
        img: Input image
        angle: Rotation angle in degrees (counter-clockwise)
        center: Center point for rotation (default: image center)
        scale: Scale factor during rotation
        expand: If True, expand output to fit entire rotated image
        
    Returns:
        Rotated image
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    if expand:
        # Calculate new image size to fit rotated image
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(img, matrix, (new_w, new_h))
    
    return cv2.warpAffine(img, matrix, (w, h))


def flip_image(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Flip image horizontally, vertically, or both.
    
    Args:
        img: Input image
        direction: Flip direction ('horizontal', 'vertical', 'both')
        
    Returns:
        Flipped image
    """
    if direction == 'horizontal':
        return cv2.flip(img, 1)
    elif direction == 'vertical':
        return cv2.flip(img, 0)
    elif direction == 'both':
        return cv2.flip(img, -1)
    return img


def translate_image(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Translate (shift) image by given amount.
    
    Args:
        img: Input image
        tx: Translation in x direction (pixels)
        ty: Translation in y direction (pixels)
        
    Returns:
        Translated image
    """
    h, w = img.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, matrix, (w, h))


def shear_image(img: np.ndarray, shear_x: float = 0, shear_y: float = 0) -> np.ndarray:
    """
    Apply shear transformation to image.
    
    Args:
        img: Input image
        shear_x: Shear factor in x direction
        shear_y: Shear factor in y direction
        
    Returns:
        Sheared image
    """
    h, w = img.shape[:2]
    matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])
    
    # Calculate new dimensions
    new_w = int(w + abs(shear_x) * h)
    new_h = int(h + abs(shear_y) * w)
    
    return cv2.warpAffine(img, matrix, (new_w, new_h))


def crop_image(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop a region from image.
    
    Args:
        img: Input image
        x: Starting x coordinate
        y: Starting y coordinate
        width: Width of crop region
        height: Height of crop region
        
    Returns:
        Cropped image
    """
    return img[y:y+height, x:x+width].copy()


def negative_image(img: np.ndarray) -> np.ndarray:
    """
    Create negative of image.
    
    Args:
        img: Input image
        
    Returns:
        Negative image
    """
    return 255 - img


def adjust_brightness(img: np.ndarray, value: int) -> np.ndarray:
    """
    Adjust image brightness.
    
    Args:
        img: Input image
        value: Brightness adjustment (-255 to 255)
        
    Returns:
        Brightness-adjusted image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if len(img.shape) == 3 else img
    
    if len(img.shape) == 3:
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image contrast.
    
    Args:
        img: Input image
        factor: Contrast factor (1.0 = no change, >1 = more contrast)
        
    Returns:
        Contrast-adjusted image
    """
    mean = np.mean(img)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to image.
    
    Args:
        img: Input image
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        
    Returns:
        Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def histogram_equalization(img: np.ndarray, use_clahe: bool = False, 
                            clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    
    Args:
        img: Input image
        use_clahe: If True, use Contrast Limited Adaptive Histogram Equalization
        clip_limit: Threshold for contrast limiting (for CLAHE)
        tile_size: Size of grid for histogram equalization (for CLAHE)
        
    Returns:
        Histogram-equalized image
    """
    if len(img.shape) == 3:
        # Convert to LAB color space for color images
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            l = clahe.apply(l)
        else:
            l = cv2.equalizeHist(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            return clahe.apply(img)
        else:
            return cv2.equalizeHist(img)


def perspective_transform(img: np.ndarray, src_points: np.ndarray, 
                          dst_points: np.ndarray, output_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Apply perspective transformation to image.
    
    Args:
        img: Input image
        src_points: Source points (4 corners) as numpy array of shape (4, 2)
        dst_points: Destination points (4 corners) as numpy array of shape (4, 2)
        output_size: Output image size (width, height)
        
    Returns:
        Transformed image
    """
    if output_size is None:
        output_size = (img.shape[1], img.shape[0])
    
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, matrix, output_size)


def affine_transform(img: np.ndarray, src_points: np.ndarray, 
                      dst_points: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to image.
    
    Args:
        img: Input image
        src_points: Source points (3 points) as numpy array of shape (3, 2)
        dst_points: Destination points (3 points) as numpy array of shape (3, 2)
        
    Returns:
        Transformed image
    """
    h, w = img.shape[:2]
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    
    matrix = cv2.getAffineTransform(src_points, dst_points)
    return cv2.warpAffine(img, matrix, (w, h))
