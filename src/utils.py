"""
Utility functions for image processing operations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to the image file
        grayscale: If True, load as grayscale image
        
    Returns:
        Loaded image as numpy array
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    """
    Save an image to file.
    
    Args:
        path: Output path for the image
        img: Image to save
        
    Returns:
        True if successful
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), img)


def show_image(img: np.ndarray, title: str = "Image", cmap: str = None):
    """
    Display an image using matplotlib.
    
    Args:
        img: Image to display
        title: Window title
        cmap: Colormap to use (e.g., 'gray')
    """
    if len(img.shape) == 3:
        # Convert BGR to RGB for display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap=cmap if cmap else ('gray' if len(img.shape) == 2 else None))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_images_side_by_side(images: list, titles: list = None, cols: int = 2):
    """
    Display multiple images side by side.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        cols: Number of columns in the grid
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide empty subplots
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.
    
    Args:
        img: Input image
        
    Returns:
        Normalized image
    """
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img, dtype=np.uint8)
    normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return normalized


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_image_info(img: np.ndarray) -> dict:
    """
    Get information about an image.
    
    Args:
        img: Input image
        
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': img.shape,
        'dtype': str(img.dtype),
        'min': float(img.min()),
        'max': float(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std()),
    }
    
    if len(img.shape) == 2:
        info['channels'] = 1
        info['height'], info['width'] = img.shape
    else:
        info['height'], info['width'], info['channels'] = img.shape
    
    return info


def pad_image(img: np.ndarray, pad_size: int, mode: str = 'constant', value: int = 0) -> np.ndarray:
    """
    Pad an image with specified padding.
    
    Args:
        img: Input image
        pad_size: Number of pixels to pad
        mode: Padding mode ('constant', 'reflect', 'replicate')
        value: Value for constant padding
        
    Returns:
        Padded image
    """
    if mode == 'constant':
        return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, 
                                   cv2.BORDER_CONSTANT, value=value)
    elif mode == 'reflect':
        return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, 
                                   cv2.BORDER_REFLECT)
    elif mode == 'replicate':
        return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, 
                                   cv2.BORDER_REPLICATE)
    return img


def create_kernel(size: int, kernel_type: str = 'average') -> np.ndarray:
    """
    Create a convolution kernel.
    
    Args:
        size: Kernel size (must be odd)
        kernel_type: Type of kernel ('average', 'gaussian', 'sharpen', 'laplacian')
        
    Returns:
        Kernel as numpy array
    """
    if size % 2 == 0:
        size += 1
    
    if kernel_type == 'average':
        return np.ones((size, size), dtype=np.float32) / (size * size)
    
    elif kernel_type == 'gaussian':
        kernel = cv2.getGaussianKernel(size, 0)
        return kernel @ kernel.T
    
    elif kernel_type == 'sharpen':
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return kernel
    
    elif kernel_type == 'laplacian':
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        return kernel
    
    return np.ones((size, size), dtype=np.float32) / (size * size)
