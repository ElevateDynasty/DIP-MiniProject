"""
Image filtering operations: smoothing, sharpening, noise removal.
"""

import cv2
import numpy as np
from typing import Tuple


def average_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply average (mean) filter for image smoothing.
    
    Args:
        img: Input image
        kernel_size: Size of the averaging kernel (must be odd)
        
    Returns:
        Smoothed image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.blur(img, (kernel_size, kernel_size))


def gaussian_filter(img: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian filter for smooth blurring.
    
    Args:
        img: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation. If 0, calculated from kernel size
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def median_filter(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter for noise removal (especially salt-and-pepper noise).
    
    Args:
        img: Input image
        kernel_size: Size of the median filter kernel (must be odd)
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)


def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, 
                      sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving smoothing).
    
    Args:
        img: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def box_filter(img: np.ndarray, kernel_size: int = 5, normalize: bool = True) -> np.ndarray:
    """
    Apply box filter.
    
    Args:
        img: Input image
        kernel_size: Size of the box filter kernel
        normalize: If True, normalize the kernel
        
    Returns:
        Filtered image
    """
    return cv2.boxFilter(img, -1, (kernel_size, kernel_size), normalize=normalize)


def sharpen_filter(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply sharpening filter to enhance edges.
    
    Args:
        img: Input image
        strength: Sharpening strength (1.0 = normal)
        
    Returns:
        Sharpened image
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Adjust kernel based on strength
    kernel = kernel * strength
    kernel[1, 1] = 1 + (kernel[1, 1] - 1) * strength
    
    return cv2.filter2D(img, -1, kernel)


def unsharp_mask(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, 
                  amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """
    Apply unsharp mask for image sharpening.
    
    Args:
        img: Input image
        kernel_size: Size of Gaussian kernel for blurring
        sigma: Gaussian standard deviation
        amount: Sharpening strength
        threshold: Minimum brightness change to apply sharpening
        
    Returns:
        Sharpened image
    """
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.abs(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    
    return sharpened


def non_local_means_denoise(img: np.ndarray, h: float = 10, 
                             template_window_size: int = 7, 
                             search_window_size: int = 21) -> np.ndarray:
    """
    Apply Non-Local Means Denoising.
    
    Args:
        img: Input image
        h: Filter strength. Higher h removes more noise but removes detail too
        template_window_size: Size of template patch (should be odd)
        search_window_size: Size of area where search is performed (should be odd)
        
    Returns:
        Denoised image
    """
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 
                                                template_window_size, search_window_size)
    else:
        return cv2.fastNlMeansDenoising(img, None, h, 
                                         template_window_size, search_window_size)


def emboss_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply emboss filter for 3D effect.
    
    Args:
        img: Input image
        
    Returns:
        Embossed image
    """
    kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def motion_blur(img: np.ndarray, size: int = 15, angle: float = 0) -> np.ndarray:
    """
    Apply motion blur effect.
    
    Args:
        img: Input image
        size: Length of the motion blur
        angle: Angle of motion blur in degrees
        
    Returns:
        Motion blurred image
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = np.ones(size, dtype=np.float32)
    kernel /= size
    
    # Rotate kernel to desired angle
    center = (size // 2, size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
    
    return cv2.filter2D(img, -1, kernel)


def custom_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a custom convolution kernel.
    
    Args:
        img: Input image
        kernel: Custom kernel as numpy array
        
    Returns:
        Filtered image
    """
    return cv2.filter2D(img, -1, kernel)


def add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
    """
    Add Gaussian noise to image.
    
    Args:
        img: Input image
        mean: Mean of the noise
        std: Standard deviation of the noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, salt_prob: float = 0.01, 
                           pepper_prob: float = 0.01) -> np.ndarray:
    """
    Add salt and pepper noise to image.
    
    Args:
        img: Input image
        salt_prob: Probability of salt noise
        pepper_prob: Probability of pepper noise
        
    Returns:
        Noisy image
    """
    output = img.copy()
    
    # Salt noise
    salt_mask = np.random.random(img.shape[:2]) < salt_prob
    if len(img.shape) == 3:
        output[salt_mask] = [255, 255, 255]
    else:
        output[salt_mask] = 255
    
    # Pepper noise
    pepper_mask = np.random.random(img.shape[:2]) < pepper_prob
    if len(img.shape) == 3:
        output[pepper_mask] = [0, 0, 0]
    else:
        output[pepper_mask] = 0
    
    return output


def add_speckle_noise(img: np.ndarray, variance: float = 0.04) -> np.ndarray:
    """
    Add speckle noise to image.
    
    Args:
        img: Input image
        variance: Variance of the noise
        
    Returns:
        Noisy image
    """
    noise = np.random.randn(*img.shape) * np.sqrt(variance)
    noisy = img.astype(np.float32) + img.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def wiener_filter(img: np.ndarray, noise_variance: float = 0.01) -> np.ndarray:
    """
    Apply Wiener filter for noise reduction (simplified implementation).
    
    Args:
        img: Input image (grayscale)
        noise_variance: Estimated noise variance
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to frequency domain
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    
    # Estimate signal power spectrum
    signal_power = np.abs(fshift) ** 2
    
    # Wiener filter
    wiener = signal_power / (signal_power + noise_variance * np.prod(img.shape))
    filtered = fshift * wiener
    
    # Convert back to spatial domain
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return np.clip(img_back, 0, 255).astype(np.uint8)
