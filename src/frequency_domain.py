"""
Frequency domain operations: FFT, DFT, filtering in frequency domain.
"""

import cv2
import numpy as np
from typing import Tuple


def compute_dft(img: np.ndarray) -> np.ndarray:
    """
    Compute 2D Discrete Fourier Transform.
    
    Args:
        img: Input image (grayscale)
        
    Returns:
        Complex DFT result (shifted to center)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Shift zero frequency to center
    dft_shift = np.fft.fftshift(dft)
    
    return dft_shift


def compute_inverse_dft(dft_shift: np.ndarray) -> np.ndarray:
    """
    Compute inverse DFT to get back spatial domain image.
    
    Args:
        dft_shift: Shifted DFT result
        
    Returns:
        Spatial domain image
    """
    # Inverse shift
    f_ishift = np.fft.ifftshift(dft_shift)
    
    # Inverse DFT
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    return np.clip(img_back, 0, 255).astype(np.uint8)


def get_magnitude_spectrum(dft_shift: np.ndarray) -> np.ndarray:
    """
    Get magnitude spectrum from DFT.
    
    Args:
        dft_shift: Shifted DFT result
        
    Returns:
        Magnitude spectrum (log-scaled for visualization)
    """
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    # Log scale for better visualization
    magnitude_spectrum = 20 * np.log(magnitude + 1)
    
    # Normalize to 0-255
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, 
                                        cv2.NORM_MINMAX).astype(np.uint8)
    
    return magnitude_spectrum


def get_phase_spectrum(dft_shift: np.ndarray) -> np.ndarray:
    """
    Get phase spectrum from DFT.
    
    Args:
        dft_shift: Shifted DFT result
        
    Returns:
        Phase spectrum
    """
    phase = np.arctan2(dft_shift[:, :, 1], dft_shift[:, :, 0])
    
    # Normalize to 0-255
    phase_spectrum = cv2.normalize(phase, None, 0, 255, 
                                    cv2.NORM_MINMAX).astype(np.uint8)
    
    return phase_spectrum


def ideal_lowpass_filter(img: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    Apply ideal low-pass filter in frequency domain.
    
    Args:
        img: Input image
        cutoff: Cutoff frequency (radius in pixels)
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create ideal low-pass filter mask
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def ideal_highpass_filter(img: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    Apply ideal high-pass filter in frequency domain.
    
    Args:
        img: Input image
        cutoff: Cutoff frequency (radius in pixels)
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create ideal high-pass filter mask
    mask = np.ones((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff:
                mask[i, j] = 0
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def butterworth_lowpass_filter(img: np.ndarray, cutoff: int = 30, 
                                order: int = 2) -> np.ndarray:
    """
    Apply Butterworth low-pass filter in frequency domain.
    
    Args:
        img: Input image
        cutoff: Cutoff frequency
        order: Order of the filter (higher = sharper cutoff)
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Butterworth low-pass filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance = np.sqrt(u ** 2 + v ** 2)
    
    # Avoid division by zero
    distance[crow, ccol] = 1
    
    h = 1 / (1 + (distance / cutoff) ** (2 * order))
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def butterworth_highpass_filter(img: np.ndarray, cutoff: int = 30, 
                                 order: int = 2) -> np.ndarray:
    """
    Apply Butterworth high-pass filter in frequency domain.
    
    Args:
        img: Input image
        cutoff: Cutoff frequency
        order: Order of the filter
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Butterworth high-pass filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance = np.sqrt(u ** 2 + v ** 2)
    
    # Avoid division by zero
    distance[crow, ccol] = 1e-10
    
    h = 1 / (1 + (cutoff / distance) ** (2 * order))
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def gaussian_lowpass_filter(img: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    Apply Gaussian low-pass filter in frequency domain.
    
    Args:
        img: Input image
        cutoff: Cutoff frequency (standard deviation)
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Gaussian low-pass filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance_sq = u ** 2 + v ** 2
    
    h = np.exp(-distance_sq / (2 * cutoff ** 2))
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def gaussian_highpass_filter(img: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """
    Apply Gaussian high-pass filter in frequency domain.
    
    Args:
        img: Input image
        cutoff: Cutoff frequency (standard deviation)
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Gaussian high-pass filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance_sq = u ** 2 + v ** 2
    
    h = 1 - np.exp(-distance_sq / (2 * cutoff ** 2))
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def bandpass_filter(img: np.ndarray, low_cutoff: int = 10, 
                     high_cutoff: int = 50) -> np.ndarray:
    """
    Apply band-pass filter in frequency domain.
    
    Args:
        img: Input image
        low_cutoff: Lower cutoff frequency
        high_cutoff: Higher cutoff frequency
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create band-pass filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance = np.sqrt(u ** 2 + v ** 2)
    
    h = np.logical_and(distance >= low_cutoff, distance <= high_cutoff).astype(np.float32)
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def bandreject_filter(img: np.ndarray, low_cutoff: int = 10, 
                       high_cutoff: int = 50) -> np.ndarray:
    """
    Apply band-reject (notch) filter in frequency domain.
    
    Args:
        img: Input image
        low_cutoff: Lower cutoff frequency
        high_cutoff: Higher cutoff frequency
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create band-reject filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance = np.sqrt(u ** 2 + v ** 2)
    
    h = np.logical_or(distance < low_cutoff, distance > high_cutoff).astype(np.float32)
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)


def homomorphic_filter(img: np.ndarray, gamma_l: float = 0.5, 
                        gamma_h: float = 2.0, cutoff: int = 30) -> np.ndarray:
    """
    Apply homomorphic filtering for illumination correction.
    
    Args:
        img: Input image
        gamma_l: Low frequency gain (< 1 to reduce)
        gamma_h: High frequency gain (> 1 to enhance)
        cutoff: Cutoff frequency
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Take log
    img_log = np.log1p(np.float32(img))
    
    # DFT
    dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create homomorphic filter
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    distance_sq = u ** 2 + v ** 2
    
    h = (gamma_h - gamma_l) * (1 - np.exp(-distance_sq / (2 * cutoff ** 2))) + gamma_l
    mask = np.stack([h, h], axis=-1).astype(np.float32)
    
    # Apply filter
    filtered = dft_shift * mask
    
    # Inverse DFT
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Exponential to reverse log
    img_exp = np.expm1(img_back)
    
    return cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def power_spectrum(img: np.ndarray) -> np.ndarray:
    """
    Compute power spectrum of image.
    
    Args:
        img: Input image
        
    Returns:
        Power spectrum
    """
    dft_shift = compute_dft(img)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    power = magnitude ** 2
    
    # Log scale
    power_spectrum = 10 * np.log10(power + 1)
    
    return cv2.normalize(power_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def notch_filter(img: np.ndarray, notch_points: list, radius: int = 10) -> np.ndarray:
    """
    Apply notch filter to remove specific frequencies.
    
    Args:
        img: Input image
        notch_points: List of (row, col) points to remove (relative to center)
        radius: Radius of notch
        
    Returns:
        Filtered image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create notch filter mask
    mask = np.ones((rows, cols, 2), np.float32)
    
    for dr, dc in notch_points:
        for i in range(rows):
            for j in range(cols):
                # Distance from notch point
                d1 = np.sqrt((i - (crow + dr)) ** 2 + (j - (ccol + dc)) ** 2)
                # Distance from conjugate point
                d2 = np.sqrt((i - (crow - dr)) ** 2 + (j - (ccol - dc)) ** 2)
                
                if d1 <= radius or d2 <= radius:
                    mask[i, j] = 0
    
    # Apply filter
    dft_shift = compute_dft(img)
    filtered = dft_shift * mask
    
    return compute_inverse_dft(filtered)
