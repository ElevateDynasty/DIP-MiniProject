"""
Feature detection and matching: ORB, SIFT, SURF, Harris corners, etc.
"""

import cv2
import numpy as np
from typing import Tuple, List


def detect_harris_corners(img: np.ndarray, block_size: int = 2, 
                           ksize: int = 3, k: float = 0.04,
                           threshold: float = 0.01) -> np.ndarray:
    """
    Detect corners using Harris corner detector.
    
    Args:
        img: Input image
        block_size: Neighborhood size for corner detection
        ksize: Aperture parameter for Sobel operator
        k: Harris detector free parameter
        threshold: Threshold for corner response (fraction of max)
        
    Returns:
        Image with corners marked
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.copy()
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    gray = np.float32(gray)
    
    # Harris corner detection
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilate for marking corners
    dst = cv2.dilate(dst, None)
    
    # Mark corners
    result[dst > threshold * dst.max()] = [0, 0, 255]
    
    return result


def detect_shi_tomasi_corners(img: np.ndarray, max_corners: int = 100,
                               quality_level: float = 0.01,
                               min_distance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect corners using Shi-Tomasi (Good Features to Track).
    
    Args:
        img: Input image
        max_corners: Maximum number of corners to detect
        quality_level: Minimal accepted quality of corners
        min_distance: Minimum Euclidean distance between corners
        
    Returns:
        Tuple of (image with corners, corner points array)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.copy()
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    
    if corners is not None:
        corners = np.intp(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
    
    return result, corners


def detect_orb_features(img: np.ndarray, n_features: int = 500) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Detect ORB (Oriented FAST and Rotated BRIEF) features.
    
    Args:
        img: Input image
        n_features: Maximum number of features to detect
        
    Returns:
        Tuple of (image with keypoints, keypoints, descriptors)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Draw keypoints
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return result, keypoints, descriptors


def detect_sift_features(img: np.ndarray, n_features: int = 0) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Detect SIFT (Scale-Invariant Feature Transform) features.
    
    Args:
        img: Input image
        n_features: Maximum number of features (0 = no limit)
        
    Returns:
        Tuple of (image with keypoints, keypoints, descriptors)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return result, keypoints, descriptors


def detect_fast_features(img: np.ndarray, threshold: int = 10,
                          non_max_suppression: bool = True) -> Tuple[np.ndarray, list]:
    """
    Detect FAST (Features from Accelerated Segment Test) features.
    
    Args:
        img: Input image
        threshold: Threshold for corner detection
        non_max_suppression: Whether to apply non-maximum suppression
        
    Returns:
        Tuple of (image with keypoints, keypoints)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Create FAST detector
    fast = cv2.FastFeatureDetector_create(threshold=threshold, 
                                           nonmaxSuppression=non_max_suppression)
    
    # Detect keypoints
    keypoints = fast.detect(gray, None)
    
    # Draw keypoints
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    
    return result, keypoints


def detect_brief_features(img: np.ndarray) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Detect features using STAR detector with BRIEF descriptors.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (image with keypoints, keypoints, descriptors)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Use ORB for detection (STAR not available in newer OpenCV)
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    
    # BRIEF descriptor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() if hasattr(cv2, 'xfeatures2d') else orb
    keypoints, descriptors = brief.compute(gray, keypoints)
    
    # Draw keypoints
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    
    return result, keypoints, descriptors


def match_features_bf(desc1: np.ndarray, desc2: np.ndarray, 
                       descriptor_type: str = 'orb') -> List:
    """
    Match features using Brute-Force matcher.
    
    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        descriptor_type: Type of descriptors ('orb', 'sift')
        
    Returns:
        List of matches
    """
    if desc1 is None or desc2 is None:
        return []
    
    if descriptor_type == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches


def match_features_flann(desc1: np.ndarray, desc2: np.ndarray,
                          descriptor_type: str = 'sift') -> List:
    """
    Match features using FLANN (Fast Library for Approximate Nearest Neighbors).
    
    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        descriptor_type: Type of descriptors ('orb', 'sift')
        
    Returns:
        List of good matches
    """
    if desc1 is None or desc2 is None:
        return []
    
    if descriptor_type == 'orb':
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Need at least 2 matches for ratio test
    if len(desc1) < 2 or len(desc2) < 2:
        return []
    
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    return good_matches


def draw_matches(img1: np.ndarray, kp1: list, img2: np.ndarray, 
                  kp2: list, matches: list, max_matches: int = 50) -> np.ndarray:
    """
    Draw matches between two images.
    
    Args:
        img1: First image
        kp1: Keypoints from first image
        img2: Second image
        kp2: Keypoints from second image
        matches: List of matches
        max_matches: Maximum matches to draw
        
    Returns:
        Image with matches drawn
    """
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches],
                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def find_homography(kp1: list, kp2: list, matches: list,
                     reproj_threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find homography matrix between two sets of matched keypoints.
    
    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: List of matches
        reproj_threshold: Reprojection threshold for RANSAC
        
    Returns:
        Tuple of (homography matrix, mask of inliers)
    """
    if len(matches) < 4:
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
    
    return H, mask


def detect_blobs(img: np.ndarray, min_threshold: int = 10,
                  max_threshold: int = 200, 
                  min_area: int = 100) -> Tuple[np.ndarray, list]:
    """
    Detect blobs in image.
    
    Args:
        img: Input image
        min_threshold: Minimum threshold for blob detection
        max_threshold: Maximum threshold
        min_area: Minimum blob area
        
    Returns:
        Tuple of (image with blobs, keypoints)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Setup blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.filterByArea = True
    params.minArea = min_area
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(gray)
    
    # Draw blobs
    result = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return result, keypoints


def detect_lines_hough(img: np.ndarray, rho: float = 1, 
                        theta: float = np.pi/180,
                        threshold: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect lines using Hough Line Transform.
    
    Args:
        img: Input image
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Accumulator threshold
        
    Returns:
        Tuple of (image with lines, lines array)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.copy()
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    
    if lines is not None:
        for line in lines:
            rho_val, theta_val = line[0]
            a = np.cos(theta_val)
            b = np.sin(theta_val)
            x0 = a * rho_val
            y0 = b * rho_val
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return result, lines


def detect_lines_houghp(img: np.ndarray, rho: float = 1,
                         theta: float = np.pi/180,
                         threshold: int = 50,
                         min_line_length: int = 50,
                         max_line_gap: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect lines using Probabilistic Hough Line Transform.
    
    Args:
        img: Input image
        rho: Distance resolution
        theta: Angle resolution
        threshold: Accumulator threshold
        min_line_length: Minimum line length
        max_line_gap: Maximum gap between line segments
        
    Returns:
        Tuple of (image with lines, lines array)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.copy()
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return result, lines


def detect_circles_hough(img: np.ndarray, dp: float = 1,
                          min_dist: int = 50,
                          param1: int = 100,
                          param2: int = 30,
                          min_radius: int = 0,
                          max_radius: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect circles using Hough Circle Transform.
    
    Args:
        img: Input image
        dp: Inverse ratio of accumulator resolution
        min_dist: Minimum distance between circle centers
        param1: Higher threshold for Canny edge detector
        param2: Accumulator threshold for circle centers
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius (0 = no limit)
        
    Returns:
        Tuple of (image with circles, circles array)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.copy()
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Blur to reduce noise
    gray = cv2.medianBlur(gray, 5)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                                param1=param1, param2=param2,
                                minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw circle
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return result, circles


def template_matching(img: np.ndarray, template: np.ndarray,
                       method: str = 'ccoeff_normed') -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    Find template in image using template matching.
    
    Args:
        img: Input image
        template: Template to find
        method: Matching method ('sqdiff', 'ccorr', 'ccoeff', with '_normed' suffix)
        
    Returns:
        Tuple of (result image with match marked, match location, match value)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = img.copy()
    else:
        gray = img
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    methods = {
        'sqdiff': cv2.TM_SQDIFF,
        'sqdiff_normed': cv2.TM_SQDIFF_NORMED,
        'ccorr': cv2.TM_CCORR,
        'ccorr_normed': cv2.TM_CCORR_NORMED,
        'ccoeff': cv2.TM_CCOEFF,
        'ccoeff_normed': cv2.TM_CCOEFF_NORMED
    }
    method_id = methods.get(method, cv2.TM_CCOEFF_NORMED)
    
    # Template matching
    match_result = cv2.matchTemplate(gray, template, method_id)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
    
    # For SQDIFF methods, minimum is best match
    if method in ['sqdiff', 'sqdiff_normed']:
        top_left = min_loc
        match_value = min_val
    else:
        top_left = max_loc
        match_value = max_val
    
    # Draw rectangle around match
    h, w = template.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
    
    return result, top_left, match_value
