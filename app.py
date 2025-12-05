"""
DIP Project - Streamlit Web Application
Interactive Digital Image Processing Tool
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Import our modules
from src.basic_operations import (
    resize_image, rotate_image, flip_image, translate_image, 
    negative_image, adjust_brightness, adjust_contrast, gamma_correction,
    histogram_equalization, perspective_transform
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
    kmeans_segmentation, watershed_segmentation, contour_detection, draw_contours
)
from src.morphology import (
    erosion, dilation, opening, closing, morphological_gradient,
    top_hat, black_hat, skeletonization, boundary_extraction
)
from src.frequency_domain import (
    get_magnitude_spectrum, compute_dft, ideal_lowpass_filter, 
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


# Page configuration
st.set_page_config(
    page_title="DIP Project - Image Processing",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def load_image(uploaded_file):
    """Load image from uploaded file."""
    image = Image.open(uploaded_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def display_image(img, caption=""):
    """Display image in Streamlit."""
    if len(img.shape) == 2:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


def download_image(img, filename="processed_image.png"):
    """Create download button for image."""
    if len(img.shape) == 2:
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    
    st.download_button(
        label="📥 Download Result",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )


def main():
    st.markdown('<p class="main-header">🖼️ Digital Image Processing</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Image Processing Tool with Python & OpenCV</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'bmp', 'pgm'])
    
    if uploaded_file is not None:
        # Load and display original image
        original_img = load_image(uploaded_file)
        
        st.sidebar.title("🔧 Operations")
        operation_category = st.sidebar.selectbox(
            "Select Category",
            ["Basic Operations", "Filters", "Edge Detection", "Segmentation", 
             "Morphology", "Frequency Domain", "Feature Detection", "Deep Learning"]
        )
        
        # Create columns for original and processed images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            display_image(original_img, "Original")
        
        # Process based on selected category
        processed_img = original_img.copy()
        
        if operation_category == "Basic Operations":
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Resize", "Rotate", "Flip", "Brightness", "Contrast", 
                 "Gamma Correction", "Negative", "Histogram Equalization"]
            )
            
            if operation == "Resize":
                scale = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
                processed_img = resize_image(original_img, scale=scale)
            
            elif operation == "Rotate":
                angle = st.sidebar.slider("Angle (degrees)", -180, 180, 0)
                processed_img = rotate_image(original_img, angle)
            
            elif operation == "Flip":
                direction = st.sidebar.selectbox("Direction", ["horizontal", "vertical", "both"])
                processed_img = flip_image(original_img, direction)
            
            elif operation == "Brightness":
                value = st.sidebar.slider("Brightness", -100, 100, 0)
                processed_img = adjust_brightness(original_img, value)
            
            elif operation == "Contrast":
                factor = st.sidebar.slider("Contrast Factor", 0.1, 3.0, 1.0, 0.1)
                processed_img = adjust_contrast(original_img, factor)
            
            elif operation == "Gamma Correction":
                gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
                processed_img = gamma_correction(original_img, gamma)
            
            elif operation == "Negative":
                processed_img = negative_image(original_img)
            
            elif operation == "Histogram Equalization":
                use_clahe = st.sidebar.checkbox("Use CLAHE", value=False)
                processed_img = histogram_equalization(original_img, use_clahe=use_clahe)
        
        elif operation_category == "Filters":
            operation = st.sidebar.selectbox(
                "Select Filter",
                ["Average", "Gaussian", "Median", "Bilateral", "Sharpen", 
                 "Unsharp Mask", "Emboss", "Motion Blur", "Denoise",
                 "Add Gaussian Noise", "Add Salt & Pepper Noise"]
            )
            
            if operation == "Average":
                kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 5, 2)
                processed_img = average_filter(original_img, kernel_size)
            
            elif operation == "Gaussian":
                kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 5, 2)
                sigma = st.sidebar.slider("Sigma", 0.0, 10.0, 0.0, 0.5)
                processed_img = gaussian_filter(original_img, kernel_size, sigma)
            
            elif operation == "Median":
                kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 5, 2)
                processed_img = median_filter(original_img, kernel_size)
            
            elif operation == "Bilateral":
                d = st.sidebar.slider("Diameter", 3, 15, 9)
                sigma_color = st.sidebar.slider("Sigma Color", 10, 200, 75)
                sigma_space = st.sidebar.slider("Sigma Space", 10, 200, 75)
                processed_img = bilateral_filter(original_img, d, sigma_color, sigma_space)
            
            elif operation == "Sharpen":
                strength = st.sidebar.slider("Strength", 0.1, 3.0, 1.0, 0.1)
                processed_img = sharpen_filter(original_img, strength)
            
            elif operation == "Unsharp Mask":
                amount = st.sidebar.slider("Amount", 0.1, 3.0, 1.0, 0.1)
                processed_img = unsharp_mask(original_img, amount=amount)
            
            elif operation == "Emboss":
                processed_img = emboss_filter(original_img)
            
            elif operation == "Motion Blur":
                size = st.sidebar.slider("Size", 5, 50, 15)
                angle = st.sidebar.slider("Angle", 0, 180, 0)
                processed_img = motion_blur(original_img, size, angle)
            
            elif operation == "Denoise":
                strength = st.sidebar.slider("Strength", 1, 30, 10)
                processed_img = non_local_means_denoise(original_img, h=strength)
            
            elif operation == "Add Gaussian Noise":
                std = st.sidebar.slider("Noise Level", 5, 100, 25)
                processed_img = add_gaussian_noise(original_img, std=std)
            
            elif operation == "Add Salt & Pepper Noise":
                prob = st.sidebar.slider("Probability", 0.01, 0.1, 0.02, 0.01)
                processed_img = add_salt_pepper_noise(original_img, salt_prob=prob, pepper_prob=prob)
        
        elif operation_category == "Edge Detection":
            operation = st.sidebar.selectbox(
                "Select Method",
                ["Sobel", "Canny", "Laplacian", "Prewitt", "Roberts", "Scharr", "Auto Canny"]
            )
            
            if operation == "Sobel":
                ksize = st.sidebar.slider("Kernel Size", 1, 7, 3, 2)
                processed_img = sobel_edge_detection(original_img, ksize=ksize)
            
            elif operation == "Canny":
                low = st.sidebar.slider("Low Threshold", 0, 200, 50)
                high = st.sidebar.slider("High Threshold", 50, 300, 150)
                processed_img = canny_edge_detection(original_img, low, high)
            
            elif operation == "Laplacian":
                ksize = st.sidebar.slider("Kernel Size", 1, 7, 3, 2)
                processed_img = laplacian_edge_detection(original_img, ksize)
            
            elif operation == "Prewitt":
                processed_img = prewitt_edge_detection(original_img)
            
            elif operation == "Roberts":
                processed_img = roberts_edge_detection(original_img)
            
            elif operation == "Scharr":
                processed_img = scharr_edge_detection(original_img)
            
            elif operation == "Auto Canny":
                sigma = st.sidebar.slider("Sigma", 0.1, 1.0, 0.33, 0.05)
                processed_img = auto_canny(original_img, sigma)
        
        elif operation_category == "Segmentation":
            operation = st.sidebar.selectbox(
                "Select Method",
                ["Simple Threshold", "Otsu Threshold", "Adaptive Threshold",
                 "K-Means", "Contour Detection"]
            )
            
            if operation == "Simple Threshold":
                thresh = st.sidebar.slider("Threshold", 0, 255, 127)
                processed_img = simple_threshold(original_img, thresh)
            
            elif operation == "Otsu Threshold":
                processed_img, thresh = otsu_threshold(original_img)
                st.sidebar.info(f"Optimal threshold: {thresh}")
            
            elif operation == "Adaptive Threshold":
                method = st.sidebar.selectbox("Method", ["gaussian", "mean"])
                block_size = st.sidebar.slider("Block Size", 3, 51, 11, 2)
                c = st.sidebar.slider("C", -20, 20, 2)
                processed_img = adaptive_threshold(original_img, method=method, 
                                                    block_size=block_size, c=c)
            
            elif operation == "K-Means":
                k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
                processed_img = kmeans_segmentation(original_img, k=k)
            
            elif operation == "Contour Detection":
                thresh = st.sidebar.slider("Threshold", 0, 255, 127)
                contours = contour_detection(original_img, threshold=thresh)
                processed_img = draw_contours(original_img, contours)
        
        elif operation_category == "Morphology":
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Erosion", "Dilation", "Opening", "Closing", 
                 "Gradient", "Top Hat", "Black Hat", "Skeleton", "Boundary"]
            )
            
            kernel_size = st.sidebar.slider("Kernel Size", 3, 21, 5, 2)
            kernel_shape = st.sidebar.selectbox("Kernel Shape", ["rect", "ellipse", "cross"])
            
            if operation == "Erosion":
                iterations = st.sidebar.slider("Iterations", 1, 10, 1)
                processed_img = erosion(original_img, kernel_size, iterations, kernel_shape)
            
            elif operation == "Dilation":
                iterations = st.sidebar.slider("Iterations", 1, 10, 1)
                processed_img = dilation(original_img, kernel_size, iterations, kernel_shape)
            
            elif operation == "Opening":
                processed_img = opening(original_img, kernel_size, kernel_shape)
            
            elif operation == "Closing":
                processed_img = closing(original_img, kernel_size, kernel_shape)
            
            elif operation == "Gradient":
                processed_img = morphological_gradient(original_img, kernel_size, kernel_shape)
            
            elif operation == "Top Hat":
                processed_img = top_hat(original_img, kernel_size, kernel_shape)
            
            elif operation == "Black Hat":
                processed_img = black_hat(original_img, kernel_size, kernel_shape)
            
            elif operation == "Skeleton":
                processed_img = skeletonization(original_img)
            
            elif operation == "Boundary":
                processed_img = boundary_extraction(original_img, kernel_size)
        
        elif operation_category == "Frequency Domain":
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Magnitude Spectrum", "Ideal Lowpass", "Ideal Highpass",
                 "Butterworth Lowpass", "Gaussian Lowpass", "Homomorphic Filter"]
            )
            
            if operation == "Magnitude Spectrum":
                dft = compute_dft(original_img)
                processed_img = get_magnitude_spectrum(dft)
            
            elif operation == "Ideal Lowpass":
                cutoff = st.sidebar.slider("Cutoff Frequency", 5, 200, 30)
                processed_img = ideal_lowpass_filter(original_img, cutoff)
            
            elif operation == "Ideal Highpass":
                cutoff = st.sidebar.slider("Cutoff Frequency", 5, 200, 30)
                processed_img = ideal_highpass_filter(original_img, cutoff)
            
            elif operation == "Butterworth Lowpass":
                cutoff = st.sidebar.slider("Cutoff Frequency", 5, 200, 30)
                order = st.sidebar.slider("Order", 1, 10, 2)
                processed_img = butterworth_lowpass_filter(original_img, cutoff, order)
            
            elif operation == "Gaussian Lowpass":
                cutoff = st.sidebar.slider("Cutoff Frequency", 5, 200, 30)
                processed_img = gaussian_lowpass_filter(original_img, cutoff)
            
            elif operation == "Homomorphic Filter":
                gamma_l = st.sidebar.slider("Gamma Low", 0.1, 1.0, 0.5, 0.1)
                gamma_h = st.sidebar.slider("Gamma High", 1.0, 5.0, 2.0, 0.1)
                cutoff = st.sidebar.slider("Cutoff", 5, 100, 30)
                processed_img = homomorphic_filter(original_img, gamma_l, gamma_h, cutoff)
        
        elif operation_category == "Feature Detection":
            operation = st.sidebar.selectbox(
                "Select Method",
                ["Harris Corners", "Shi-Tomasi Corners", "ORB Features",
                 "Hough Lines", "Hough Circles"]
            )
            
            if operation == "Harris Corners":
                threshold = st.sidebar.slider("Threshold", 0.001, 0.1, 0.01, 0.001)
                processed_img = detect_harris_corners(original_img, threshold=threshold)
            
            elif operation == "Shi-Tomasi Corners":
                max_corners = st.sidebar.slider("Max Corners", 10, 500, 100)
                processed_img, _ = detect_shi_tomasi_corners(original_img, max_corners=max_corners)
            
            elif operation == "ORB Features":
                n_features = st.sidebar.slider("Number of Features", 100, 2000, 500)
                processed_img, _, _ = detect_orb_features(original_img, n_features=n_features)
            
            elif operation == "Hough Lines":
                threshold = st.sidebar.slider("Threshold", 10, 200, 50)
                min_length = st.sidebar.slider("Min Line Length", 10, 200, 50)
                processed_img, _ = detect_lines_houghp(original_img, threshold=threshold,
                                                        min_line_length=min_length)
            
            elif operation == "Hough Circles":
                min_dist = st.sidebar.slider("Min Distance", 10, 200, 50)
                param2 = st.sidebar.slider("Threshold", 10, 100, 30)
                processed_img, _ = detect_circles_hough(original_img, min_dist=min_dist, 
                                                         param2=param2)
        
        elif operation_category == "Deep Learning":
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Face Detection", "Eye Detection", "Pencil Sketch", 
                 "Cartoon Effect", "Stylization", "Detail Enhance"]
            )
            
            if operation == "Face Detection":
                processed_img, faces = detect_faces_haar(original_img)
                st.sidebar.info(f"Detected {len(faces)} face(s)")
            
            elif operation == "Eye Detection":
                processed_img, eyes = detect_eyes(original_img)
                st.sidebar.info(f"Detected {len(eyes)} eye(s)")
            
            elif operation == "Pencil Sketch":
                shade = st.sidebar.slider("Shade Factor", 0.01, 0.1, 0.05, 0.01)
                gray_sketch, color_sketch = pencil_sketch(original_img, shade_factor=shade)
                show_color = st.sidebar.checkbox("Show Color Sketch", value=False)
                processed_img = color_sketch if show_color else cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
            
            elif operation == "Cartoon Effect":
                processed_img = neural_style_transfer_simple(original_img, style='cartoon')
            
            elif operation == "Stylization":
                sigma_s = st.sidebar.slider("Sigma S", 10, 200, 60)
                sigma_r = st.sidebar.slider("Sigma R", 0.1, 1.0, 0.45, 0.05)
                processed_img = stylization(original_img, sigma_s, sigma_r)
            
            elif operation == "Detail Enhance":
                sigma_s = st.sidebar.slider("Sigma S", 1, 50, 10)
                sigma_r = st.sidebar.slider("Sigma R", 0.05, 0.5, 0.15, 0.05)
                processed_img = detail_enhance(original_img, sigma_s, sigma_r)
        
        # Display processed image
        with col2:
            st.subheader("Processed Image")
            display_image(processed_img, f"{operation_category}: {operation}")
            download_image(processed_img)
    
    else:
        # Show welcome message when no image is uploaded
        st.info("👆 Please upload an image from the sidebar to get started!")
        
        st.markdown("""
        ### Features
        
        This application provides a comprehensive set of image processing tools:
        
        - **Basic Operations**: Resize, rotate, flip, brightness, contrast, gamma correction
        - **Filters**: Gaussian, median, bilateral, sharpen, denoise
        - **Edge Detection**: Sobel, Canny, Laplacian, Prewitt, Roberts
        - **Segmentation**: Thresholding, K-means, watershed, contours
        - **Morphology**: Erosion, dilation, opening, closing, skeleton
        - **Frequency Domain**: DFT, lowpass/highpass filters, homomorphic
        - **Feature Detection**: Harris, Shi-Tomasi, ORB, Hough transforms
        - **Deep Learning**: Face detection, eye detection, artistic effects
        """)


if __name__ == "__main__":
    main()
