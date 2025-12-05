# DIP Project - Digital Image Processing

A modern Digital Image Processing web application built with **React.js** + **FastAPI**.

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3.3-38B2AC?logo=tailwindcss)

## 🚀 Features

### Image Processing Operations (60+ Operations)

| Category | Operations |
|----------|------------|
| **Preset Filters** | Vintage, Noir, Warm, Cool, Dramatic, Fade (Instagram-style) |
| **Basic** | Grayscale, Negative, Flip, Rotate, Brightness, Contrast, Gamma, Histogram Equalization |
| **Filters** | Gaussian, Median, Bilateral, Sharpen, Unsharp Mask, Emboss, Denoise, Motion Blur |
| **Edge Detection** | Sobel, Canny, Laplacian, Prewitt, Scharr, Roberts, Auto Canny |
| **Segmentation** | Binary, Otsu, Adaptive Threshold, K-Means, Watershed, Contours |
| **Morphology** | Erosion, Dilation, Opening, Closing, Gradient, Skeleton, Boundary |
| **Frequency Domain** | FFT Spectrum, Low-Pass, High-Pass, Band-Pass, Butterworth, Gaussian LP |
| **Feature Detection** | Harris Corners, Shi-Tomasi, ORB, Hough Lines/Circles |
| **AI / Deep Learning** | Face Detection, Eye Detection, Background Removal, Object Detection, OCR, Colorize, HDR, Pencil Sketch, Cartoon, Stylization |

### Advanced Features
- 🎬 **Real-time Video Processing** - WebSocket-based live video effects
- 📦 **Batch Processing** - Process multiple images, download as ZIP
- ✂️ **Background Removal** - AI-powered background removal
- 🔍 **Object Detection** - Automatic object detection with bounding boxes
- 📝 **OCR Text Extraction** - Detect text regions in images
- 🎨 **Image Inpainting** - Remove objects and fill gaps
- 🖌️ **Custom Filter Builder** - Create custom convolution kernels

### Modern UI
- 🎨 Beautiful dark/light theme with toggle
- 📱 Fully responsive design
- 🖼️ Drag & drop image upload
- 🔍 Zoom controls (25% - 300%)
- ↔️ Interactive before/after comparison slider
- ⏪ Full history with undo/redo support
- ✏️ Image annotation tools (pen, shapes, text, eraser)
- 🎚️ Parameter sliders for adjustable operations
- 💾 Download processed images

## 🛠️ Tech Stack

### Backend (FastAPI)
- **FastAPI** - High-performance Python web framework
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **MediaPipe** - AI face/hand detection
- **PyTorch** - Deep learning (Super Resolution)

### Frontend (React)
- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **TailwindCSS** - Utility-first CSS framework
- **Axios** - HTTP client
- **React Dropzone** - File upload
- **React Icons** - Icon library

## 📦 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+

### 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

🌐 Open `http://localhost:3000` in your browser!

## 📡 API Documentation

FastAPI provides interactive API docs at `http://localhost:8000/docs`

### Sample Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/basic/grayscale` | Convert to grayscale |
| POST | `/api/filters/gaussian?kernel_size=5` | Apply Gaussian blur |
| POST | `/api/edge/canny` | Canny edge detection |
| POST | `/api/presets/vintage` | Apply vintage filter |
| POST | `/api/ai/remove-background` | Remove image background |
| POST | `/api/ai/detect-objects` | Detect objects in image |
| POST | `/api/batch/process` | Batch process images (ZIP) |
| WS | `/ws/video` | Real-time video processing |
| GET | `/operations` | List all available operations |

## 📁 Project Structure

```
DIPProject/
├── backend/
│   ├── main.py              # FastAPI application (60+ endpoints)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── api/imageApi.js  # API client
│   │   ├── components/
│   │   │   ├── Header.jsx          # Navigation + theme toggle
│   │   │   ├── ImageUploader.jsx   # Drag & drop upload
│   │   │   ├── ImageViewer.jsx     # Image display + zoom
│   │   │   ├── OperationsPanel.jsx # 60+ operations with sliders
│   │   │   ├── HistoryPanel.jsx    # Undo/redo history
│   │   │   ├── CompareSlider.jsx   # Before/after comparison
│   │   │   └── AnnotationTools.jsx # Drawing tools
│   │   ├── App.jsx          # Main app with state management
│   │   └── index.css        # Tailwind + custom styles
│   ├── package.json
│   └── vite.config.js
├── src/                     # Python image processing modules
│   ├── basic_operations.py
│   ├── filters.py
│   ├── edge_detection.py
│   ├── segmentation.py
│   ├── morphology.py
│   ├── frequency_domain.py
│   ├── feature_detection.py
│   └── deep_learning.py
├── app.py                   # Streamlit app (alternative UI)
├── notebooks/demo.ipynb     # Jupyter notebook demo
└── test_images/             # Sample images
```

## 🖥️ Alternative UIs

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) - Computer vision library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://reactjs.org/) - UI library
- [TailwindCSS](https://tailwindcss.com/) - CSS framework
- [MediaPipe](https://mediapipe.dev/) - ML solutions
