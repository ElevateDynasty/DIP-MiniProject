# DIP Project - Digital Image Processing

A modern Digital Image Processing web application built with **React.js** + **FastAPI**.

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3.3-38B2AC?logo=tailwindcss)

## 🚀 Features

### Image Processing Operations (50+ Operations)

| Category | Operations |
|----------|------------|
| **Basic** | Grayscale, Negative, Flip, Rotate, Brightness, Contrast, Histogram Equalization |
| **Filters** | Gaussian, Median, Bilateral, Sharpen, Unsharp Mask, Emboss, Denoise |
| **Edge Detection** | Sobel, Canny, Laplacian, Prewitt, Scharr, Roberts |
| **Segmentation** | Binary, Otsu, Adaptive Threshold, K-Means, Watershed |
| **Morphology** | Erosion, Dilation, Opening, Closing, Gradient, Skeleton |
| **Frequency Domain** | FFT Spectrum, Low-Pass, High-Pass, Band-Pass Filters |
| **Feature Detection** | Harris Corners, Contours, Hough Lines/Circles |
| **AI Effects** | Face Detection, Face Mesh, Hand Detection, Pencil Sketch, Cartoon, HDR |

### Modern UI
- 🎨 Beautiful dark theme with gradient accents
- 📱 Fully responsive design
- 🖼️ Drag & drop image upload
- 🔍 Zoom controls
- ↔️ Before/After comparison mode
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
| POST | `/api/filters/gaussian` | Apply Gaussian blur |
| POST | `/api/edge/canny` | Canny edge detection |
| POST | `/api/ai/face-detection` | Detect faces in image |
| GET | `/operations` | List all available operations |

## 📁 Project Structure

```
DIPProject/
├── backend/
│   ├── main.py              # FastAPI application (50+ endpoints)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── api/imageApi.js  # API client
│   │   ├── components/      # React components
│   │   ├── App.jsx          # Main app
│   │   └── index.css        # Tailwind styles
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
