import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { FaCloudUploadAlt, FaImage } from 'react-icons/fa'

function ImageUploader({ onUpload }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0])
    }
  }, [onUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.pgm']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024 // 10MB
  })

  return (
    <div className="max-w-2xl mx-auto animate-fadeIn">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-3">
          Digital Image Processing
        </h2>
        <p className="text-gray-400">
          Upload an image to apply various image processing operations
        </p>
      </div>

      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center gap-4">
          <div className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors ${
            isDragActive ? 'bg-primary-500/20' : 'bg-gray-700/50'
          }`}>
            {isDragActive ? (
              <FaImage className="text-4xl text-primary-400" />
            ) : (
              <FaCloudUploadAlt className="text-4xl text-gray-400" />
            )}
          </div>

          <div>
            <p className="text-lg text-white font-medium">
              {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to browse files
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="px-2 py-1 bg-gray-700 rounded">PNG</span>
            <span className="px-2 py-1 bg-gray-700 rounded">JPG</span>
            <span className="px-2 py-1 bg-gray-700 rounded">JPEG</span>
            <span className="px-2 py-1 bg-gray-700 rounded">BMP</span>
            <span className="px-2 py-1 bg-gray-700 rounded">WEBP</span>
          </div>

          <p className="text-xs text-gray-600">Maximum file size: 10MB</p>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
        {[
          { title: 'Filters', desc: 'Blur, Sharpen, Denoise' },
          { title: 'Edge Detection', desc: 'Sobel, Canny, Laplacian' },
          { title: 'Segmentation', desc: 'Threshold, K-Means' },
          { title: 'AI Effects', desc: 'Face Detection, Sketch' },
        ].map((feature, idx) => (
          <div key={idx} className="card p-4 text-center">
            <h3 className="text-white font-medium text-sm">{feature.title}</h3>
            <p className="text-gray-500 text-xs mt-1">{feature.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ImageUploader
