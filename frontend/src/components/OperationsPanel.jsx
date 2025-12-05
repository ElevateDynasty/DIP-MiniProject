import { useState } from 'react'
import { FaChevronDown, FaChevronUp, FaSpinner } from 'react-icons/fa'

const OPERATIONS = {
  'Basic Operations': [
    { id: 'grayscale', name: 'Grayscale', endpoint: 'basic/grayscale' },
    { id: 'negative', name: 'Negative', endpoint: 'basic/negative' },
    { id: 'flip-h', name: 'Flip Horizontal', endpoint: 'basic/flip', params: { direction: 'horizontal' } },
    { id: 'flip-v', name: 'Flip Vertical', endpoint: 'basic/flip', params: { direction: 'vertical' } },
    { id: 'rotate-90', name: 'Rotate 90°', endpoint: 'basic/rotate', params: { angle: 90 } },
    { id: 'rotate-180', name: 'Rotate 180°', endpoint: 'basic/rotate', params: { angle: 180 } },
    { id: 'brightness', name: 'Brightness +50', endpoint: 'basic/brightness', params: { value: 50 } },
    { id: 'contrast', name: 'Contrast 1.5x', endpoint: 'basic/contrast', params: { factor: 1.5 } },
    { id: 'histogram-eq', name: 'Histogram Equalization', endpoint: 'basic/histogram-equalization' },
  ],
  'Filters': [
    { id: 'gaussian', name: 'Gaussian Blur', endpoint: 'filters/gaussian', params: { kernel_size: 5 } },
    { id: 'median', name: 'Median Filter', endpoint: 'filters/median', params: { kernel_size: 5 } },
    { id: 'bilateral', name: 'Bilateral Filter', endpoint: 'filters/bilateral' },
    { id: 'sharpen', name: 'Sharpen', endpoint: 'filters/sharpen' },
    { id: 'unsharp', name: 'Unsharp Mask', endpoint: 'filters/unsharp-mask' },
    { id: 'emboss', name: 'Emboss', endpoint: 'filters/emboss' },
    { id: 'denoise', name: 'Denoise', endpoint: 'filters/denoise' },
  ],
  'Edge Detection': [
    { id: 'sobel', name: 'Sobel', endpoint: 'edge/sobel' },
    { id: 'canny', name: 'Canny', endpoint: 'edge/canny' },
    { id: 'laplacian', name: 'Laplacian', endpoint: 'edge/laplacian' },
    { id: 'prewitt', name: 'Prewitt', endpoint: 'edge/prewitt' },
    { id: 'scharr', name: 'Scharr', endpoint: 'edge/scharr' },
    { id: 'roberts', name: 'Roberts', endpoint: 'edge/roberts' },
  ],
  'Segmentation': [
    { id: 'binary', name: 'Binary Threshold', endpoint: 'segment/binary', params: { threshold: 127 } },
    { id: 'otsu', name: 'Otsu Threshold', endpoint: 'segment/otsu' },
    { id: 'adaptive', name: 'Adaptive Threshold', endpoint: 'segment/adaptive' },
    { id: 'kmeans', name: 'K-Means (3 clusters)', endpoint: 'segment/kmeans', params: { k: 3 } },
    { id: 'watershed', name: 'Watershed', endpoint: 'segment/watershed' },
  ],
  'Morphology': [
    { id: 'erosion', name: 'Erosion', endpoint: 'morph/erosion' },
    { id: 'dilation', name: 'Dilation', endpoint: 'morph/dilation' },
    { id: 'opening', name: 'Opening', endpoint: 'morph/opening' },
    { id: 'closing', name: 'Closing', endpoint: 'morph/closing' },
    { id: 'gradient', name: 'Gradient', endpoint: 'morph/gradient' },
    { id: 'skeleton', name: 'Skeleton', endpoint: 'morph/skeleton' },
  ],
  'Frequency Domain': [
    { id: 'fft-spectrum', name: 'FFT Spectrum', endpoint: 'freq/fft' },
    { id: 'lowpass', name: 'Low-Pass Filter', endpoint: 'freq/lowpass', params: { cutoff: 30 } },
    { id: 'highpass', name: 'High-Pass Filter', endpoint: 'freq/highpass', params: { cutoff: 30 } },
    { id: 'bandpass', name: 'Band-Pass Filter', endpoint: 'freq/bandpass' },
  ],
  'Feature Detection': [
    { id: 'harris', name: 'Harris Corners', endpoint: 'features/harris' },
    { id: 'contours', name: 'Contours', endpoint: 'features/contours' },
    { id: 'hough-lines', name: 'Hough Lines', endpoint: 'features/hough-lines' },
    { id: 'hough-circles', name: 'Hough Circles', endpoint: 'features/hough-circles' },
  ],
  'AI / Deep Learning': [
    { id: 'face-detect', name: 'Face Detection', endpoint: 'ai/face-detection' },
    { id: 'face-mesh', name: 'Face Mesh', endpoint: 'ai/face-mesh' },
    { id: 'hand-detect', name: 'Hand Detection', endpoint: 'ai/hand-detection' },
    { id: 'pencil-sketch', name: 'Pencil Sketch', endpoint: 'ai/pencil-sketch' },
    { id: 'cartoon', name: 'Cartoon Effect', endpoint: 'ai/cartoon' },
    { id: 'oil-painting', name: 'Oil Painting', endpoint: 'ai/oil-painting' },
    { id: 'hdr', name: 'HDR Effect', endpoint: 'ai/hdr' },
    { id: 'super-res', name: 'Super Resolution', endpoint: 'ai/super-resolution' },
  ],
}

function OperationsPanel({ onProcess, isProcessing }) {
  const [expandedCategories, setExpandedCategories] = useState({
    'Basic Operations': true,
    'Filters': true,
  })

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }))
  }

  const handleOperationClick = (operation) => {
    if (isProcessing) return
    onProcess(operation.endpoint, operation.params || {})
  }

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      <div className="p-3 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Operations</h2>
        <p className="text-xs text-gray-500">Click to apply</p>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {Object.entries(OPERATIONS).map(([category, operations]) => (
          <div key={category} className="border border-gray-700 rounded-lg overflow-hidden">
            <button
              onClick={() => toggleCategory(category)}
              className="w-full flex items-center justify-between p-3 bg-gray-800 hover:bg-gray-700 transition-colors"
            >
              <span className="font-medium text-white text-sm">{category}</span>
              {expandedCategories[category] ? (
                <FaChevronUp className="text-gray-400" />
              ) : (
                <FaChevronDown className="text-gray-400" />
              )}
            </button>

            {expandedCategories[category] && (
              <div className="p-2 grid grid-cols-2 gap-1.5">
                {operations.map((op) => (
                  <button
                    key={op.id}
                    onClick={() => handleOperationClick(op)}
                    disabled={isProcessing}
                    className="px-3 py-2 text-xs font-medium rounded bg-gray-700 hover:bg-primary-600 text-gray-300 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left truncate"
                    title={op.name}
                  >
                    {isProcessing ? (
                      <FaSpinner className="animate-spin inline mr-1" />
                    ) : null}
                    {op.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Processing Indicator */}
      {isProcessing && (
        <div className="p-3 border-t border-gray-700 bg-primary-900/30">
          <div className="flex items-center gap-2 text-primary-400 text-sm">
            <FaSpinner className="animate-spin" />
            <span>Processing image...</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default OperationsPanel
