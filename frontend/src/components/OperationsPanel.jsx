import { useState } from 'react'
import { FaChevronDown, FaChevronUp, FaSpinner } from 'react-icons/fa'
import { FaSliders } from 'react-icons/fa6'

const OPERATIONS = {
  'Preset Filters ✨': [
    { id: 'vintage', name: 'Vintage', endpoint: 'presets/vintage', icon: '🎞️' },
    { id: 'noir', name: 'Noir', endpoint: 'presets/noir', icon: '🖤' },
    { id: 'warm', name: 'Warm', endpoint: 'presets/warm', icon: '🌅' },
    { id: 'cool', name: 'Cool', endpoint: 'presets/cool', icon: '❄️' },
    { id: 'dramatic', name: 'Dramatic', endpoint: 'presets/dramatic', icon: '🎭' },
    { id: 'fade', name: 'Fade', endpoint: 'presets/fade', icon: '🌫️' },
  ],
  'Basic Operations': [
    { id: 'grayscale', name: 'Grayscale', endpoint: 'basic/grayscale' },
    { id: 'negative', name: 'Negative', endpoint: 'basic/negative' },
    { id: 'flip-h', name: 'Flip Horizontal', endpoint: 'basic/flip', params: { direction: 'horizontal' } },
    { id: 'flip-v', name: 'Flip Vertical', endpoint: 'basic/flip', params: { direction: 'vertical' } },
    { id: 'rotate', name: 'Rotate', endpoint: 'basic/rotate', hasSlider: true, sliderConfig: { name: 'angle', min: -180, max: 180, default: 90, step: 15 } },
    { id: 'brightness', name: 'Brightness', endpoint: 'basic/brightness', hasSlider: true, sliderConfig: { name: 'value', min: -100, max: 100, default: 30, step: 10 } },
    { id: 'contrast', name: 'Contrast', endpoint: 'basic/contrast', hasSlider: true, sliderConfig: { name: 'factor', min: 0.5, max: 3, default: 1.5, step: 0.1 } },
    { id: 'gamma', name: 'Gamma', endpoint: 'basic/gamma', hasSlider: true, sliderConfig: { name: 'gamma', min: 0.1, max: 3, default: 1.0, step: 0.1 } },
    { id: 'histogram-eq', name: 'Histogram Eq', endpoint: 'basic/histogram-equalization' },
  ],
  'Filters': [
    { id: 'gaussian', name: 'Gaussian Blur', endpoint: 'filters/gaussian', hasSlider: true, sliderConfig: { name: 'kernel_size', min: 3, max: 31, default: 5, step: 2 } },
    { id: 'median', name: 'Median Filter', endpoint: 'filters/median', hasSlider: true, sliderConfig: { name: 'kernel_size', min: 3, max: 31, default: 5, step: 2 } },
    { id: 'bilateral', name: 'Bilateral Filter', endpoint: 'filters/bilateral' },
    { id: 'sharpen', name: 'Sharpen', endpoint: 'filters/sharpen' },
    { id: 'unsharp', name: 'Unsharp Mask', endpoint: 'filters/unsharp-mask' },
    { id: 'emboss', name: 'Emboss', endpoint: 'filters/emboss' },
    { id: 'denoise', name: 'Denoise', endpoint: 'filters/denoise' },
    { id: 'motion-blur', name: 'Motion Blur', endpoint: 'filters/motion-blur' },
  ],
  'Edge Detection': [
    { id: 'sobel', name: 'Sobel', endpoint: 'edge/sobel' },
    { id: 'canny', name: 'Canny', endpoint: 'edge/canny', hasSlider: true, sliderConfig: { name: 'threshold1', min: 10, max: 200, default: 50, step: 10 } },
    { id: 'laplacian', name: 'Laplacian', endpoint: 'edge/laplacian' },
    { id: 'prewitt', name: 'Prewitt', endpoint: 'edge/prewitt' },
    { id: 'scharr', name: 'Scharr', endpoint: 'edge/scharr' },
    { id: 'roberts', name: 'Roberts', endpoint: 'edge/roberts' },
    { id: 'auto-canny', name: 'Auto Canny', endpoint: 'edge/auto-canny' },
  ],
  'Segmentation': [
    { id: 'binary', name: 'Binary Threshold', endpoint: 'segment/binary', hasSlider: true, sliderConfig: { name: 'threshold', min: 0, max: 255, default: 127, step: 5 } },
    { id: 'otsu', name: 'Otsu Threshold', endpoint: 'segment/otsu' },
    { id: 'adaptive', name: 'Adaptive Threshold', endpoint: 'segment/adaptive' },
    { id: 'kmeans', name: 'K-Means', endpoint: 'segment/kmeans', hasSlider: true, sliderConfig: { name: 'k', min: 2, max: 10, default: 3, step: 1 } },
    { id: 'watershed', name: 'Watershed', endpoint: 'segment/watershed' },
    { id: 'contours', name: 'Contours', endpoint: 'segment/contours' },
  ],
  'Morphology': [
    { id: 'erosion', name: 'Erosion', endpoint: 'morph/erosion', hasSlider: true, sliderConfig: { name: 'kernel_size', min: 3, max: 15, default: 5, step: 2 } },
    { id: 'dilation', name: 'Dilation', endpoint: 'morph/dilation', hasSlider: true, sliderConfig: { name: 'kernel_size', min: 3, max: 15, default: 5, step: 2 } },
    { id: 'opening', name: 'Opening', endpoint: 'morph/opening' },
    { id: 'closing', name: 'Closing', endpoint: 'morph/closing' },
    { id: 'gradient', name: 'Gradient', endpoint: 'morph/gradient' },
    { id: 'skeleton', name: 'Skeleton', endpoint: 'morph/skeleton' },
  ],
  'Frequency Domain': [
    { id: 'fft-spectrum', name: 'FFT Spectrum', endpoint: 'freq/fft' },
    { id: 'lowpass', name: 'Low-Pass Filter', endpoint: 'freq/lowpass', hasSlider: true, sliderConfig: { name: 'cutoff', min: 5, max: 100, default: 30, step: 5 } },
    { id: 'highpass', name: 'High-Pass Filter', endpoint: 'freq/highpass', hasSlider: true, sliderConfig: { name: 'cutoff', min: 5, max: 100, default: 30, step: 5 } },
    { id: 'bandpass', name: 'Band-Pass Filter', endpoint: 'freq/bandpass' },
  ],
  'Feature Detection': [
    { id: 'harris', name: 'Harris Corners', endpoint: 'features/harris' },
    { id: 'shi-tomasi', name: 'Shi-Tomasi', endpoint: 'features/shi-tomasi' },
    { id: 'orb', name: 'ORB Features', endpoint: 'features/orb' },
    { id: 'hough-lines', name: 'Hough Lines', endpoint: 'features/hough-lines' },
    { id: 'hough-circles', name: 'Hough Circles', endpoint: 'features/hough-circles' },
  ],
  'AI / Deep Learning 🤖': [
    { id: 'face-detect', name: 'Face Detection', endpoint: 'dl/face-detection', icon: '👤' },
    { id: 'eye-detect', name: 'Eye Detection', endpoint: 'dl/eye-detection', icon: '👁️' },
    { id: 'remove-bg', name: 'Remove Background', endpoint: 'ai/remove-background', icon: '✂️' },
    { id: 'detect-objects', name: 'Detect Objects', endpoint: 'ai/detect-objects', icon: '🔍' },
    { id: 'extract-text', name: 'Extract Text', endpoint: 'ai/extract-text', icon: '📝' },
    { id: 'colorize', name: 'Colorize', endpoint: 'ai/colorize', icon: '🎨' },
    { id: 'hdr', name: 'HDR Effect', endpoint: 'ai/hdr-effect', icon: '🌟' },
    { id: 'pencil-sketch', name: 'Pencil Sketch', endpoint: 'dl/pencil-sketch', icon: '✏️' },
    { id: 'cartoon', name: 'Cartoon', endpoint: 'dl/cartoon', icon: '🎨' },
    { id: 'stylization', name: 'Stylization', endpoint: 'dl/stylization', icon: '🖼️' },
  ],
}

function OperationsPanel({ onProcess, isProcessing }) {
  const [expandedCategories, setExpandedCategories] = useState({
    'Preset Filters ✨': true,
    'Basic Operations': false,
  })
  const [sliderValues, setSliderValues] = useState({})
  const [activeSlider, setActiveSlider] = useState(null)

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }))
  }

  const handleSliderChange = (opId, value) => {
    setSliderValues(prev => ({
      ...prev,
      [opId]: value
    }))
  }

  const handleOperationClick = (operation) => {
    if (isProcessing) return
    
    if (operation.hasSlider) {
      if (activeSlider === operation.id) {
        const value = sliderValues[operation.id] ?? operation.sliderConfig.default
        onProcess(operation.endpoint, { [operation.sliderConfig.name]: value })
        setActiveSlider(null)
      } else {
        setActiveSlider(operation.id)
        if (sliderValues[operation.id] === undefined) {
          setSliderValues(prev => ({
            ...prev,
            [operation.id]: operation.sliderConfig.default
          }))
        }
      }
    } else {
      onProcess(operation.endpoint, operation.params || {})
    }
  }

  const applyWithSlider = (operation) => {
    if (isProcessing) return
    const value = sliderValues[operation.id] ?? operation.sliderConfig.default
    onProcess(operation.endpoint, { [operation.sliderConfig.name]: value })
    setActiveSlider(null)
  }

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      <div className="p-3 border-b border-gray-700 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Operations</h2>
          <p className="text-xs text-gray-500">60+ effects • Click to apply</p>
        </div>
        <FaSliders className="text-primary-400" />
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
              <div className="p-2 space-y-1">
                <div className="grid grid-cols-2 gap-1.5">
                  {operations.map((op) => (
                    <button
                      key={op.id}
                      onClick={() => handleOperationClick(op)}
                      disabled={isProcessing}
                      className={`px-3 py-2 text-xs font-medium rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left truncate flex items-center gap-1 ${
                        activeSlider === op.id 
                          ? 'bg-primary-600 text-white ring-2 ring-primary-400' 
                          : 'bg-gray-700 hover:bg-primary-600 text-gray-300 hover:text-white'
                      }`}
                      title={op.name}
                    >
                      {op.icon && <span>{op.icon}</span>}
                      {op.name}
                      {op.hasSlider && <FaSliders className="ml-auto text-[10px] opacity-60" />}
                    </button>
                  ))}
                </div>
                
                {/* Slider for active operation */}
                {operations.map((op) => (
                  activeSlider === op.id && op.hasSlider && (
                    <div key={`slider-${op.id}`} className="mt-2 p-3 bg-gray-800 rounded-lg border border-primary-500/30">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-gray-400 capitalize">{op.sliderConfig.name.replace('_', ' ')}</span>
                        <span className="text-xs text-primary-400 font-mono bg-gray-900 px-2 py-0.5 rounded">
                          {sliderValues[op.id] ?? op.sliderConfig.default}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={op.sliderConfig.min}
                        max={op.sliderConfig.max}
                        step={op.sliderConfig.step}
                        value={sliderValues[op.id] ?? op.sliderConfig.default}
                        onChange={(e) => handleSliderChange(op.id, parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                      />
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-[10px] text-gray-500">{op.sliderConfig.min}</span>
                        <button
                          onClick={() => applyWithSlider(op)}
                          disabled={isProcessing}
                          className="btn btn-primary text-xs px-4 py-1"
                        >
                          {isProcessing ? <FaSpinner className="animate-spin" /> : 'Apply'}
                        </button>
                        <span className="text-[10px] text-gray-500">{op.sliderConfig.max}</span>
                      </div>
                    </div>
                  )
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {isProcessing && (
        <div className="p-3 border-t border-gray-700 bg-primary-900/30">
          <div className="flex items-center gap-2 text-primary-400 text-sm">
            <FaSpinner className="animate-spin" />
            <span>Processing...</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default OperationsPanel
