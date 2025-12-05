import { useState, useRef, useEffect } from 'react'
import { FaTimes, FaArrowsAltH } from 'react-icons/fa'

function CompareSlider({ originalImage, processedImage, onClose }) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef(null)

  const handleMove = (clientX) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPosition(percentage)
  }

  const handleMouseDown = () => setIsDragging(true)
  const handleMouseUp = () => setIsDragging(false)
  
  const handleMouseMove = (e) => {
    if (isDragging) {
      handleMove(e.clientX)
    }
  }

  const handleTouchMove = (e) => {
    handleMove(e.touches[0].clientX)
  }

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
      return () => {
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging])

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      {/* Header */}
      <div className="p-3 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FaArrowsAltH className="text-primary-400" />
          <h2 className="text-lg font-semibold text-white">Compare</h2>
        </div>
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm"
        >
          <FaTimes />
        </button>
      </div>

      {/* Comparison View */}
      <div 
        ref={containerRef}
        className="flex-1 relative overflow-hidden cursor-ew-resize select-none"
        onMouseDown={handleMouseDown}
        onTouchMove={handleTouchMove}
      >
        {/* Processed Image (Background) */}
        <div className="absolute inset-0">
          <img
            src={processedImage}
            alt="Processed"
            className="w-full h-full object-contain"
            draggable={false}
          />
          <span className="absolute bottom-4 right-4 bg-primary-500/80 px-2 py-1 rounded text-xs text-white">
            Processed
          </span>
        </div>

        {/* Original Image (Clipped) */}
        <div 
          className="absolute inset-0 overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <img
            src={originalImage}
            alt="Original"
            className="absolute top-0 left-0 w-full h-full object-contain"
            style={{ 
              width: containerRef.current ? `${containerRef.current.offsetWidth}px` : '100%',
              maxWidth: 'none'
            }}
            draggable={false}
          />
          <span className="absolute bottom-4 left-4 bg-black/80 px-2 py-1 rounded text-xs text-white">
            Original
          </span>
        </div>

        {/* Slider Handle */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white shadow-lg cursor-ew-resize z-10"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 bg-white rounded-full shadow-lg flex items-center justify-center">
            <FaArrowsAltH className="text-gray-700" />
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="p-3 border-t border-gray-700 text-center">
        <p className="text-xs text-gray-500">Drag the slider to compare before and after</p>
      </div>
    </div>
  )
}

export default CompareSlider
