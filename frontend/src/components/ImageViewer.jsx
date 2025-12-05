import { FaDownload, FaUndo, FaRedo, FaTimes, FaExpand, FaSearchPlus, FaSearchMinus } from 'react-icons/fa'
import { useState } from 'react'

function ImageViewer({ originalImage, processedImage, onReset }) {
  const [zoom, setZoom] = useState(100)
  const [compareMode, setCompareMode] = useState(false)

  const handleDownload = () => {
    if (!processedImage) return
    
    const link = document.createElement('a')
    link.href = processedImage
    link.download = 'processed-image.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 25, 200))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 25, 25))
  const handleResetZoom = () => setZoom(100)

  const displayImage = processedImage || originalImage

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <button
            onClick={handleZoomOut}
            className="btn btn-ghost btn-sm"
            title="Zoom Out"
          >
            <FaSearchMinus />
          </button>
          <span className="text-sm text-gray-400 w-12 text-center">{zoom}%</span>
          <button
            onClick={handleZoomIn}
            className="btn btn-ghost btn-sm"
            title="Zoom In"
          >
            <FaSearchPlus />
          </button>
          <button
            onClick={handleResetZoom}
            className="btn btn-ghost btn-sm"
            title="Reset Zoom"
          >
            <FaExpand />
          </button>
        </div>

        <div className="flex items-center gap-2">
          {processedImage && (
            <>
              <button
                onClick={() => setCompareMode(!compareMode)}
                className={`btn btn-sm ${compareMode ? 'btn-primary' : 'btn-ghost'}`}
              >
                Compare
              </button>
              <button
                onClick={handleDownload}
                className="btn btn-ghost btn-sm"
                title="Download"
              >
                <FaDownload />
              </button>
            </>
          )}
          <button
            onClick={onReset}
            className="btn btn-ghost btn-sm text-red-400 hover:text-red-300"
            title="Start Over"
          >
            <FaTimes />
          </button>
        </div>
      </div>

      {/* Image Display */}
      <div className="flex-1 overflow-auto p-4">
        {compareMode && processedImage ? (
          <div className="grid grid-cols-2 gap-4 h-full">
            <div className="relative">
              <span className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-xs text-white">
                Original
              </span>
              <img
                src={originalImage}
                alt="Original"
                className="w-full h-full object-contain rounded"
                style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'center' }}
              />
            </div>
            <div className="relative">
              <span className="absolute top-2 left-2 bg-primary-500/80 px-2 py-1 rounded text-xs text-white">
                Processed
              </span>
              <img
                src={processedImage}
                alt="Processed"
                className="w-full h-full object-contain rounded"
                style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'center' }}
              />
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <img
              src={displayImage}
              alt="Preview"
              className="max-w-full max-h-full object-contain rounded transition-transform"
              style={{ transform: `scale(${zoom / 100})` }}
            />
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div className="px-4 py-2 border-t border-gray-700 flex items-center justify-between text-xs text-gray-500">
        <span>{processedImage ? 'Processed' : 'Original'}</span>
        <span>Zoom: {zoom}%</span>
      </div>
    </div>
  )
}

export default ImageViewer
