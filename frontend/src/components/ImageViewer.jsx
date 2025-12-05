import { FaDownload, FaUndo, FaRedo, FaTimes, FaExpand, FaSearchPlus, FaSearchMinus, FaColumns, FaPen } from 'react-icons/fa'
import { useState } from 'react'

function ImageViewer({ 
  originalImage, 
  processedImage, 
  onReset, 
  onCompare, 
  onAnnotate,
  canUndo, 
  canRedo, 
  onUndo, 
  onRedo 
}) {
  const [zoom, setZoom] = useState(100)

  const handleDownload = () => {
    if (!processedImage && !originalImage) return
    
    const link = document.createElement('a')
    link.href = processedImage || originalImage
    link.download = 'processed-image.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 25, 300))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 25, 25))
  const handleResetZoom = () => setZoom(100)

  const displayImage = processedImage || originalImage

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomOut}
            className="btn btn-ghost btn-sm"
            title="Zoom Out"
          >
            <FaSearchMinus />
          </button>
          <span className="text-sm text-gray-400 w-14 text-center font-mono">{zoom}%</span>
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

        <div className="flex items-center gap-1">
          {/* Undo/Redo */}
          <button
            onClick={onUndo}
            disabled={!canUndo}
            className={`btn btn-ghost btn-sm ${!canUndo ? 'opacity-40' : ''}`}
            title="Undo"
          >
            <FaUndo />
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            className={`btn btn-ghost btn-sm ${!canRedo ? 'opacity-40' : ''}`}
            title="Redo"
          >
            <FaRedo />
          </button>

          <div className="w-px h-6 bg-gray-700 mx-1" />

          {/* Compare */}
          {processedImage && (
            <button
              onClick={onCompare}
              className="btn btn-ghost btn-sm"
              title="Compare Before/After"
            >
              <FaColumns />
            </button>
          )}

          {/* Annotate */}
          <button
            onClick={onAnnotate}
            className="btn btn-ghost btn-sm"
            title="Annotate Image"
          >
            <FaPen />
          </button>

          {/* Download */}
          <button
            onClick={handleDownload}
            className="btn btn-ghost btn-sm"
            title="Download"
          >
            <FaDownload />
          </button>

          <div className="w-px h-6 bg-gray-700 mx-1" />

          {/* Reset */}
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
      <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-gray-800/50">
        <div 
          className="relative"
          style={{ 
            transform: `scale(${zoom / 100})`, 
            transformOrigin: 'center',
            transition: 'transform 0.2s ease'
          }}
        >
          <img
            src={displayImage}
            alt="Preview"
            className="max-w-full max-h-full object-contain rounded shadow-2xl"
            draggable={false}
          />
          
          {/* Processed Badge */}
          {processedImage && (
            <span className="absolute top-2 right-2 bg-primary-500/90 px-2 py-1 rounded text-xs text-white font-medium">
              Processed
            </span>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="px-4 py-2 border-t border-gray-700 flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <span className={processedImage ? 'text-primary-400' : ''}>
            {processedImage ? '● Processed' : '○ Original'}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span>Zoom: {zoom}%</span>
          {processedImage && (
            <span className="text-primary-400">
              Click Compare to see before/after
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export default ImageViewer
