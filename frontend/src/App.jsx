import { useState, useCallback } from 'react'
import Header from './components/Header'
import ImageUploader from './components/ImageUploader'
import ImageViewer from './components/ImageViewer'
import OperationsPanel from './components/OperationsPanel'
import { processImage } from './api/imageApi'

function App() {
  const [originalImage, setOriginalImage] = useState(null)
  const [processedImage, setProcessedImage] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentOperation, setCurrentOperation] = useState(null)
  const [error, setError] = useState(null)

  const handleImageUpload = useCallback((file) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setOriginalImage({
        file,
        url: e.target.result,
        name: file.name
      })
      setProcessedImage(null)
      setError(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleProcessImage = useCallback(async (endpoint, params = {}) => {
    if (!originalImage) return

    setIsProcessing(true)
    setError(null)
    setCurrentOperation(endpoint)

    try {
      const result = await processImage(originalImage.file, endpoint, params)
      setProcessedImage(result)
    } catch (err) {
      console.error('Processing error:', err)
      setError(err.message || 'Failed to process image')
    } finally {
      setIsProcessing(false)
    }
  }, [originalImage])

  const handleDownload = useCallback(() => {
    if (!processedImage) return

    const link = document.createElement('a')
    link.href = processedImage
    link.download = `processed_${originalImage?.name || 'image.png'}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [processedImage, originalImage])

  const handleReset = useCallback(() => {
    setProcessedImage(null)
    setCurrentOperation(null)
    setError(null)
  }, [])

  const handleClear = useCallback(() => {
    setOriginalImage(null)
    setProcessedImage(null)
    setCurrentOperation(null)
    setError(null)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <Header />
      
      <main className="container mx-auto px-4 py-6">
        {!originalImage ? (
          <ImageUploader onUpload={handleImageUpload} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Operations Panel */}
            <div className="lg:col-span-1">
              <OperationsPanel
                onProcess={handleProcessImage}
                isProcessing={isProcessing}
                currentOperation={currentOperation}
                onReset={handleReset}
                onClear={handleClear}
              />
            </div>

            {/* Image Viewer */}
            <div className="lg:col-span-3">
              <ImageViewer
                originalImage={originalImage?.url}
                processedImage={processedImage}
                isProcessing={isProcessing}
                error={error}
                onDownload={handleDownload}
              />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-4 text-gray-500 text-sm">
        <p>DIP Project - Digital Image Processing with React & FastAPI</p>
      </footer>
    </div>
  )
}

export default App
