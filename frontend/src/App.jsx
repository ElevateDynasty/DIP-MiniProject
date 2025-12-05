import { useState, useCallback, useEffect } from 'react'
import Header from './components/Header'
import ImageUploader from './components/ImageUploader'
import ImageViewer from './components/ImageViewer'
import OperationsPanel from './components/OperationsPanel'
import HistoryPanel from './components/HistoryPanel'
import CompareSlider from './components/CompareSlider'
import AnnotationTools from './components/AnnotationTools'
import { processImage } from './api/imageApi'

function App() {
  const [originalImage, setOriginalImage] = useState(null)
  const [originalFile, setOriginalFile] = useState(null)
  const [processedImage, setProcessedImage] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [darkMode, setDarkMode] = useState(true)
  const [showCompare, setShowCompare] = useState(false)
  const [showAnnotations, setShowAnnotations] = useState(false)
  
  // History for undo/redo
  const [history, setHistory] = useState([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [operationHistory, setOperationHistory] = useState([])

  // Apply theme
  useEffect(() => {
    document.documentElement.classList.toggle('light-mode', !darkMode)
  }, [darkMode])

  const handleImageUpload = useCallback((file) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setOriginalImage(e.target.result)
      setOriginalFile(file)
      setProcessedImage(null)
      setError(null)
      setHistory([e.target.result])
      setHistoryIndex(0)
      setOperationHistory([{ name: 'Original', time: new Date() }])
    }
    reader.readAsDataURL(file)
  }, [])

  const handleProcess = useCallback(async (endpoint, params = {}) => {
    if (!originalFile) return

    setIsProcessing(true)
    setError(null)

    try {
      // Use the current image (processed or original) for chaining operations
      const currentImage = processedImage || originalImage
      
      // Convert data URL to blob if we're using processed image
      let fileToProcess = originalFile
      if (processedImage) {
        const response = await fetch(processedImage)
        const blob = await response.blob()
        fileToProcess = new File([blob], 'image.png', { type: 'image/png' })
      }

      const result = await processImage(fileToProcess, endpoint, params)
      setProcessedImage(result)
      
      // Add to history
      const newHistory = history.slice(0, historyIndex + 1)
      newHistory.push(result)
      setHistory(newHistory)
      setHistoryIndex(newHistory.length - 1)
      
      // Add to operation history
      const opName = endpoint.split('/').pop().replace(/-/g, ' ')
      setOperationHistory(prev => [...prev, { 
        name: opName.charAt(0).toUpperCase() + opName.slice(1), 
        time: new Date(),
        params 
      }])
    } catch (err) {
      setError(err.message || 'Failed to process image')
      console.error('Processing error:', err)
    } finally {
      setIsProcessing(false)
    }
  }, [originalFile, processedImage, originalImage, history, historyIndex])

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1
      setHistoryIndex(newIndex)
      if (newIndex === 0) {
        setProcessedImage(null)
      } else {
        setProcessedImage(history[newIndex])
      }
    }
  }, [historyIndex, history])

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1
      setHistoryIndex(newIndex)
      setProcessedImage(history[newIndex])
    }
  }, [historyIndex, history])

  const handleReset = useCallback(() => {
    setOriginalImage(null)
    setOriginalFile(null)
    setProcessedImage(null)
    setError(null)
    setHistory([])
    setHistoryIndex(-1)
    setOperationHistory([])
    setShowCompare(false)
    setShowAnnotations(false)
  }, [])

  const handleHistoryJump = useCallback((index) => {
    setHistoryIndex(index)
    if (index === 0) {
      setProcessedImage(null)
    } else {
      setProcessedImage(history[index])
    }
  }, [history])

  const canUndo = historyIndex > 0
  const canRedo = historyIndex < history.length - 1

  return (
    <div className={`min-h-screen transition-colors ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <Header 
        darkMode={darkMode} 
        setDarkMode={setDarkMode}
        canUndo={canUndo}
        canRedo={canRedo}
        onUndo={handleUndo}
        onRedo={handleRedo}
      />

      <main className="container mx-auto px-4 py-6">
        {error && (
          <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400 animate-fadeIn">
            {error}
          </div>
        )}

        {!originalImage ? (
          <ImageUploader onUpload={handleImageUpload} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-[calc(100vh-12rem)]">
            {/* Operations Panel */}
            <div className="lg:col-span-1 h-full overflow-hidden">
              <OperationsPanel
                onProcess={handleProcess}
                isProcessing={isProcessing}
              />
            </div>

            {/* Image Viewer */}
            <div className="lg:col-span-2 h-full">
              {showCompare && processedImage ? (
                <CompareSlider
                  originalImage={originalImage}
                  processedImage={processedImage}
                  onClose={() => setShowCompare(false)}
                />
              ) : showAnnotations ? (
                <AnnotationTools
                  image={processedImage || originalImage}
                  onSave={(annotatedImage) => {
                    setProcessedImage(annotatedImage)
                    setShowAnnotations(false)
                  }}
                  onClose={() => setShowAnnotations(false)}
                />
              ) : (
                <ImageViewer
                  originalImage={originalImage}
                  processedImage={processedImage}
                  onReset={handleReset}
                  onCompare={() => setShowCompare(true)}
                  onAnnotate={() => setShowAnnotations(true)}
                  canUndo={canUndo}
                  canRedo={canRedo}
                  onUndo={handleUndo}
                  onRedo={handleRedo}
                />
              )}
            </div>

            {/* History Panel */}
            <div className="lg:col-span-1 h-full overflow-hidden">
              <HistoryPanel
                history={history}
                historyIndex={historyIndex}
                operationHistory={operationHistory}
                onJump={handleHistoryJump}
                onReset={handleReset}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
