import { useState, useRef, useEffect } from 'react'
import { FaTimes, FaPen, FaSquare, FaCircle, FaFont, FaEraser, FaSave, FaUndo } from 'react-icons/fa'

function AnnotationTools({ image, onSave, onClose }) {
  const canvasRef = useRef(null)
  const [tool, setTool] = useState('pen')
  const [color, setColor] = useState('#ff0000')
  const [lineWidth, setLineWidth] = useState(3)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPos, setStartPos] = useState({ x: 0, y: 0 })
  const [history, setHistory] = useState([])
  const [text, setText] = useState('')

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.src = image
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      saveState()
    }
  }, [image])

  const saveState = () => {
    const canvas = canvasRef.current
    setHistory(prev => [...prev, canvas.toDataURL()])
  }

  const undo = () => {
    if (history.length > 1) {
      const newHistory = history.slice(0, -1)
      setHistory(newHistory)
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      const img = new Image()
      img.src = newHistory[newHistory.length - 1]
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0)
      }
    }
  }

  const getPos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    const clientY = e.touches ? e.touches[0].clientY : e.clientY
    
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    }
  }

  const startDrawing = (e) => {
    const pos = getPos(e)
    setIsDrawing(true)
    setStartPos(pos)
    
    if (tool === 'pen' || tool === 'eraser') {
      const ctx = canvasRef.current.getContext('2d')
      ctx.beginPath()
      ctx.moveTo(pos.x, pos.y)
    }
  }

  const draw = (e) => {
    if (!isDrawing) return
    const pos = getPos(e)
    const ctx = canvasRef.current.getContext('2d')

    if (tool === 'pen') {
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.lineCap = 'round'
      ctx.lineTo(pos.x, pos.y)
      ctx.stroke()
    } else if (tool === 'eraser') {
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = lineWidth * 3
      ctx.lineCap = 'round'
      ctx.lineTo(pos.x, pos.y)
      ctx.stroke()
    }
  }

  const stopDrawing = (e) => {
    if (!isDrawing) return
    setIsDrawing(false)
    
    const pos = getPos(e)
    const ctx = canvasRef.current.getContext('2d')
    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth
    ctx.fillStyle = color

    if (tool === 'rectangle') {
      const width = pos.x - startPos.x
      const height = pos.y - startPos.y
      ctx.strokeRect(startPos.x, startPos.y, width, height)
    } else if (tool === 'circle') {
      const radius = Math.sqrt(Math.pow(pos.x - startPos.x, 2) + Math.pow(pos.y - startPos.y, 2))
      ctx.beginPath()
      ctx.arc(startPos.x, startPos.y, radius, 0, 2 * Math.PI)
      ctx.stroke()
    } else if (tool === 'text' && text) {
      ctx.font = `${lineWidth * 8}px Arial`
      ctx.fillText(text, startPos.x, startPos.y)
    }

    saveState()
  }

  const handleSave = () => {
    const canvas = canvasRef.current
    onSave(canvas.toDataURL('image/png'))
  }

  const tools = [
    { id: 'pen', icon: FaPen, name: 'Pen' },
    { id: 'rectangle', icon: FaSquare, name: 'Rectangle' },
    { id: 'circle', icon: FaCircle, name: 'Circle' },
    { id: 'text', icon: FaFont, name: 'Text' },
    { id: 'eraser', icon: FaEraser, name: 'Eraser' },
  ]

  const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ffffff', '#000000']

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      {/* Header */}
      <div className="p-3 border-b border-gray-700 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Annotate</h2>
        <div className="flex items-center gap-2">
          <button onClick={undo} className="btn btn-ghost btn-sm" title="Undo">
            <FaUndo />
          </button>
          <button onClick={handleSave} className="btn btn-primary btn-sm">
            <FaSave /> Save
          </button>
          <button onClick={onClose} className="btn btn-ghost btn-sm">
            <FaTimes />
          </button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="p-2 border-b border-gray-700 flex flex-wrap items-center gap-2">
        {/* Tools */}
        <div className="flex gap-1">
          {tools.map(t => (
            <button
              key={t.id}
              onClick={() => setTool(t.id)}
              className={`p-2 rounded transition-colors ${
                tool === t.id ? 'bg-primary-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title={t.name}
            >
              <t.icon />
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-gray-700" />

        {/* Colors */}
        <div className="flex gap-1">
          {colors.map(c => (
            <button
              key={c}
              onClick={() => setColor(c)}
              className={`w-6 h-6 rounded-full border-2 transition-transform ${
                color === c ? 'border-white scale-110' : 'border-gray-600'
              }`}
              style={{ backgroundColor: c }}
            />
          ))}
        </div>

        <div className="w-px h-6 bg-gray-700" />

        {/* Line Width */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Size:</span>
          <input
            type="range"
            min="1"
            max="20"
            value={lineWidth}
            onChange={(e) => setLineWidth(parseInt(e.target.value))}
            className="w-20 accent-primary-500"
          />
          <span className="text-xs text-gray-400 w-4">{lineWidth}</span>
        </div>

        {/* Text Input */}
        {tool === 'text' && (
          <>
            <div className="w-px h-6 bg-gray-700" />
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text..."
              className="px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded text-white"
            />
          </>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-gray-800">
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="max-w-full max-h-full object-contain cursor-crosshair rounded shadow-lg"
        />
      </div>
    </div>
  )
}

export default AnnotationTools
