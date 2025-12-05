import { FaHistory, FaTrash, FaClock } from 'react-icons/fa'

function HistoryPanel({ history, historyIndex, operationHistory, onJump, onReset }) {
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  return (
    <div className="card h-full flex flex-col animate-fadeIn">
      <div className="p-3 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FaHistory className="text-primary-400" />
          <div>
            <h2 className="text-lg font-semibold text-white">History</h2>
            <p className="text-xs text-gray-500">{history.length} states</p>
          </div>
        </div>
        {history.length > 1 && (
          <button
            onClick={onReset}
            className="btn btn-ghost btn-sm text-red-400 hover:text-red-300"
            title="Clear all"
          >
            <FaTrash />
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {operationHistory.map((op, index) => (
          <button
            key={index}
            onClick={() => onJump(index)}
            className={`w-full p-3 rounded-lg text-left transition-all ${
              index === historyIndex
                ? 'bg-primary-600 text-white ring-2 ring-primary-400'
                : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  index === historyIndex ? 'bg-white text-primary-600' : 'bg-gray-700 text-gray-400'
                }`}>
                  {index + 1}
                </span>
                <span className="font-medium text-sm truncate">{op.name}</span>
              </div>
              {index === historyIndex && (
                <span className="text-xs bg-primary-500 px-2 py-0.5 rounded">Current</span>
              )}
            </div>
            <div className="flex items-center gap-1 mt-1 ml-8 text-xs opacity-60">
              <FaClock className="text-[10px]" />
              <span>{formatTime(op.time)}</span>
              {op.params && Object.keys(op.params).length > 0 && (
                <span className="ml-2">
                  ({Object.entries(op.params).map(([k, v]) => `${k}: ${v}`).join(', ')})
                </span>
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Thumbnail Preview */}
      {history.length > 0 && (
        <div className="p-3 border-t border-gray-700">
          <p className="text-xs text-gray-500 mb-2">Preview</p>
          <div className="grid grid-cols-4 gap-1">
            {history.slice(-8).map((img, idx) => {
              const actualIndex = history.length - 8 + idx
              const displayIndex = actualIndex < 0 ? idx : actualIndex
              return (
                <button
                  key={idx}
                  onClick={() => onJump(displayIndex)}
                  className={`aspect-square rounded overflow-hidden border-2 transition-all ${
                    displayIndex === historyIndex
                      ? 'border-primary-500 ring-2 ring-primary-400/50'
                      : 'border-gray-700 hover:border-gray-500'
                  }`}
                >
                  <img
                    src={img}
                    alt={`State ${displayIndex + 1}`}
                    className="w-full h-full object-cover"
                  />
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default HistoryPanel
