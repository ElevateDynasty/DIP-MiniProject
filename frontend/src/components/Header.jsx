import { FaImage, FaGithub, FaMoon, FaSun, FaUndo, FaRedo } from 'react-icons/fa'

function Header({ darkMode, setDarkMode, canUndo, canRedo, onUndo, onRedo }) {
  return (
    <header className={`backdrop-blur-md border-b sticky top-0 z-50 transition-colors ${
      darkMode ? 'bg-slate-800/50 border-slate-700/50' : 'bg-white/80 border-gray-200'
    }`}>
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
              <FaImage className="text-white text-xl" />
            </div>
            <div>
              <h1 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                DIP Project
              </h1>
              <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Digital Image Processing
              </p>
            </div>
          </div>

          {/* Center - Undo/Redo */}
          <div className="flex items-center gap-1">
            <button
              onClick={onUndo}
              disabled={!canUndo}
              className={`btn btn-ghost ${!canUndo ? 'opacity-40' : ''}`}
              title="Undo (Ctrl+Z)"
            >
              <FaUndo />
            </button>
            <button
              onClick={onRedo}
              disabled={!canRedo}
              className={`btn btn-ghost ${!canRedo ? 'opacity-40' : ''}`}
              title="Redo (Ctrl+Y)"
            >
              <FaRedo />
            </button>
          </div>

          {/* Right - Links & Theme Toggle */}
          <div className="flex items-center gap-2">
            {/* Theme Toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`btn btn-ghost p-2 rounded-full ${
                darkMode ? 'text-yellow-400' : 'text-gray-600'
              }`}
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? <FaSun className="text-lg" /> : <FaMoon className="text-lg" />}
            </button>

            <a
              href="https://github.com/ElevateDynasty/DIP-MiniProject"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost"
            >
              <FaGithub className="text-lg" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
