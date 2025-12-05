import { FaImage, FaGithub } from 'react-icons/fa'

function Header() {
  return (
    <header className="bg-slate-800/50 backdrop-blur-md border-b border-slate-700/50 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
              <FaImage className="text-white text-xl" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">DIP Project</h1>
              <p className="text-xs text-gray-400">Digital Image Processing</p>
            </div>
          </div>

          {/* Links */}
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/ElevateDynasty/DIP-Demo"
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
