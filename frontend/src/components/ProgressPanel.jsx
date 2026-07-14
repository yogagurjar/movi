const STATUS_ICONS = {
  pending: '⏳',
  downloading: '⬇️',
  extracting_audio: '🔊',
  transcribing: '📝',
  detecting_scenes: '🎬',
  extracting_keyframes: '🖼️',
  matching: '🧠',
  rendering: '🎞️',
  cleaning: '🧹',
  completed: '✅',
  failed: '❌',
}

export default function ProgressPanel({ status, progress, stage, error }) {
  const isComplete = status === 'completed'
  const isFailed = status === 'failed'
  const isActive = !isComplete && !isFailed

  const barColor = isFailed
    ? 'bg-red-500'
    : isComplete
    ? 'bg-green-500'
    : 'bg-accent'

  return (
    <div className="bg-cinema-800 rounded-2xl border border-cinema-700 p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">{STATUS_ICONS[status] || '⚙️'}</span>
          <span className="text-sm font-medium">{stage}</span>
        </div>
        <span className="text-xs font-mono text-cinema-400 tabular-nums">
          {Math.round(progress)}%
        </span>
      </div>

      <div className="h-2.5 rounded-full bg-cinema-900 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${barColor} ${
            isActive ? 'progress-striped' : ''
          }`}
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>

      {isActive && (
        <p className="text-xs text-cinema-400 text-center animate-pulse">
          Please wait, this may take several minutes...
        </p>
      )}

      {isFailed && error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <p className="text-xs text-red-300 font-mono leading-relaxed">{error}</p>
        </div>
      )}
    </div>
  )
}
