import { useState, useCallback, useRef, useEffect } from 'react'
import UrlInput from './components/UrlInput'
import ProgressPanel from './components/ProgressPanel'
import DownloadButton from './components/DownloadButton'

const API_BASE = ''

const STAGES_LABELS = {
  pending: 'Queued',
  downloading: 'Downloading files from Google Drive',
  extracting_audio: 'Extracting audio from video',
  transcribing: 'Transcribing voiceover with Whisper',
  detecting_scenes: 'Detecting scene changes',
  extracting_keyframes: 'Extracting keyframe images',
  matching: 'Matching voice to scenes (CLIP + AI)',
  rendering: 'Rendering final video',
  cleaning: 'Cleaning up temporary files',
  completed: 'Complete!',
  failed: 'Failed',
}

export default function App() {
  const [movieUrl, setMovieUrl] = useState('')
  const [voiceoverUrl, setVoiceoverUrl] = useState('')
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const pollingRef = useRef(null)

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  const startPolling = useCallback((id) => {
    stopPolling()
    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${id}`)
        if (!res.ok) throw new Error('Status fetch failed')
        const data = await res.json()
        setStatus(data)
        if (data.status === 'completed' || data.status === 'failed') {
          stopPolling()
          setLoading(false)
        }
      } catch (e) {
        console.error('Polling error:', e)
      }
    }, 2000)
  }, [stopPolling])

  const handleSubmit = async () => {
    if (!movieUrl.trim() || !voiceoverUrl.trim()) return
    setError(null)
    setLoading(true)
    setStatus(null)
    setJobId(null)

    try {
      const res = await fetch(`${API_BASE}/process-gdrive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          movie_url: movieUrl.trim(),
          voiceover_url: voiceoverUrl.trim(),
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Submission failed')
      }
      const data = await res.json()
      setJobId(data.job_id)
      startPolling(data.job_id)
    } catch (e) {
      setError(e.message)
      setLoading(false)
    }
  }

  const handleReset = () => {
    stopPolling()
    setJobId(null)
    setStatus(null)
    setLoading(false)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-cinema-900 text-white">
      <header className="border-b border-cinema-700 bg-cinema-800/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center text-lg font-bold">
              M
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">Movie Recap Generator</h1>
              <p className="text-xs text-cinema-400 font-mono">AI-Powered Scene Matching</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-cinema-400">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
            {status ? status.status : 'Ready'}
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-12">
        {!jobId && (
          <div className="space-y-8">
            <div className="text-center space-y-3 mb-12">
              <h2 className="text-3xl font-bold tracking-tight">
                Turn any movie into a{' '}
                <span className="text-accent">recap video</span>
              </h2>
              <p className="text-cinema-300 max-w-xl mx-auto text-sm leading-relaxed">
                Upload a movie and a voiceover script. Our AI matches your narration to the perfect scenes
                using CLIP embeddings + NVIDIA Vision verification.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <UrlInput
                label="Movie URL"
                placeholder="https://drive.google.com/file/d/..."
                value={movieUrl}
                onChange={setMovieUrl}
                icon="🎬"
              />
              <UrlInput
                label="Voiceover URL"
                placeholder="https://drive.google.com/file/d/..."
                value={voiceoverUrl}
                onChange={setVoiceoverUrl}
                icon="🎙️"
              />
            </div>

            {error && (
              <div className="bg-accent-muted/20 border border-accent/30 rounded-lg p-4 text-sm text-red-300">
                {error}
              </div>
            )}

            <div className="flex justify-center">
              <button
                onClick={handleSubmit}
                disabled={loading || !movieUrl.trim() || !voiceoverUrl.trim()}
                className="px-10 py-3.5 rounded-xl bg-accent hover:bg-accent-hover disabled:bg-cinema-600 disabled:text-cinema-400 text-white font-semibold text-sm transition-all duration-200 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Processing...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Generate Recap
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {jobId && status && (
          <div className="max-w-lg mx-auto space-y-8">
            <div className="text-center">
              <h2 className="text-xl font-bold">Processing Job</h2>
              <p className="text-cinema-400 text-xs font-mono mt-1">{jobId}</p>
            </div>

            <ProgressPanel
              status={status.status}
              progress={status.progress}
              stage={STAGES_LABELS[status.status] || status.current_stage}
              error={status.error}
            />

            {status.status === 'completed' && (
              <div className="text-center space-y-4">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 text-green-400 text-sm border border-green-500/20">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                  Recap video is ready
                </div>
                <DownloadButton jobId={jobId} />
                <div>
                  <button
                    onClick={handleReset}
                    className="text-cinema-400 hover:text-white text-sm transition-colors underline underline-offset-2"
                  >
                    Create another recap
                  </button>
                </div>
              </div>
            )}

            {status.status === 'failed' && (
              <div className="text-center space-y-3">
                <button
                  onClick={handleReset}
                  className="px-6 py-2.5 rounded-xl bg-cinema-700 hover:bg-cinema-600 text-white text-sm transition-colors"
                >
                  Try Again
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-cinema-800 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-cinema-500">
          <span>Movie Recap Generator v1.0</span>
          <span>Powered by CLIP + NVIDIA Vision API</span>
        </div>
      </footer>
    </div>
  )
}
