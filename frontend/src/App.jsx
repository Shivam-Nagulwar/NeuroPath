/**
 * NeuroPath — App.jsx
 * Root component. Holds all global state and orchestrates API calls.
 */

import { useState, useCallback } from 'react'
import Sidebar    from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import InputBar   from './components/InputBar'
import { sendChatMessage, analyseMRI, generateReport, clearSession } from './api/neuropath'

const STEP_TITLES = {
  1: '🩺 Symptom Triage — Chat with NeuroPath AI',
  2: '🔬 MRI Analysis — Upload your brain scan',
  3: '📄 Full Report — Integrated diagnosis',
}

export default function App() {
  // ── Global State ──────────────────────────────────────────
  const [messages,           setMessages]           = useState([])
  const [apiHistory,         setApiHistory]         = useState([])   // format for backend
  const [triageSummary,      setTriageSummary]      = useState('')
  const [mriDone,            setMriDone]            = useState(false)
  const [currentStep,        setCurrentStep]        = useState(1)
  const [isLoading,          setIsLoading]          = useState(false)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)
  const [backendStatus,      setBackendStatus]      = useState('online') // 'online' | 'error'

  // ── Step 1 — Send chat message ────────────────────────────
  const handleSendMessage = useCallback(async (text) => {
    // Optimistically add user message
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setIsLoading(true)

    try {
      const result = await sendChatMessage(text, apiHistory)

      // Update API history (used for next request)
      setApiHistory(result.history)

      // Add assistant reply to visible messages
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.reply,
      }])

      // Save triage summary if generated
      if (result.triage_summary) {
        setTriageSummary(result.triage_summary)
      }

      setBackendStatus('online')
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ Connection error: ${err.message}\n\nMake sure the backend is running on localhost:8000`,
      }])
      setBackendStatus('error')
    } finally {
      setIsLoading(false)
    }
  }, [apiHistory])

  // ── Step 2 — Send MRI image ───────────────────────────────
  const handleSendMRI = useCallback(async (file, previewDataUrl) => {
    // Show a user message with the image name
    setMessages(prev => [...prev, {
      role: 'user',
      content: `📎 Uploaded MRI scan: ${file.name}`,
    }])
    setIsLoading(true)
    setCurrentStep(2)

    try {
      const result = await analyseMRI(file)

      // Inject MRI result card as a special message type
      setMessages(prev => [...prev, {
        role: 'assistant',
        type: 'mri_result',
        content: '',
        data: {
          ...result,
          original_b64: previewDataUrl
            ? previewDataUrl.split(',')[1]   // strip "data:image/...;base64,"
            : null,
        },
      }])

      setMriDone(true)
      setBackendStatus('online')
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ MRI analysis failed: ${err.message}`,
      }])
      setBackendStatus('error')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // ── Step 3 — Generate report ──────────────────────────────
  const handleGenerateReport = useCallback(async () => {
    setIsGeneratingReport(true)
    setCurrentStep(3)

    try {
      const result = await generateReport()

      setMessages(prev => [...prev, {
        role: 'assistant',
        type: 'report',
        content: '',
        data: result,
      }])

      setBackendStatus('online')
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ Report generation failed: ${err.message}`,
      }])
      setBackendStatus('error')
    } finally {
      setIsGeneratingReport(false)
    }
  }, [])

  // ── Clear session ─────────────────────────────────────────
  const handleClear = useCallback(async () => {
    if (!window.confirm('Clear all messages and reset the session?')) return
    await clearSession().catch(() => {})
    setMessages([])
    setApiHistory([])
    setTriageSummary('')
    setMriDone(false)
    setCurrentStep(1)
  }, [])

  // ── Suggestion chip clicked ───────────────────────────────
  const handleSuggestion = useCallback((text) => {
    handleSendMessage(text)
  }, [handleSendMessage])

  // ─────────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* Sidebar */}
      <Sidebar
        currentStep={currentStep}
        setCurrentStep={setCurrentStep}
        triageSummary={triageSummary}
        mriDone={mriDone}
        onGenerateReport={handleGenerateReport}
        onClear={handleClear}
        isGeneratingReport={isGeneratingReport}
      />

      {/* Main area */}
      <div className="main-area">

        {/* Top bar */}
        <div className="topbar">
          <div className="topbar-title">
            {STEP_TITLES[currentStep]}
          </div>
          <div className="topbar-status">
            <div className={`status-dot ${backendStatus === 'error' ? 'error' : ''}`}
              style={backendStatus === 'error' ? {
                background: 'var(--red)',
                boxShadow: '0 0 8px var(--red)',
              } : {}}
            />
            {backendStatus === 'error' ? 'Backend offline' : 'System online'}
          </div>
        </div>

        {/* Chat window */}
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          onSuggestion={handleSuggestion}
        />

        {/* Input bar */}
        <InputBar
          onSendMessage={handleSendMessage}
          onSendMRI={handleSendMRI}
          isLoading={isLoading}
          currentStep={currentStep}
        />

      </div>
    </div>
  )
}
