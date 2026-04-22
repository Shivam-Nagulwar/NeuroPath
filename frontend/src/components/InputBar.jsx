/**
 * InputBar — sticky bottom input with:
 *   - Auto-growing textarea
 *   - MRI image attachment button (Step 2)
 *   - Send button
 *   - MRI file preview chip
 */

import { useRef, useState } from 'react'
import { Send, Paperclip, X } from 'lucide-react'

export default function InputBar({
  onSendMessage,
  onSendMRI,
  isLoading,
  currentStep,
}) {
  const [text, setText]           = useState('')
  const [mriFile, setMriFile]     = useState(null)
  const [mriPreview, setMriPreview] = useState(null)
  const textareaRef               = useRef(null)
  const fileInputRef              = useRef(null)

  // ── Auto-grow textarea ──────────────────────────────────
  function handleTextChange(e) {
    setText(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 120) + 'px'
    }
  }

  // ── MRI file selected ───────────────────────────────────
  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setMriFile(file)
    const reader = new FileReader()
    reader.onload = (ev) => setMriPreview(ev.target.result)
    reader.readAsDataURL(file)
    // reset input so same file can be re-selected
    e.target.value = ''
  }

  function removeMriFile() {
    setMriFile(null)
    setMriPreview(null)
  }

  // ── Submit ──────────────────────────────────────────────
  function handleSubmit() {
    if (isLoading) return

    if (mriFile) {
      onSendMRI(mriFile, mriPreview)
      setMriFile(null)
      setMriPreview(null)
      setText('')
      resetHeight()
      return
    }

    if (text.trim()) {
      onSendMessage(text.trim())
      setText('')
      resetHeight()
    }
  }

  function resetHeight() {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const canSend = !isLoading && (text.trim().length > 0 || !!mriFile)

  const placeholder = mriFile
    ? 'Press send to analyse this MRI scan…'
    : currentStep === 2
    ? 'Upload an MRI image using 📎, or type a message…'
    : 'Describe your symptoms… (Shift+Enter for new line)'

  return (
    <div className="input-area">
      {/* MRI file preview chip */}
      {mriFile && mriPreview && (
        <div style={{ marginBottom: '8px' }}>
          <div className="mri-preview-chip">
            <img src={mriPreview} alt="MRI preview" />
            <span>{mriFile.name}</span>
            <button onClick={removeMriFile}><X size={12} /></button>
          </div>
        </div>
      )}

      <div className="input-bar">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          disabled={isLoading}
        />

        <div className="input-actions">
          {/* Attach MRI */}
          <div className="btn-attach" title="Upload MRI scan">
            <Paperclip size={15} />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleFileChange}
            />
          </div>

          {/* Send */}
          <button
            className="btn-send"
            onClick={handleSubmit}
            disabled={!canSend}
            title="Send"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  )
}
