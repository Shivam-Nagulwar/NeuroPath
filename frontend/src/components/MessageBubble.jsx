/**
 * MessageBubble — renders a single chat message.
 * Handles: user text, assistant text, triage summary blocks,
 *          MRI result cards, report panels, and loading indicator.
 */

import MRIResultCard from './MRIResultCard'
import ReportPanel from './ReportPanel'
import GatekeeperRejectionCard from './GatekeeperRejectionCard'

function extractVisibleText(text) {
  // Strip the raw triage summary block from the visible bubble text
  if (!text) return ''
  const start = text.indexOf('---TRIAGE SUMMARY---')
  if (start === -1) return text
  return text.slice(0, start).trim()
}

function extractTriageBlock(text) {
  if (!text) return null
  if (!text.includes('---TRIAGE SUMMARY---')) return null
  const start = text.indexOf('---TRIAGE SUMMARY---')
  const end = text.indexOf('---END SUMMARY---')
  if (end === -1) return null
  return text.slice(start, end + '---END SUMMARY---'.length)
}

export function TypingIndicator() {
  return (
    <div className="message-row">
      <div className="avatar ai">NP</div>
      <div className="bubble ai">
        <div className="typing-indicator">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  )
}

export default function MessageBubble({ message }) {
  const { role, content, type, data } = message

  // ── Special message types injected by App.jsx ────────────
  if (type === 'gatekeeper_rejection') {
    return (
      <div className="message-row">
        <div className="avatar ai">NP</div>
        <div style={{ flex: 1, maxWidth: '85%' }}>
          <GatekeeperRejectionCard data={data} />
        </div>
      </div>
    )
  }

  if (type === 'mri_result') {
    return (
      <div className="message-row">
        <div className="avatar ai">NP</div>
        <div style={{ flex: 1, maxWidth: '85%' }}>
          <MRIResultCard data={data} />
        </div>
      </div>
    )
  }

  if (type === 'report') {
    return (
      <div className="message-row">
        <div className="avatar ai">NP</div>
        <div style={{ flex: 1, maxWidth: '85%' }}>
          <ReportPanel reportText={data.report} generatedAt={data.generated_at} />
        </div>
      </div>
    )
  }

  // ── Standard text messages ────────────────────────────────
  const isUser = role === 'user'
  const visibleText = isUser ? content : extractVisibleText(content)
  const triageBlock = !isUser ? extractTriageBlock(content) : null

  return (
    <div className={`message-row ${isUser ? 'user' : ''}`}>
      <div className={`avatar ${isUser ? 'user' : 'ai'}`}>
        {isUser ? 'YOU' : 'NP'}
      </div>

      <div className={`bubble ${isUser ? 'user' : 'ai'}`}>
        {/* Main message text */}
        <div style={{ whiteSpace: 'pre-wrap' }}>{visibleText}</div>

        {/* Triage summary rendered as a styled block */}
        {triageBlock && (
          <div className="triage-block">
            <div className="triage-block-header">
              <span>◈</span> Triage Summary
            </div>
            {triageBlock
              .replace('---TRIAGE SUMMARY---', '')
              .replace('---END SUMMARY---', '')
              .trim()}
          </div>
        )}
      </div>
    </div>
  )
}