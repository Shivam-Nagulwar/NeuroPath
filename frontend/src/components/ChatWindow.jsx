/**
 * ChatWindow — scrollable message list.
 * Shows empty state when no messages, auto-scrolls to bottom on new messages.
 */

import { useEffect, useRef } from 'react'
import MessageBubble, { TypingIndicator } from './MessageBubble'

const SUGGESTIONS = [
  "I've had severe headaches for 3 weeks",
  "Sudden vision changes in my left eye",
  "Dizziness and balance problems lately",
  "Memory issues and personality changes",
]

export default function ChatWindow({ messages, isLoading, onSuggestion }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="chat-window">
        <div className="empty-state">
          <div className="empty-icon">🧠</div>
          <div className="empty-title">NeuroPath AI</div>
          <p className="empty-sub">
            Start by describing your neurological symptoms. The AI will assess
            your risk level before guiding you to MRI analysis.
          </p>
          <div className="suggestion-chips">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="chip"
                onClick={() => onSuggestion(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}
      {isLoading && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  )
}
