/**
 * GatekeeperRejectionCard
 * Shown inline in the chat when the uploaded image fails the
 * brain MRI validation check (gatekeeper model returns non_brain_mri).
 */

import { ShieldX, AlertTriangle, CheckCircle2 } from 'lucide-react'

export default function GatekeeperRejectionCard({ data }) {
  if (!data) return null
  const { title, message, suggestions, gate_confidence } = data

  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid rgba(255, 69, 96, 0.4)',
      borderRadius: 'var(--radius-xl)',
      overflow: 'hidden',
      animation: 'fadeSlideIn 0.3s ease',
    }}>

      {/* Header */}
      <div style={{
        padding: '14px 18px',
        borderBottom: '1px solid rgba(255,69,96,0.2)',
        background: 'rgba(255,69,96,0.06)',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
      }}>
        <div style={{
          width: 32, height: 32,
          borderRadius: 'var(--radius-sm)',
          background: 'var(--red-dim)',
          border: '1px solid rgba(255,69,96,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <ShieldX size={16} color="var(--red)" />
        </div>
        <div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '14px',
            fontWeight: 700,
            color: 'var(--red)',
          }}>
            {title || 'Invalid Image'}
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            color: 'var(--text-muted)',
            letterSpacing: '0.08em',
            marginTop: '2px',
          }}>
            GATEKEEPER · MOBILENETV2
            {gate_confidence !== undefined && (
              <span style={{ marginLeft: '8px', color: 'var(--red)', opacity: 0.7 }}>
                Brain MRI confidence: {gate_confidence}%
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Message */}
      <div style={{ padding: '16px 18px' }}>
        <div style={{
          display: 'flex',
          gap: '10px',
          padding: '12px 14px',
          background: 'var(--red-dim)',
          border: '1px solid rgba(255,69,96,0.2)',
          borderRadius: 'var(--radius-md)',
          marginBottom: '14px',
        }}>
          <AlertTriangle size={15} color="var(--red)" style={{ flexShrink: 0, marginTop: 1 }} />
          <p style={{ fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.6, margin: 0 }}>
            {message}
          </p>
        </div>

        {/* Suggestions */}
        {suggestions?.length > 0 && (
          <div>
            <p style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              color: 'var(--cyan)',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              marginBottom: '10px',
            }}>
              ◈ What to check
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
              {suggestions.map((s, i) => (
                <div key={i} style={{
                  display: 'flex',
                  gap: '8px',
                  alignItems: 'flex-start',
                }}>
                  <CheckCircle2 size={13} color="var(--text-muted)"
                    style={{ flexShrink: 0, marginTop: 2 }} />
                  <span style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    {s}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Retry prompt */}
        <div style={{
          marginTop: '16px',
          padding: '10px 14px',
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
          fontSize: '12px',
          color: 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span>📎</span>
          Use the attachment button below to upload a valid brain MRI scan and try again.
        </div>
      </div>

    </div>
  )
}
