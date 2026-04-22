import { MessageSquare, ScanLine, FileText, CheckCircle, Circle, Loader } from 'lucide-react'

const STEPS = [
  {
    id: 1,
    title: 'Symptom Triage',
    desc: 'Chat with AI assistant',
    icon: MessageSquare,
  },
  {
    id: 2,
    title: 'MRI Analysis',
    desc: 'Upload & analyse scan',
    icon: ScanLine,
  },
  {
    id: 3,
    title: 'Full Report',
    desc: 'Integrated diagnosis',
    icon: FileText,
  },
]

function parseTriage(summary) {
  if (!summary) return null
  const get = (key) => {
    const match = summary.match(new RegExp(`${key}:\\s*(.+)`))
    return match ? match[1].trim() : '—'
  }
  return {
    risk:     get('RISK LEVEL'),
    symptoms: get('KEY SYMPTOMS'),
    duration: get('DURATION'),
    severity: get('SEVERITY SCORE'),
    mri:      get('MRI RECOMMENDATION'),
    action:   get('ADVISED ACTION'),
  }
}

export default function Sidebar({
  currentStep,
  setCurrentStep,
  triageSummary,
  mriDone,
  onGenerateReport,
  onClear,
  isGeneratingReport,
}) {
  const parsed = parseTriage(triageSummary)
  const riskClass = parsed
    ? parsed.risk.toLowerCase().includes('high') ? 'high'
    : parsed.risk.toLowerCase().includes('medium') ? 'medium'
    : 'low'
    : ''

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">🧠</div>
          <div>
            <div className="logo-text">Neuro<span>Path</span></div>
            <div className="logo-sub">AI Diagnostic System</div>
          </div>
        </div>
      </div>

      {/* Step Navigation */}
      <nav className="step-nav">
        <div className="step-nav-label">Workflow</div>
        {STEPS.map((step) => {
          const Icon = step.icon
          const isActive = currentStep === step.id
          const isDone =
            (step.id === 1 && !!triageSummary) ||
            (step.id === 2 && mriDone)

          return (
            <button
              key={step.id}
              className={`step-btn ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}
              onClick={() => setCurrentStep(step.id)}
            >
              <span className="step-num">0{step.id}</span>
              <div className="step-icon">
                {isDone
                  ? <CheckCircle size={14} />
                  : <Icon size={14} />}
              </div>
              <div className="step-info">
                <div className="step-title">{step.title}</div>
                <div className="step-desc">{step.desc}</div>
              </div>
            </button>
          )
        })}
      </nav>

      <div className="sidebar-divider" />

      {/* Triage Summary */}
      <div className="sidebar-section">
        <div className="sidebar-section-label">Triage Summary</div>

        {parsed ? (
          <div className={`triage-card risk-${riskClass}`}>
            <div className={`risk-badge ${riskClass}`}>
              <span>●</span>
              {parsed.risk}
            </div>

            <div className="triage-row">
              <span className="triage-key">Key Symptoms</span>
              <span className="triage-val">{parsed.symptoms}</span>
            </div>
            <div className="triage-row">
              <span className="triage-key">Duration</span>
              <span className="triage-val">{parsed.duration}</span>
            </div>
            <div className="triage-row">
              <span className="triage-key">Severity</span>
              <span className="triage-val">{parsed.severity} / 10</span>
            </div>
            <div className="triage-row">
              <span className="triage-key">MRI Rec.</span>
              <span className="triage-val">{parsed.mri}</span>
            </div>
          </div>
        ) : (
          <div style={{
            padding: '12px',
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius-md)',
            border: '1px dashed var(--border-subtle)',
            textAlign: 'center',
          }}>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Complete the triage chat to see the risk summary here
            </p>
          </div>
        )}
      </div>

      {/* Footer Buttons */}
      <div className="sidebar-footer">
        <button
          className="btn-report"
          onClick={onGenerateReport}
          disabled={!mriDone || isGeneratingReport}
        >
          {isGeneratingReport
            ? <><Loader size={14} style={{ animation: 'spin 1s linear infinite' }} /> Generating…</>
            : <><FileText size={14} /> Generate Report</>}
        </button>
        <button className="btn-clear" onClick={onClear}>
          Clear Session
        </button>
      </div>
    </aside>
  )
}
