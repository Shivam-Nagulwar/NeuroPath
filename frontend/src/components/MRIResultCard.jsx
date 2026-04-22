/**
 * MRIResultCard — inline card shown in the chat thread after MRI analysis.
 * Shows: original MRI, Grad-CAM overlay, heatmap, prediction, probabilities,
 *        risk flag, and symptom-location correlation.
 */

import { ScanLine, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react'

function getRiskMeta(riskFlag) {
  const f = (riskFlag || '').toLowerCase()
  if (f.includes('high'))      return { cls: 'high', Icon: AlertTriangle }
  if (f.includes('no tumour')) return { cls: 'ok',   Icon: CheckCircle }
  return                              { cls: 'warn',  Icon: AlertCircle }
}

export default function MRIResultCard({ data }) {
  if (!data) return null

  const {
    pred_label, confidence, description, risk_flag,
    probabilities, gradcam_b64, heatmap_b64, correlation,
    original_b64,
  } = data

  const { cls, Icon } = getRiskMeta(risk_flag)

  const images = [
    { b64: original_b64,  label: 'Original MRI' },
    { b64: gradcam_b64,   label: 'Grad-CAM Overlay' },
    { b64: heatmap_b64,   label: 'Activation Heatmap' },
  ].filter(i => i.b64)

  const probEntries = Object.entries(probabilities || {})
    .sort((a, b) => b[1] - a[1])

  return (
    <div className="mri-card">
      {/* Header */}
      <div className="mri-card-header">
        <div className="mri-card-title">
          <ScanLine size={14} style={{ color: 'var(--cyan)' }} />
          MRI Analysis Result
        </div>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
        }}>
          XCEPTION · GRAD-CAM
        </span>
      </div>

      {/* Images */}
      {images.length > 0 && (
        <div className="mri-images" style={{
          gridTemplateColumns: `repeat(${images.length}, 1fr)`,
        }}>
          {images.map(({ b64, label }) => (
            <div key={label} className="mri-image-panel">
              <img
                src={`data:image/png;base64,${b64}`}
                alt={label}
              />
              <div className="mri-image-label">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Results grid */}
      <div className="mri-results">

        {/* Prediction */}
        <div className="prediction-block">
          <div className="pred-label-key">Prediction</div>
          <div className="pred-label-val">{pred_label}</div>
          <div className="confidence-bar-wrap">
            <div className="confidence-bar-track">
              <div
                className="confidence-bar-fill"
                style={{ width: `${confidence}%` }}
              />
            </div>
            <div className="confidence-pct">{confidence.toFixed(1)}% confidence</div>
          </div>
          <div style={{
            marginTop: '8px',
            fontSize: '11px',
            color: 'var(--text-muted)',
            lineHeight: 1.6,
          }}>
            {description}
          </div>
        </div>

        {/* Probability bars */}
        <div className="prediction-block">
          <div className="pred-label-key">All Probabilities</div>
          <div className="prob-list" style={{ marginTop: '6px' }}>
            {probEntries.map(([label, prob]) => (
              <div key={label} className="prob-row">
                <span className="prob-name">{label}</span>
                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Risk flag */}
        <div className={`risk-flag-block ${cls}`}>
          <Icon size={14} />
          {risk_flag}
        </div>

        {/* Correlation */}
        {correlation && (
          <div className="correlation-block">
            <div className="correlation-title">◈ Symptom-Location Correlation</div>
            <div className="correlation-text">{correlation}</div>
          </div>
        )}

      </div>
    </div>
  )
}
