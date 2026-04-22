/**
 * ReportPanel — renders the full Gemini-written diagnostic report inline.
 * Includes a download-as-txt button.
 */

import { FileText, Download } from 'lucide-react'

export default function ReportPanel({ reportText, generatedAt }) {
  if (!reportText) return null

  function handleDownload() {
    const blob = new Blob([reportText], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `NeuroPath_Report_${generatedAt?.replace(/[: ]/g, '-') || 'report'}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="report-panel">
      {/* Header */}
      <div className="report-header">
        <div className="report-title">
          <FileText size={14} style={{ color: 'var(--cyan)' }} />
          Integrated Diagnostic Report
          {generatedAt && (
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              color: 'var(--text-muted)',
              fontWeight: 400,
              marginLeft: '4px',
            }}>
              {generatedAt}
            </span>
          )}
        </div>
        <button className="btn-download" onClick={handleDownload}>
          <Download size={12} />
          Download
        </button>
      </div>

      {/* Report body */}
      <div className="report-body">
        {reportText}
      </div>
    </div>
  )
}
