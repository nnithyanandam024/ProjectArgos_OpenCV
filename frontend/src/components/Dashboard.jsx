import React from 'react';

const MODULES = [
  { key: 'phone',         label: 'Phone Detection'    },
  { key: 'fall',          label: 'Fall Detection'     },
  { key: 'abandoned_bag', label: 'Abandoned Bag'      },
  { key: 'zone_breach',   label: 'Unauthorized Entry' },
];

const ALERT_LABELS = {
  phone:         'Phone Detected',
  fall:          'Fall Detected',
  abandoned_bag: 'Abandoned Bag',
  zone_breach:   'Zone Breach',
};

export default function Dashboard({ liveAlerts = [], riskScore = 0, heatmapPts = [], incidentCount = 0 }) {
  const activeTypes = new Set(liveAlerts.map(a => a.type));
  const riskClass   = riskScore < 30 ? 'low' : riskScore < 60 ? 'medium' : 'high';

  return (
    <div className="page">
      <div className="page-heading">
        <div className="page-title">Command Dashboard</div>
        <div className="page-sub">AI-powered security monitoring overview</div>
      </div>

      {/* ── Stat tiles ── */}
      <div className="stat-tiles">
        <div className="stat-tile">
          <div className={`stat-tile-val ${liveAlerts.length > 0 ? 'c-danger' : 'c-success'}`}>
            {liveAlerts.length}
          </div>
          <div className="stat-tile-lbl">Active Alerts</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile-val c-warning">{incidentCount}</div>
          <div className="stat-tile-lbl">Total Incidents</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile-val c-accent">{MODULES.length}/{MODULES.length}</div>
          <div className="stat-tile-lbl">Modules Active</div>
        </div>
        <div className="stat-tile">
          <div className={`stat-tile-val ${riskScore < 30 ? 'c-success' : riskScore < 60 ? 'c-warning' : 'c-danger'}`}>
            {riskScore}
          </div>
          <div className="stat-tile-lbl">Risk Score</div>
        </div>
      </div>

      {/* ── Three-column grid ── */}
      <div className="dash-grid">
        {/* LEFT */}
        <div className="dash-col">
          <div className="card">
            <div className="card-title">Detection Modules</div>
            <div className="module-list">
              {MODULES.map(m => {
                const active = activeTypes.has(m.key);
                return (
                  <div key={m.key} className={`module-row ${active ? 'is-alert' : ''}`}>
                    <div className="module-indicator" />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="module-name">{m.label}</div>
                      <div className="module-state">{active ? 'ALERT ACTIVE' : 'monitoring'}</div>
                    </div>
                    <span className="module-tag">{active ? '!' : 'OK'}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* CENTRE */}
        <div className="dash-col">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="card-title" style={{ textAlign: 'left' }}>Live Risk Score</div>
            <div style={{ padding: '20px 0 16px' }}>
              <div style={{
                fontSize: 56, fontWeight: 800, lineHeight: 1,
                letterSpacing: '-2px', fontVariantNumeric: 'tabular-nums',
                color: riskScore < 30 ? 'var(--success)' : riskScore < 60 ? 'var(--warning)' : 'var(--danger)',
              }}>
                {riskScore}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>/ 100</div>
              <div style={{ marginTop: 14 }}>
                <span className={`risk-badge ${riskClass}`}>
                  {riskClass.toUpperCase()} RISK
                </span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Active Alerts</div>
            {liveAlerts.length === 0 ? (
              <div className="no-alerts">No active threats detected</div>
            ) : (
              <div className="alert-list">
                {liveAlerts.map((a, i) => (
                  <div key={i} className="alert-row">
                    <div className="alert-dot" />
                    <span className="alert-lbl">{ALERT_LABELS[a.type] || a.type}</span>
                    <span className="alert-conf">{(a.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT */}
        <div className="dash-col">
          <div className="card">
            <div className="card-title">System Info</div>
            {[
              { lbl: 'Model',    val: 'YOLOv8n ONNX' },
              { lbl: 'Pose',     val: 'YOLOv8n-pose' },
              { lbl: 'Backend',  val: 'FastAPI / ORT' },
              { lbl: 'Stream',   val: 'WebSocket' },
              { lbl: 'Latency',  val: '~55 ms/frame' },
              { lbl: 'Port',     val: '8000' },
            ].map(s => (
              <div key={s.lbl} className="info-row">
                <span className="info-lbl">{s.lbl}</span>
                <span className="info-val">{s.val}</span>
              </div>
            ))}
          </div>

          <div className="card">
            <div className="card-title">Quick Start</div>
            <div className="step-list">
              {[
                ['1', 'Go to', 'Live Monitor'],
                ['2', 'Click', 'Start Camera'],
                ['3', 'Draw a', 'Restricted Zone'],
                ['4', 'Monitor', 'Real-time Alerts'],
              ].map(([n, a, b]) => (
                <div key={n} className="step-row">
                  <div className="step-num">{n}</div>
                  <div className="step-text">{a} <b>{b}</b></div>
                </div>
              ))}
            </div>
          </div>

          <div className="info-card">
            <div className="info-card-title">HackZen 2026</div>
            <div className="info-card-body">
              Three ONNX models running simultaneously — detection, pose estimation, and fire/smoke — at ~55 ms/frame on CPU.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

