import React from 'react';

const MODULES = [
  { key: 'phone',         label: 'Phone Detection' },
  { key: 'fall',          label: 'Fall Detection' },
  { key: 'abandoned_bag', label: 'Abandoned Bag' },
  { key: 'zone_breach',   label: 'Unauthorized Entry' },
];

export default function Dashboard({ liveAlerts = [], riskScore = 0, heatmapPts = [], incidentCount = 0 }) {
  const activeTypes = new Set(liveAlerts.map(a => a.type));

  return (
    <div className="page">
      <div className="page-title">Command Dashboard</div>

      {/* Stat tiles across top */}
      <div className="stat-tiles" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 20 }}>
        <div className={`stat-tile ${liveAlerts.length > 0 ? 'danger' : 'success'}`}>
          <div className="stat-tile-value">{liveAlerts.length}</div>
          <div className="stat-tile-label">Active Alerts</div>
        </div>
        <div className="stat-tile warning">
          <div className="stat-tile-value">{incidentCount}</div>
          <div className="stat-tile-label">Total Incidents</div>
        </div>
        <div className="stat-tile accent">
          <div className="stat-tile-value">4/4</div>
          <div className="stat-tile-label">Modules Online</div>
        </div>
        <div className={`stat-tile ${riskScore < 30 ? 'success' : riskScore < 60 ? 'warning' : 'danger'}`}>
          <div className="stat-tile-value">{riskScore}</div>
          <div className="stat-tile-label">Risk Score</div>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* LEFT — Modules */}
        <div className="dashboard-left">
          <div className="card">
            <div className="card-title">Detection Modules</div>
            <div className="module-cards">
              {MODULES.map(m => {
                const active = activeTypes.has(m.key);
                return (
                  <div key={m.key} className={`module-card ${active ? 'active' : ''}`}>
                    <div className="module-dot" />
                    <div className="module-info">
                      <div className="module-name">{m.label}</div>
                      <div className="module-state">{active ? 'ALERT ACTIVE' : 'monitoring'}</div>
                    </div>
                    <span className="module-badge">{active ? '!' : 'OK'}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* MAIN — Risk gauge + active alerts */}
        <div className="dashboard-main">
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="card-title">Live Risk Score</div>
            <RiskGaugeInline score={riskScore} />
          </div>

          <div className="card">
            <div className="card-title">Active Alerts</div>
            {liveAlerts.length === 0 ? (
              <div className="no-alerts">No active threats detected</div>
            ) : (
              <div className="alert-pills">
                {liveAlerts.map((a, i) => {
                  const m = MODULES.find(x => x.key === a.type);
                  return (
                    <div key={i} className="alert-pill">
                      <div className="alert-pill-dot" />
                      <span className="alert-pill-label">{m?.label || a.type}</span>
                      <span className="alert-pill-conf">{(a.confidence * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT — System info + quick start */}
        <div className="dashboard-right">
          <div className="card">
            <div className="card-title">System Info</div>
            <div>
              {[
                { label: 'Model',    value: 'YOLOv8n ONNX' },
                { label: 'Backend',  value: 'FastAPI' },
                { label: 'Stream',   value: 'WebSocket' },
                { label: 'Classes',  value: '80 COCO' },
                { label: 'Port',     value: '8000' },
              ].map(s => (
                <div key={s.label} className="stat-row">
                  <span className="stat-label">{s.label}</span>
                  <span className="stat-value">{s.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-title">Quick Start</div>
            <div className="quick-list">
              {[
                ['1', 'Go to', 'Live Monitor'],
                ['2', 'Click', 'Start Camera'],
                ['3', 'Optionally draw a', 'Restricted Zone'],
                ['4', 'Watch real-time', 'Alerts'],
              ].map(([n, pre, bold]) => (
                <div key={n} className="quick-step">
                  <div className="step-num">{n}</div>
                  <div className="step-text">{pre} <strong>{bold}</strong></div>
                </div>
              ))}
            </div>
          </div>

          <div className="hackzen-card">
            <div className="hackzen-title">HackZen 2026</div>
            <div className="hackzen-text">
              Single YOLOv8n model powering 5 detection modules simultaneously — no additional models required.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* Inline light gauge — avoids import */
function RiskGaugeInline({ score = 0 }) {
  const R = 90, CX = 110, CY = 110;
  const ARC_START = 210, ARC_SPAN = 240;
  const c = Math.max(0, Math.min(100, score));
  const fillAngle = ARC_START + (c / 100) * ARC_SPAN;
  function polar(deg, r) { const rad = (deg - 90) * (Math.PI / 180); return { x: CX + r * Math.cos(rad), y: CY + r * Math.sin(rad) }; }
  function arc(s, e, r) { const a = polar(s, r); const b = polar(e, r); return `M${a.x} ${a.y} A${r} ${r} 0 ${e - s > 180 ? 1 : 0} 1 ${b.x} ${b.y}`; }
  const colour = c < 30 ? '#34d399' : c < 60 ? '#fb923c' : '#f87171';
  const level  = c < 30 ? 'LOW'     : c < 60 ? 'MEDIUM'  : 'HIGH';
  const levelClass = c < 30 ? 'risk-low' : c < 60 ? 'risk-medium' : 'risk-high';
  return (
    <div className="gauge-wrap">
      <svg width={220} height={180} viewBox="0 0 220 200" overflow="visible">
        <path d={arc(ARC_START, ARC_START + ARC_SPAN, R)} fill="none" stroke="#1e2336" strokeWidth={8} />
        {c > 0 && <path d={arc(ARC_START, fillAngle, R)} fill="none" stroke={colour} strokeWidth={8} strokeLinecap="round" style={{ filter: `drop-shadow(0 0 6px ${colour}80)` }} />}
        <text x={CX} y={CY + 14} textAnchor="middle" fontSize="44" fontWeight="800" fill={colour} fontFamily="Inter">{c}</text>
        <text x={CX} y={CY + 34} textAnchor="middle" fontSize="11" fill="#5a6280" fontFamily="Inter" fontWeight="600" letterSpacing="1">RISK SCORE</text>
        <text x={28}  y={166} fontSize="11" fill="#5a6280" textAnchor="middle">0</text>
        <text x={192} y={166} fontSize="11" fill="#5a6280" textAnchor="middle">100</text>
      </svg>
      <span className={`risk-level ${levelClass}`}>{level} RISK</span>
    </div>
  );
}
