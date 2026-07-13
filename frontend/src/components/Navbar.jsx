import React from 'react';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'live',      label: 'Live Monitor' },
  { id: 'incidents', label: 'Incident Log' },
];

export default function Navbar({ tab, setTab, riskScore = 0, incidentCount = 0 }) {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div>
          <div className="brand-name">Project Argos</div>
          <div className="brand-sub">Campus Guardian</div>
        </div>
      </div>

      <div className="navbar-tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {t.id === 'incidents' && incidentCount > 0 && (
              <span className="incident-badge">{incidentCount}</span>
            )}
          </button>
        ))}
      </div>

      <div className="navbar-right">
        <div className="status-chip">
          <span className="status-dot" />
          System Online
        </div>
      </div>
    </nav>
  );
}
