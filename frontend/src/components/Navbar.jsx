import React from 'react';

export default function Navbar({ tab, setTab, riskScore, incidentCount }) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'live',      label: 'Live Monitor' },
    { id: 'incidents', label: 'Incident Log' },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="logo-icon">A</div>
        <div>
          <span className="brand-name">Project Argos</span>
          <span className="brand-sub">AI Campus Guardian</span>
        </div>
      </div>

      <div className="navbar-tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {t.id === 'incidents' && incidentCount > 0 && (
              <span style={{
                marginLeft: 6, background: '#f87171', color: '#fff',
                borderRadius: '99px', padding: '1px 7px', fontSize: 11, fontWeight: 700,
              }}>
                {incidentCount}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="navbar-status">
        <span className="dot" />
        SYSTEM ONLINE
      </div>
    </nav>
  );
}
