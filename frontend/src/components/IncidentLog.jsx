import React, { useState, useEffect, useRef } from 'react';

const API = 'http://localhost:8000';

const ALERT_META = {
  phone:         { label: 'Phone Detected',   short: 'Phone'  },
  fall:          { label: 'Fall Detected',    short: 'Fall'   },
  abandoned_bag: { label: 'Abandoned Bag',    short: 'Bag'    },
  zone_breach:   { label: 'Zone Breach',      short: 'Zone'   },
};

const FILTERS = ['all', 'phone', 'fall', 'abandoned_bag', 'zone_breach'];

export default function IncidentLog({ incidentCount }) {
  const [incidents, setIncidents] = useState([]);
  const [filter,    setFilter]    = useState('all');
  const [loading,   setLoading]   = useState(false);
  const prevCount   = useRef(incidentCount);

  const fetchIncidents = async () => {
    setLoading(true);
    try {
      const data = await (await fetch(`${API}/incidents`)).json();
      setIncidents(data);
    } catch { /* backend offline */ }
    finally { setLoading(false); }
  };

  const clearIncidents = async () => {
    await fetch(`${API}/incidents`, { method: 'DELETE' });
    setIncidents([]);
  };

  // Initial load
  useEffect(() => {
    fetchIncidents();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-fetch whenever incidentCount changes (triggered by new alerts in LiveFeed)
  useEffect(() => {
    if (incidentCount !== prevCount.current) {
      prevCount.current = incidentCount;
      fetchIncidents();
    }
  }); // no deps array → runs every render, but only fetches when count actually changed

  const filtered = filter === 'all' ? incidents : incidents.filter(i => i.type === filter);

  return (
    <div className="page">
      <div className="page-title">Incident Log</div>

      <div className="incidents-page">
        {/* Toolbar */}
        <div className="incidents-toolbar">
          <div className="filter-chips">
            {FILTERS.map(f => (
              <button key={f} className={`chip ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
                {f === 'all' ? 'All' : ALERT_META[f]?.short}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={fetchIncidents}>&#8635; Refresh</button>
            {incidents.length > 0 && <button className="btn danger" onClick={clearIncidents}>Clear All</button>}
          </div>
        </div>

        {/* Count bar */}
        {filtered.length > 0 && (
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {filtered.length} incident{filtered.length !== 1 ? 's' : ''}{filter !== 'all' ? ` · ${ALERT_META[filter]?.label}` : ''}
          </div>
        )}

        {/* Timeline */}
        <div className="card">
          {loading && <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0', fontSize: 13 }}>Loading incidents...</div>}

          {!loading && filtered.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">&#128737;</div>
              <div className="empty-title">No Incidents Recorded</div>
              <div className="empty-sub">Start the Live Monitor to begin detection</div>
            </div>
          )}

          <div className="incident-timeline">
            {filtered.map((inc, i) => {
              const m = ALERT_META[inc.type] || { label: inc.type };
              return (
                <div key={i} className="incident-item">
                  {inc.snapshot
                    ? <img className="incident-thumb" src={`data:image/jpeg;base64,${inc.snapshot}`} alt="snapshot" />
                    : <div className="incident-thumb-placeholder">&#128249;</div>
                  }
                  <div className="incident-body">
                    <span className="incident-type">{m.label}</span>
                    <div className="incident-time">{inc.timestamp}</div>
                    <div className="incident-conf">Confidence: {(inc.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
