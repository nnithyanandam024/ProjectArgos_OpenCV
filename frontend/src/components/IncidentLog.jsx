import React, { useState, useEffect, useRef } from 'react';

const API = 'http://localhost:8000';

const ALERT_META = {
  phone:         { label: 'Phone Detected',  short: 'Phone'  },
  fall:          { label: 'Fall Detected',   short: 'Fall'   },
  abandoned_bag: { label: 'Abandoned Bag',   short: 'Bag'    },
  zone_breach:   { label: 'Zone Breach',     short: 'Zone'   },
};

const FILTERS = ['all', 'phone', 'fall', 'abandoned_bag', 'zone_breach'];

export default function IncidentLog({ incidentCount }) {
  const [incidents, setIncidents] = useState([]);
  const [filter,    setFilter]    = useState('all');
  const [loading,   setLoading]   = useState(false);
  const prevCount = useRef(incidentCount);

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

  useEffect(() => { fetchIncidents(); }, []); // eslint-disable-line

  useEffect(() => {
    if (incidentCount !== prevCount.current) {
      prevCount.current = incidentCount;
      fetchIncidents();
    }
  });

  const filtered = filter === 'all' ? incidents : incidents.filter(i => i.type === filter);

  return (
    <div className="page">
      <div className="page-heading">
        <div className="page-title">Incident Log</div>
        <div className="page-sub">All recorded detection events this session</div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Toolbar */}
        <div className="toolbar">
          <div className="chips">
            {FILTERS.map(f => (
              <button
                key={f}
                className={`chip ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f === 'all' ? 'All' : ALERT_META[f]?.short}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={fetchIncidents}>Refresh</button>
            {incidents.length > 0 && (
              <button className="btn danger" onClick={clearIncidents}>Clear All</button>
            )}
          </div>
        </div>

        {filtered.length > 0 && (
          <div className="count-bar">
            {filtered.length} incident{filtered.length !== 1 ? 's' : ''}
            {filter !== 'all' ? ` · ${ALERT_META[filter]?.label}` : ''}
          </div>
        )}

        {/* Timeline */}
        <div className="card" style={{ padding: 0 }}>
          {loading && (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              Loading...
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="empty">
              <div className="empty-title">No Incidents Recorded</div>
              <div className="empty-sub">Start the Live Monitor to begin detection</div>
            </div>
          )}

          <div className="timeline">
            {filtered.map((inc, i) => {
              const m = ALERT_META[inc.type] || { label: inc.type };
              return (
                <div key={i} className="inc-row">
                  {inc.snapshot
                    ? <img className="inc-thumb" src={`data:image/jpeg;base64,${inc.snapshot}`} alt="snapshot" />
                    : <div className="inc-thumb-ph">No img</div>
                  }
                  <div className="inc-body">
                    <span className="inc-type-tag">{m.label}</span>
                    <div className="inc-time">{inc.timestamp}</div>
                    <div className="inc-conf">Confidence: {(inc.confidence * 100).toFixed(1)}%</div>
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
