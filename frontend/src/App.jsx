import React, { useState } from 'react';
import './index.css';
import Navbar       from './components/Navbar';
import Dashboard    from './components/Dashboard';
import LiveFeed     from './components/LiveFeed';
import IncidentLog  from './components/IncidentLog';

export default function App() {
  const [tab, setTab]         = useState('dashboard');
  const [liveAlerts, setLiveAlerts]   = useState([]);
  const [riskScore, setRiskScore]     = useState(0);
  const [heatmapPts, setHeatmapPts]   = useState([]);
  const [incidentCount, setIncidentCount] = useState(0);

  return (
    <>
      <Navbar
        tab={tab}
        setTab={setTab}
        riskScore={riskScore}
        incidentCount={incidentCount}
      />
      <div style={{ display: tab === 'dashboard' ? 'block' : 'none' }}>
        <Dashboard
          liveAlerts={liveAlerts}
          riskScore={riskScore}
          heatmapPts={heatmapPts}
          incidentCount={incidentCount}
        />
      </div>
      <div style={{ display: tab === 'live' ? 'block' : 'none' }}>
        <LiveFeed
          onAlerts={setLiveAlerts}
          onRiskScore={setRiskScore}
          onHeatmapPts={setHeatmapPts}
          onIncidentCount={setIncidentCount}
        />
      </div>
      <div style={{ display: tab === 'incidents' ? 'block' : 'none' }}>
        <IncidentLog incidentCount={incidentCount} />
      </div>
    </>
  );
}
