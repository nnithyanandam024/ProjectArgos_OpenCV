import React, { useRef, useEffect, useState, useCallback } from 'react';

const WS_URL = 'ws://localhost:8000/ws/guardian';

const ALERT_LABELS = {
  phone:         'Phone Detected',
  fall:          'Fall Detected',
  abandoned_bag: 'Abandoned Bag',
  zone_breach:   'Zone Breach',
};

export default function LiveFeed({ onAlerts, onRiskScore, onHeatmapPts, onIncidentCount }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const heatRef   = useRef(null);
  const zoneRef   = useRef(null);
  const wsRef     = useRef(null);
  const streamRef = useRef(null);
  const heatGrid  = useRef(null);

  const [running,  setRunning]  = useState(false);
  const [feedSrc,  setFeedSrc]  = useState(null);
  const [alerts,   setAlerts]   = useState([]);
  const [risk,     setRisk]     = useState(0);
  const [fps,      setFps]      = useState(0);
  const [drawZone, setDrawZone] = useState(false);
  const [zonePts,  setZonePts]  = useState([]);
  const [showHeat, setShowHeat] = useState(false);
  const [stats,    setStats]    = useState({ detections: 0, frames: 0 });
  const [error,    setError]    = useState(null);

  const fpsCounter = useRef({ frames: 0, last: Date.now() });
  const GRID_W = 64, GRID_H = 48;

  useEffect(() => {
    heatGrid.current = Array.from({ length: GRID_H }, () => new Float32Array(GRID_W));
  }, []);

  const accumulateHeat = (pts, w, h) => {
    if (!pts || !heatGrid.current) return;
    pts.forEach(([cx, cy]) => {
      const gx = Math.floor((cx / w) * GRID_W);
      const gy = Math.floor((cy / h) * GRID_H);
      if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H)
        heatGrid.current[gy][gx] += 1;
    });
  };

  const renderHeatmap = useCallback(() => {
    const canvas = heatRef.current;
    if (!canvas || !heatGrid.current) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let maxV = 1;
    for (let y = 0; y < GRID_H; y++)
      for (let x = 0; x < GRID_W; x++)
        maxV = Math.max(maxV, heatGrid.current[y][x]);
    const cw = canvas.width / GRID_W, ch = canvas.height / GRID_H;
    for (let y = 0; y < GRID_H; y++)
      for (let x = 0; x < GRID_W; x++) {
        const v = heatGrid.current[y][x] / maxV;
        if (v < 0.05) continue;
        const r = Math.floor(255 * Math.min(1, v * 2));
        const g = Math.floor(255 * Math.min(1, (1 - v) * 2));
        ctx.fillStyle = `rgba(${r},${g},0,${v * 0.7})`;
        ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
      }
  }, []);

  const handleZoneClick = (e) => {
    if (!drawZone) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const prev = e.currentTarget.previousSibling;
    const iw = prev.naturalWidth  || prev.clientWidth;
    const ih = prev.naturalHeight || prev.clientHeight;
    const px = ((e.clientX - rect.left) / rect.width)  * iw;
    const py = ((e.clientY - rect.top)  / rect.height) * ih;
    setZonePts(p => [...p, [Math.round(px), Math.round(py)]]);
  };

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => { setRunning(true); sendFrame(); };
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        setFeedSrc(`data:image/jpeg;base64,${data.frame}`);
        setAlerts(data.alerts || []);
        setRisk(data.risk_score || 0);
        onAlerts?.(data.alerts || []);
        onRiskScore?.(data.risk_score || 0);
        onHeatmapPts?.(data.heatmap_pts || []);
        accumulateHeat(data.heatmap_pts || [], 640, 480);
        if (showHeat) renderHeatmap();
        setStats(s => ({ ...s, detections: (data.detections || []).length, frames: s.frames + 1 }));
        const now = Date.now();
        fpsCounter.current.frames++;
        if (now - fpsCounter.current.last >= 1000) {
          setFps(fpsCounter.current.frames);
          fpsCounter.current.frames = 0;
          fpsCounter.current.last = now;
        }
        if ((data.alerts || []).length > 0) onIncidentCount?.(c => c + data.alerts.length);
        sendFrame();
      };
      ws.onerror = () => stopCamera();
      ws.onclose = () => setRunning(false);
    } catch (err) {
      setError('Camera access denied: ' + err.message);
    }
  };

  const sendFrame = () => {
    const video = videoRef.current, canvas = canvasRef.current, ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2 || video.videoWidth === 0) { setTimeout(sendFrame, 100); return; }
    canvas.width = 640; canvas.height = 480;
    canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
    ws.send(JSON.stringify({
      frame: canvas.toDataURL('image/jpeg', 0.8).split(',')[1],
      params: { conf: 0.35, zone: zonePts },
    }));
  };

  const stopCamera = () => {
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach(t => t.stop());
    setRunning(false); setFeedSrc(null); setAlerts([]); setRisk(0);
    onAlerts?.([]); onRiskScore?.(0);
  };

  useEffect(() => () => stopCamera(), []);
  useEffect(() => { if (showHeat) renderHeatmap(); }, [showHeat, renderHeatmap]);

  const riskClass    = risk < 30 ? 'low' : risk < 60 ? 'medium' : 'high';
  const uniqueAlerts = [...new Map(alerts.map(a => [a.type, a])).values()];

  return (
    <div className="page">
      <div className="page-heading">
        <div className="page-title">Live Monitor</div>
        <div className="page-sub">Real-time camera feed with AI threat detection</div>
      </div>

      <div className="live-layout">
        {/* ── Main feed column ── */}
        <div className="live-col-main">
          <div className="video-wrap">
            {feedSrc ? (
              <>
                <img src={feedSrc} className="video-img" alt="Live feed" />
                {showHeat && (
                  <canvas ref={heatRef} className="heat-canvas" width={640} height={480} />
                )}
                <canvas
                  ref={zoneRef}
                  className="zone-canvas"
                  style={{ cursor: drawZone ? 'crosshair' : 'default', pointerEvents: drawZone ? 'all' : 'none' }}
                  onClick={handleZoneClick}
                />
                <div className="alert-overlay">
                  {uniqueAlerts.map(a => (
                    <div key={a.type} className="alert-chip">
                      <div className="alert-chip-dot" />
                      {ALERT_LABELS[a.type] || a.type}
                    </div>
                  ))}
                </div>
                <div className="fps-tag">{fps} FPS</div>
              </>
            ) : (
              <div className="video-placeholder">
                <div className="ph-title">Camera Feed</div>
                <div className="ph-sub">Click "Start Camera" to begin detection</div>
              </div>
            )}
          </div>

          {error && <div className="error-strip">{error}</div>}

          <div className="controls">
            {!running
              ? <button className="btn primary" onClick={startCamera}>Start Camera</button>
              : <button className="btn danger"  onClick={stopCamera}>Stop Camera</button>
            }
            <button
              className={`btn ${drawZone ? 'active-draw' : ''}`}
              onClick={() => setDrawZone(d => !d)}
            >
              {drawZone ? 'Drawing Zone...' : 'Draw Zone'}
            </button>
            {zonePts.length > 0 && (
              <button className="btn" onClick={() => setZonePts([])}>Clear Zone</button>
            )}
            <button
              className={`btn ${showHeat ? 'active-draw' : ''}`}
              onClick={() => setShowHeat(h => !h)}
            >
              Heatmap
            </button>
          </div>

          {zonePts.length > 0 && (
            <div>
              <span className="zone-info">
                {zonePts.length} zone point{zonePts.length !== 1 ? 's' : ''}
                {zonePts.length >= 3 ? ' — Active' : ' — Need 3+ points'}
              </span>
            </div>
          )}
        </div>

        {/* ── Side panel ── */}
        <div className="live-col-side">
          {/* Risk */}
          <div className="card" style={{ textAlign: 'center' }}>
            <div className="card-title" style={{ textAlign: 'left' }}>Threat Level</div>
            <div className={`risk-big ${riskClass}`}>{risk}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>/ 100</div>
            <div style={{ marginTop: 10 }}>
              <span className={`risk-badge ${riskClass}`}>{riskClass.toUpperCase()} RISK</span>
            </div>
          </div>

          {/* Active threats */}
          <div className="card">
            <div className="card-title">Active Threats</div>
            {uniqueAlerts.length === 0 ? (
              <div className="no-alerts">No threats detected</div>
            ) : (
              <div className="alert-list">
                {uniqueAlerts.map(a => (
                  <div key={a.type} className="alert-row">
                    <div className="alert-dot" />
                    <span className="alert-lbl">{ALERT_LABELS[a.type] || a.type}</span>
                    <span className="alert-conf">{(a.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Session stats */}
          <div className="card">
            <div className="card-title">Session Stats</div>
            {[
              { lbl: 'Active Alerts',    val: alerts.length },
              { lbl: 'Objects in Frame', val: stats.detections },
              { lbl: 'Frames Sent',      val: stats.frames },
              { lbl: 'Live FPS',         val: fps },
              { lbl: 'Zone Points',      val: zonePts.length },
            ].map(s => (
              <div key={s.lbl} className="info-row">
                <span className="info-lbl">{s.lbl}</span>
                <span className="info-val">{s.val}</span>
              </div>
            ))}
          </div>

          {/* How to use */}
          <div className="card">
            <div className="card-title">How to Use</div>
            <div className="step-list">
              {[
                ['1', 'Click', 'Start Camera'],
                ['2', 'Hold a phone to trigger', 'Phone Alert'],
                ['3', 'Lie down to trigger', 'Fall Alert'],
                ['4', 'Leave a bag 5s for', 'Bag Alert'],
                ['5', 'Draw a zone to guard', 'Restricted Area'],
              ].map(([n, a, b]) => (
                <div key={n} className="step-row">
                  <div className="step-num">{n}</div>
                  <div className="step-text">{a} <b>{b}</b></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <video ref={videoRef} autoPlay muted playsInline
        style={{ position: 'absolute', width: 1, height: 1, opacity: 0.01, pointerEvents: 'none' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}
