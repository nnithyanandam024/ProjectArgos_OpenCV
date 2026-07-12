import React from 'react';

const R = 90, CX = 110, CY = 110;
const ARC_START = 210;  // degrees
const ARC_SPAN  = 240;  // total sweep

function polarToXY(deg, r) {
  const rad = (deg - 90) * (Math.PI / 180);
  return { x: CX + r * Math.cos(rad), y: CY + r * Math.sin(rad) };
}

function describeArc(startDeg, endDeg, r) {
  const s = polarToXY(startDeg, r);
  const e = polarToXY(endDeg,   r);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

export default function RiskGauge({ score = 0 }) {
  const clamped   = Math.max(0, Math.min(100, score));
  const fillAngle = ARC_START + (clamped / 100) * ARC_SPAN;
  const arcStart  = ARC_START;
  const arcEnd    = ARC_START + ARC_SPAN;

  const colour = 'var(--text-primary)';
  const level  = clamped < 30 ? 'LOW'     : clamped < 60 ? 'MEDIUM'  : 'HIGH';
  const levelClass = clamped < 30 ? 'risk-low' : clamped < 60 ? 'risk-medium' : 'risk-high';

  const trackPath = describeArc(arcStart, arcEnd, R);
  const fillPath  = clamped > 0 ? describeArc(arcStart, fillAngle, R) : null;

  return (
    <div className="gauge-wrap">
      <svg className="gauge-svg" width={220} height={180} viewBox="0 0 220 200">
        {/* Track */}
        <path d={trackPath} className="gauge-track" />
        {/* Fill */}
        {fillPath && (
          <path
            d={fillPath}
            className="gauge-fill"
            stroke={colour}
            style={{ filter: 'drop-shadow(0 0 3px var(--text-primary))' }}
          />
        )}
        {/* Score text */}
        <text x={CX} y={CY + 10} className="gauge-label gauge-score" fill="var(--text-primary)">
          {clamped}
        </text>
        <text x={CX} y={CY + 30} className="gauge-label gauge-sub" fill="var(--text-muted)">
          RISK SCORE
        </text>
        {/* Min / Max labels */}
        <text x={28}  y={165} fontSize="11" fill="var(--text-dim)" textAnchor="middle">0</text>
        <text x={192} y={165} fontSize="11" fill="var(--text-dim)" textAnchor="middle">100</text>
      </svg>
      <span className={`risk-level ${levelClass}`}>{level} RISK</span>
    </div>
  );
}
