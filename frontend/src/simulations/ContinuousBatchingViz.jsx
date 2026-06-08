import { useMemo } from 'react';
import { CONTINUOUS_BATCHING_DATA, REQUEST_COLORS } from './simulationData';

/**
 * ContinuousBatchingViz — Shows requests arriving over time and being dynamically
 * batched. When a request finishes, its slot is immediately filled by a new one.
 */
export default function ContinuousBatchingViz({ step }) {
  const { maxBatchSize, requests } = CONTINUOUS_BATCHING_DATA;

  // Simulate the state at the current step
  const state = useMemo(() => {
    const waiting = [];
    const active = [];
    const completed = [];
    const events = [];

    // Clone requests with mutable state
    const reqStates = requests.map(r => ({
      ...r,
      status: 'pending',
      generated: 0,
    }));

    for (let s = 0; s <= step; s++) {
      // Admit new arrivals to waiting queue
      for (const r of reqStates) {
        if (r.arrivalStep === s && r.status === 'pending') {
          r.status = 'waiting';
          events.push({ step: s, text: `R${r.id} arrived`, highlight: `"${r.prompt}"` });
        }
      }

      // Fill empty batch slots from waiting queue
      const currentActive = reqStates.filter(r => r.status === 'active');
      const currentWaiting = reqStates.filter(r => r.status === 'waiting');
      const slotsAvailable = maxBatchSize - currentActive.length;

      for (let i = 0; i < Math.min(slotsAvailable, currentWaiting.length); i++) {
        currentWaiting[i].status = 'active';
        events.push({ step: s, text: `R${currentWaiting[i].id} admitted to batch` });
      }

      // Generate one token for each active request
      for (const r of reqStates) {
        if (r.status === 'active') {
          r.generated++;
          if (r.generated >= r.maxNewTokens) {
            r.status = 'done';
            events.push({ step: s, text: `R${r.id} completed`, highlight: `${r.maxNewTokens} tokens` });
          }
        }
      }
    }

    // Categorize
    for (const r of reqStates) {
      if (r.status === 'done') completed.push(r);
      else if (r.status === 'active') active.push(r);
      else if (r.status === 'waiting') waiting.push(r);
    }

    return { waiting, active, completed, events: events.slice(-12) };
  }, [step]);

  const batchUtilization = state.active.length / maxBatchSize;

  return (
    <div className="sim-col" style={{ gap: '1.25rem' }}>
      {/* Stats */}
      <div className="sim-stats">
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#a5b4fc' }}>{step}</span>
          <span className="sim-stat-label">Step</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fde047' }}>{state.waiting.length}</span>
          <span className="sim-stat-label">Waiting</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#86efac' }}>{state.active.length}</span>
          <span className="sim-stat-label">Active</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#9ca3af' }}>{state.completed.length}</span>
          <span className="sim-stat-label">Completed</span>
        </div>
      </div>

      {/* Batch utilization */}
      <div className="viz-card" style={{ padding: '0.75rem 1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
          <span className="sim-section-label">Batch Utilization</span>
          <span style={{ fontSize: '0.7rem', color: '#a5b4fc', fontWeight: 600 }}>
            {state.active.length}/{maxBatchSize} slots
          </span>
        </div>
        <div className="sim-meter">
          <div
            className={`sim-meter-fill ${batchUtilization >= 0.75 ? 'green' : batchUtilization >= 0.5 ? 'amber' : 'red'}`}
            style={{ width: `${batchUtilization * 100}%` }}
          />
        </div>
      </div>

      <div className="sim-row">
        {/* Waiting Queue */}
        <div className="viz-card">
          <h4>Waiting Queue</h4>
          <div className="sim-col" style={{ gap: '0.35rem' }}>
            {state.waiting.length === 0 && <span className="sim-empty">Queue empty</span>}
            {state.waiting.map(r => {
              const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
              return (
                <div key={r.id} className="request-bar" style={{ background: color.bg, borderLeft: `3px solid ${color.border}` }}>
                  <span className="req-label" style={{ color: color.text }}>R{r.id}</span>
                  <span className="req-prompt">{r.prompt}</span>
                  <span className="sim-badge waiting">Waiting</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Active Batch */}
        <div className="viz-card">
          <h4>Active Batch</h4>
          <div className="sim-col" style={{ gap: '0.35rem' }}>
            {Array.from({ length: maxBatchSize }, (_, i) => {
              const req = state.active[i];
              if (!req) {
                return (
                  <div key={`empty-${i}`} className="batch-slot empty">
                    <span className="slot-index">#{i}</span>
                    <span style={{ color: '#374151', fontSize: '0.7rem' }}>Empty slot</span>
                  </div>
                );
              }
              const color = REQUEST_COLORS[req.id % REQUEST_COLORS.length];
              return (
                <div key={req.id} className="batch-slot occupied" style={{ background: color.bg, borderColor: color.border }}>
                  <span className="slot-index" style={{ color: color.text }}>#{i}</span>
                  <div className="slot-content">
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: color.text }}>R{req.id}</span>
                    <span style={{ fontSize: '0.65rem', color: '#9ca3af', marginLeft: '0.25rem' }}>
                      {req.generated}/{req.maxNewTokens}
                    </span>
                    {/* Progress dots */}
                    {Array.from({ length: req.maxNewTokens }, (_, t) => (
                      <div
                        key={t}
                        className="slot-token"
                        style={{
                          background: t < req.generated ? color.border : 'rgba(255,255,255,0.06)',
                          border: `1px solid ${t < req.generated ? color.border : 'rgba(255,255,255,0.1)'}`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Completed */}
      <div className="viz-card">
        <h4>Completed Requests</h4>
        <div className="sim-token-row" style={{ gap: '0.5rem' }}>
          {state.completed.length === 0 && <span className="sim-empty">None yet</span>}
          {state.completed.map(r => {
            const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
            return (
              <div key={r.id} className="sim-badge done" style={{ background: color.bg, borderColor: color.border, color: color.text }}>
                R{r.id} ✓ {r.maxNewTokens} tokens
              </div>
            );
          })}
        </div>
      </div>

      {/* Event Log */}
      <div className="viz-card">
        <h4>Event Log</h4>
        <div className="event-log">
          {state.events.length === 0 && <span className="sim-empty">Press Play to begin</span>}
          {state.events.map((e, i) => (
            <div key={i} className="event-log-entry">
              <span className="log-step">Step {e.step}</span>
              <span className="log-event">
                {e.text}
                {e.highlight && <> — <span className="log-highlight">{e.highlight}</span></>}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
