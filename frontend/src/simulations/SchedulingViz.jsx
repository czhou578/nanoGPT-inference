import { useMemo } from 'react';
import { SCHEDULING_DATA, REQUEST_COLORS } from './simulationData';

/**
 * SchedulingViz — Demonstrates FCFS vs Priority scheduling with preemption.
 * Shows requests being admitted, preempted, and re-queued based on priority.
 */
export default function SchedulingViz({ step }) {
  const { maxBatchSize, maxKvTokens, requests: initialRequests } = SCHEDULING_DATA;

  const state = useMemo(() => {
    const reqStates = initialRequests.map(r => ({
      ...r,
      status: 'pending',
      generated: 0,
      kvTokens: 0,
      preemptCount: 0,
    }));
    const events = [];

    for (let s = 0; s <= step; s++) {
      // Admit new arrivals
      for (const r of reqStates) {
        if (r.arrivalStep === s && r.status === 'pending') {
          r.status = 'waiting';
          events.push({ step: s, text: `R${r.id} arrived`, highlight: `Priority: ${r.label}` });
        }
      }

      const active = reqStates.filter(r => r.status === 'active');
      const waiting = reqStates.filter(r => r.status === 'waiting');

      // Check if new high-priority request should preempt
      if (waiting.length > 0 && active.length >= maxBatchSize) {
        const bestWaiting = [...waiting].sort((a, b) => a.priority - b.priority)[0];
        const worstActive = [...active].sort((a, b) => b.priority - a.priority || a.arrivalStep - b.arrivalStep)[0];

        if (bestWaiting.priority < worstActive.priority) {
          // Preempt!
          worstActive.status = 'waiting';
          worstActive.generated = 0;
          worstActive.kvTokens = 0;
          worstActive.preemptCount++;
          events.push({ step: s, text: `R${worstActive.id} preempted!`, highlight: `by R${bestWaiting.id} (higher priority)` });
        }
      }

      // Admit from waiting queue (priority order)
      const currentActive = reqStates.filter(r => r.status === 'active');
      const sortedWaiting = reqStates
        .filter(r => r.status === 'waiting')
        .sort((a, b) => a.priority - b.priority || a.arrivalStep - b.arrivalStep);

      const slotsAvailable = maxBatchSize - currentActive.length;
      for (let i = 0; i < Math.min(slotsAvailable, sortedWaiting.length); i++) {
        const req = sortedWaiting[i];
        const totalKvUsed = reqStates
          .filter(r => r.status === 'active')
          .reduce((sum, r) => sum + r.promptLength + r.generated, 0);

        if (totalKvUsed + req.promptLength <= maxKvTokens) {
          req.status = 'active';
          req.kvTokens = req.promptLength;
          events.push({ step: s, text: `R${req.id} admitted`, highlight: `${req.label} priority` });
        }
      }

      // Decode step
      for (const r of reqStates) {
        if (r.status === 'active' && s > 0) {
          r.generated++;
          r.kvTokens = r.promptLength + r.generated;
          if (r.generated >= r.maxNewTokens) {
            r.status = 'done';
            events.push({ step: s, text: `R${r.id} completed`, highlight: `${r.maxNewTokens} tokens` });
          }
        }
      }
    }

    const waitingList = reqStates.filter(r => r.status === 'waiting');
    const activeList = reqStates.filter(r => r.status === 'active');
    const doneList = reqStates.filter(r => r.status === 'done');
    const totalKv = activeList.reduce((sum, r) => sum + r.promptLength + r.generated, 0);

    return { reqStates, waitingList, activeList, doneList, events: events.slice(-12), totalKv };
  }, [step]);

  const priorityColor = (label) => {
    if (label === 'High') return { badge: 'high', text: '#fca5a5' };
    if (label === 'Med') return { badge: 'med', text: '#fde047' };
    return { badge: 'low', text: '#9ca3af' };
  };

  return (
    <div className="sim-col" style={{ gap: '1.25rem' }}>
      {/* Stats */}
      <div className="sim-stats">
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#a5b4fc' }}>{step}</span>
          <span className="sim-stat-label">Step</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fde047' }}>{state.waitingList.length}</span>
          <span className="sim-stat-label">Waiting</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#86efac' }}>{state.activeList.length}/{maxBatchSize}</span>
          <span className="sim-stat-label">Active</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fca5a5' }}>
            {state.reqStates.reduce((sum, r) => sum + r.preemptCount, 0)}
          </span>
          <span className="sim-stat-label">Preemptions</span>
        </div>
      </div>

      {/* KV Memory */}
      <div className="viz-card" style={{ padding: '0.75rem 1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
          <span className="sim-section-label">KV Memory Usage</span>
          <span style={{ fontSize: '0.7rem', color: state.totalKv > maxKvTokens * 0.8 ? '#fca5a5' : '#86efac', fontWeight: 600 }}>
            {state.totalKv}/{maxKvTokens} tokens
          </span>
        </div>
        <div className="sim-meter">
          <div
            className={`sim-meter-fill ${state.totalKv > maxKvTokens * 0.8 ? 'red' : 'green'}`}
            style={{ width: `${Math.min(100, (state.totalKv / maxKvTokens) * 100)}%` }}
          />
        </div>
      </div>

      <div className="sim-row">
        {/* Waiting Queue */}
        <div className="viz-card">
          <h4>Waiting Queue (Priority Order)</h4>
          <div className="sim-col" style={{ gap: '0.35rem' }}>
            {state.waitingList.length === 0 && <span className="sim-empty">Queue empty</span>}
            {[...state.waitingList].sort((a, b) => a.priority - b.priority).map(r => {
              const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
              const pc = priorityColor(r.label);
              return (
                <div key={r.id} className="request-bar" style={{ background: color.bg, borderLeft: `3px solid ${color.border}` }}>
                  <span className="req-label" style={{ color: color.text }}>R{r.id}</span>
                  <span className={`sim-badge ${pc.badge}`}>{r.label}</span>
                  {r.preemptCount > 0 && (
                    <span style={{ fontSize: '0.6rem', color: '#fca5a5' }}>⟲ {r.preemptCount}×</span>
                  )}
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
              const req = state.activeList[i];
              if (!req) {
                return (
                  <div key={`empty-${i}`} className="batch-slot empty">
                    <span className="slot-index">#{i}</span>
                    <span style={{ color: '#374151', fontSize: '0.7rem' }}>Empty slot</span>
                  </div>
                );
              }
              const color = REQUEST_COLORS[req.id % REQUEST_COLORS.length];
              const pc = priorityColor(req.label);
              return (
                <div key={req.id} className="batch-slot occupied" style={{ background: color.bg, borderColor: color.border }}>
                  <span className="slot-index" style={{ color: color.text }}>#{i}</span>
                  <div className="slot-content">
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: color.text }}>R{req.id}</span>
                    <span className={`sim-badge ${pc.badge}`} style={{ fontSize: '0.55rem' }}>{req.label}</span>
                    <span style={{ fontSize: '0.65rem', color: '#9ca3af', marginLeft: 'auto' }}>
                      {req.generated}/{req.maxNewTokens}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Completed */}
      <div className="viz-card">
        <h4>Completed</h4>
        <div className="sim-token-row" style={{ gap: '0.5rem' }}>
          {state.doneList.length === 0 && <span className="sim-empty">None yet</span>}
          {state.doneList.map(r => {
            const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
            const pc = priorityColor(r.label);
            return (
              <div key={r.id} className="sim-badge done" style={{ background: color.bg, borderColor: color.border, color: color.text }}>
                R{r.id} ✓ <span style={{ color: pc.text, marginLeft: '0.2rem' }}>{r.label}</span>
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
