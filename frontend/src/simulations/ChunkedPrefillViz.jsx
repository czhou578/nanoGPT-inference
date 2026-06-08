import { useMemo } from 'react';
import { CHUNKED_PREFILL_DATA, REQUEST_COLORS } from './simulationData';

/**
 * ChunkedPrefillViz — Shows how long prompts are split into chunks and
 * interleaved with decode steps. Visualizes token budget allocation.
 */
export default function ChunkedPrefillViz({ step }) {
  const { tokenBudget, requests } = CHUNKED_PREFILL_DATA;

  const state = useMemo(() => {
    const reqStates = requests.map(r => ({
      ...r,
      status: 'pending',
      prefillCursor: 0,
      generated: 0,
    }));
    const events = [];
    const budgetHistory = [];

    for (let s = 0; s <= step; s++) {
      let prefillUsed = 0;
      let decodeUsed = 0;

      // Admit new arrivals
      for (const r of reqStates) {
        if (r.arrivalStep === s && r.status === 'pending') {
          r.status = 'prefilling';
          events.push({ step: s, text: `R${r.id} arrived`, highlight: `${r.promptLength} prompt tokens` });
        }
      }

      const activeDecoding = reqStates.filter(r => r.status === 'active');
      decodeUsed = activeDecoding.length;
      const budgetForPrefill = tokenBudget - decodeUsed;

      // Prefill one request with remaining budget
      const prefilling = reqStates.find(r => r.status === 'prefilling');
      if (prefilling && budgetForPrefill > 0) {
        const tokensLeft = prefilling.promptLength - prefilling.prefillCursor;
        const chunkSize = Math.min(budgetForPrefill, tokensLeft);
        prefilling.prefillCursor += chunkSize;
        prefillUsed = chunkSize;

        if (prefilling.prefillCursor >= prefilling.promptLength) {
          prefilling.status = 'active';
          prefilling.generated = 1; // first decode token from prefill
          events.push({ step: s, text: `R${prefilling.id} prefill complete`, highlight: 'Promoted to active' });
        } else {
          events.push({ step: s, text: `R${prefilling.id} prefill chunk`, highlight: `${chunkSize} tokens (${prefilling.prefillCursor}/${prefilling.promptLength})` });
        }
      }

      // Decode step for active requests
      for (const r of activeDecoding) {
        r.generated++;
        if (r.generated >= r.maxNewTokens) {
          r.status = 'done';
          events.push({ step: s, text: `R${r.id} completed`, highlight: `${r.maxNewTokens} tokens generated` });
        }
      }

      budgetHistory.push({ step: s, prefill: prefillUsed, decode: decodeUsed, free: tokenBudget - prefillUsed - decodeUsed });
    }

    const waiting = reqStates.filter(r => r.status === 'pending');
    const prefilling = reqStates.filter(r => r.status === 'prefilling');
    const active = reqStates.filter(r => r.status === 'active');
    const done = reqStates.filter(r => r.status === 'done');
    const latestBudget = budgetHistory[budgetHistory.length - 1] || { prefill: 0, decode: 0, free: tokenBudget };

    return { reqStates, waiting, prefilling, active, done, events: events.slice(-10), latestBudget };
  }, [step]);

  return (
    <div className="sim-col" style={{ gap: '1.25rem' }}>
      {/* Stats */}
      <div className="sim-stats">
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#a5b4fc' }}>{step}</span>
          <span className="sim-stat-label">Step</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#d8b4fe' }}>{state.prefilling.length}</span>
          <span className="sim-stat-label">Prefilling</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#86efac' }}>{state.active.length}</span>
          <span className="sim-stat-label">Decoding</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#9ca3af' }}>{state.done.length}</span>
          <span className="sim-stat-label">Done</span>
        </div>
      </div>

      {/* Token Budget Bar */}
      <div className="viz-card">
        <h4>Token Budget (capacity: {tokenBudget})</h4>
        <div className="budget-bar">
          {Array.from({ length: tokenBudget }, (_, i) => {
            let cls = 'budget-slot empty';
            if (i < state.latestBudget.prefill) cls = 'budget-slot prefill';
            else if (i < state.latestBudget.prefill + state.latestBudget.decode) cls = 'budget-slot decode';
            return <div key={i} className={cls} />;
          })}
        </div>
        <div className="sim-legend" style={{ marginTop: '0.5rem' }}>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(168, 85, 247, 0.6)' }} />
            Prefill ({state.latestBudget.prefill})
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(34, 197, 94, 0.6)' }} />
            Decode ({state.latestBudget.decode})
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(255, 255, 255, 0.06)' }} />
            Free ({state.latestBudget.free})
          </div>
        </div>
      </div>

      {/* Request Status */}
      <div className="viz-card">
        <h4>Request Pipeline</h4>
        <div className="sim-col" style={{ gap: '0.5rem' }}>
          {state.reqStates.map(r => {
            const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
            return (
              <div key={r.id} className="request-bar" style={{ background: color.bg, borderLeft: `3px solid ${color.border}` }}>
                <span className="req-label" style={{ color: color.text }}>R{r.id}</span>

                {/* Prefill chunk bar */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  <div className="chunk-bar">
                    {Array.from({ length: r.promptLength }, (_, i) => {
                      let cls = 'chunk-segment pending';
                      if (i < r.prefillCursor) cls = 'chunk-segment processed';
                      else if (r.status === 'prefilling' && i === r.prefillCursor) cls = 'chunk-segment current';
                      return (
                        <div
                          key={i}
                          className={cls}
                          style={{
                            flex: 1,
                            background: i < r.prefillCursor ? color.border : 'rgba(255,255,255,0.06)',
                          }}
                        />
                      );
                    })}
                  </div>
                  {r.status === 'active' || r.status === 'done' ? (
                    <div style={{ display: 'flex', gap: '2px' }}>
                      {Array.from({ length: r.maxNewTokens }, (_, i) => (
                        <div
                          key={i}
                          style={{
                            flex: 1, height: '6px', borderRadius: '2px',
                            background: i < r.generated ? '#22c55e' : 'rgba(255,255,255,0.06)',
                            transition: 'background 0.3s',
                          }}
                        />
                      ))}
                    </div>
                  ) : null}
                </div>

                <span className={`sim-badge ${r.status === 'pending' ? 'waiting' : r.status}`}>
                  {r.status === 'pending' ? 'Pending' : r.status}
                </span>
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
