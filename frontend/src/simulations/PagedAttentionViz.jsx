import { useMemo } from 'react';
import { PAGED_ATTENTION_DATA, REQUEST_COLORS } from './simulationData';

/**
 * PagedAttentionViz — Visualizes virtual-to-physical block mapping for KV cache memory.
 * Shows a block table per request and a shared physical memory grid.
 */
export default function PagedAttentionViz({ step }) {
  const { blockSize, totalPhysicalBlocks, requests } = PAGED_ATTENTION_DATA;

  const state = useMemo(() => {
    const reqStates = requests.map(r => ({
      ...r,
      status: 'pending',
      blockTable: [],
      filledSlots: 0,
      generated: 0,
    }));
    const freeBlocks = Array.from({ length: totalPhysicalBlocks }, (_, i) => i);
    const events = [];

    const allocateBlock = () => {
      if (freeBlocks.length === 0) return -1;
      return freeBlocks.pop();
    };

    for (let s = 0; s <= step; s++) {
      // Admit pending requests (all arrive at step 0 for simplicity here)
      for (const r of reqStates) {
        if (r.status === 'pending' && s === 0) {
          r.status = 'prefilling';

          // Allocate blocks for prompt
          const blocksNeeded = Math.ceil(r.promptLength / blockSize);
          for (let b = 0; b < blocksNeeded; b++) {
            const phys = allocateBlock();
            if (phys >= 0) r.blockTable.push(phys);
          }
          r.filledSlots = r.promptLength;
          r.status = 'active';
          r.generated = 0;
          events.push({ step: s, text: `R${r.id} prefilled`, highlight: `${blocksNeeded} blocks allocated` });
        }
      }

      // Generate one token for each active request
      if (s > 0) {
        for (const r of reqStates) {
          if (r.status === 'active') {
            r.generated++;
            r.filledSlots++;

            // Need a new block?
            if (r.filledSlots > r.blockTable.length * blockSize) {
              const phys = allocateBlock();
              if (phys >= 0) {
                r.blockTable.push(phys);
                events.push({ step: s, text: `R${r.id} new block`, highlight: `Phys block ${phys}` });
              }
            }

            if (r.generated >= r.maxNewTokens) {
              r.status = 'done';
              // Free blocks
              for (const b of r.blockTable) freeBlocks.push(b);
              events.push({ step: s, text: `R${r.id} completed`, highlight: `${r.blockTable.length} blocks freed` });
            }
          }
        }
      }
    }

    // Build physical block ownership map
    const blockOwnership = {};
    for (const r of reqStates) {
      if (r.status !== 'done') {
        for (const b of r.blockTable) {
          blockOwnership[b] = r.id;
        }
      }
    }

    return {
      reqStates,
      freeBlocks: new Set(freeBlocks),
      blockOwnership,
      events: events.slice(-10),
      numFree: freeBlocks.length,
    };
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
          <span className="sim-stat-value" style={{ color: '#86efac' }}>{totalPhysicalBlocks - state.numFree}</span>
          <span className="sim-stat-label">Allocated</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#6b7280' }}>{state.numFree}</span>
          <span className="sim-stat-label">Free Blocks</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fde047' }}>{blockSize}</span>
          <span className="sim-stat-label">Block Size</span>
        </div>
      </div>

      <div className="sim-row">
        {/* Block Tables */}
        <div className="viz-card">
          <h4>Block Tables (Virtual → Physical)</h4>
          <div className="sim-col" style={{ gap: '0.75rem' }}>
            {state.reqStates.filter(r => r.status !== 'done' && r.status !== 'pending').map(r => {
              const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
              return (
                <div key={r.id}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: color.text }}>R{r.id}</span>
                    <span className={`sim-badge ${r.status}`}>{r.status}</span>
                    <span style={{ fontSize: '0.65rem', color: '#6b7280', marginLeft: 'auto' }}>
                      {r.filledSlots} tokens in {r.blockTable.length} blocks
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                    {r.blockTable.map((phys, vi) => {
                      const slotsInBlock = Math.min(blockSize, r.filledSlots - vi * blockSize);
                      const isFull = slotsInBlock >= blockSize;
                      return (
                        <div key={vi} style={{
                          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.15rem',
                        }}>
                          <div style={{
                            padding: '0.25rem 0.5rem', borderRadius: '5px', fontSize: '0.65rem',
                            background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                            color: '#9ca3af',
                          }}>
                            V{vi}
                          </div>
                          <span style={{ fontSize: '0.6rem', color: '#4b5563' }}>↓</span>
                          <div style={{
                            padding: '0.25rem 0.5rem', borderRadius: '5px', fontSize: '0.65rem',
                            fontWeight: 600, background: color.bg, border: `1px solid ${color.border}`,
                            color: color.text, opacity: isFull ? 1 : 0.7,
                          }}>
                            P{phys}
                            <span style={{ fontSize: '0.5rem', marginLeft: '0.25rem', opacity: 0.7 }}>
                              {Math.max(0, slotsInBlock)}/{blockSize}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
            {state.reqStates.filter(r => r.status !== 'done' && r.status !== 'pending').length === 0 && (
              <span className="sim-empty">All requests completed</span>
            )}
          </div>
        </div>

        {/* Physical Memory Grid */}
        <div className="viz-card">
          <h4>Physical Memory Pool ({totalPhysicalBlocks} blocks)</h4>
          <div className="sim-block-grid">
            {Array.from({ length: totalPhysicalBlocks }, (_, i) => {
              const owner = state.blockOwnership[i];
              const isAllocated = owner !== undefined;
              const color = isAllocated ? REQUEST_COLORS[owner % REQUEST_COLORS.length] : null;
              return (
                <div
                  key={i}
                  className={`sim-block-cell ${isAllocated ? 'allocated' : 'free'}`}
                  style={isAllocated ? { background: color.bg, borderColor: color.border } : {}}
                  title={isAllocated ? `Block ${i}: R${owner}` : `Block ${i}: Free`}
                >
                  <span className="block-label" style={isAllocated ? { color: color.text } : { color: '#374151' }}>
                    {i}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="sim-legend" style={{ marginTop: '0.75rem' }}>
            {state.reqStates.filter(r => r.status !== 'done' && r.status !== 'pending').map(r => {
              const color = REQUEST_COLORS[r.id % REQUEST_COLORS.length];
              return (
                <div key={r.id} className="sim-legend-item">
                  <div className="sim-legend-dot" style={{ background: color.border }} />
                  R{r.id}
                </div>
              );
            })}
            <div className="sim-legend-item">
              <div className="sim-legend-dot" style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)' }} />
              Free
            </div>
          </div>
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
