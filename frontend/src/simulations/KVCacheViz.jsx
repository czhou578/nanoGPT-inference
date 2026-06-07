import { useMemo } from 'react';
import { KV_CACHE_DATA } from './simulationData';

/**
 * KVCacheViz — Side-by-side comparison of inference with and without KV cache.
 * Left: full recompute every step. Right: only new token computed, cache reused.
 */
export default function KVCacheViz({ step }) {
  const { promptTokens, generatedTokens, maxSteps } = KV_CACHE_DATA;
  const currentStep = Math.min(step, maxSteps);

  const allTokens = useMemo(
    () => [...promptTokens, ...generatedTokens.slice(0, currentStep)],
    [currentStep]
  );

  // No-cache: every prior token is recomputed each step
  const noCacheWork = useMemo(() => {
    if (currentStep === 0) return 0;
    let total = 0;
    for (let s = 1; s <= currentStep; s++) {
      total += promptTokens.length + s; // recompute everything
    }
    return total;
  }, [currentStep]);

  // With-cache: prompt is processed once (prefill), then 1 token per step
  const cacheWork = useMemo(() => {
    if (currentStep === 0) return 0;
    return promptTokens.length + currentStep; // prefill once + 1 per decode step
  }, [currentStep]);

  const savings = noCacheWork > 0 ? Math.round((1 - cacheWork / noCacheWork) * 100) : 0;

  return (
    <div className="sim-col" style={{ gap: '1.25rem' }}>
      {/* Stats */}
      <div className="sim-stats">
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#a5b4fc' }}>{currentStep}</span>
          <span className="sim-stat-label">Decode Step</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fca5a5' }}>{noCacheWork}</span>
          <span className="sim-stat-label">No-Cache Ops</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#86efac' }}>{cacheWork}</span>
          <span className="sim-stat-label">Cached Ops</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fde047' }}>{savings}%</span>
          <span className="sim-stat-label">Compute Saved</span>
        </div>
      </div>

      {/* Savings meter */}
      <div className="viz-card" style={{ padding: '0.75rem 1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
          <span className="sim-section-label">Compute Savings</span>
          <span style={{ fontSize: '0.7rem', color: '#86efac', fontWeight: 600 }}>{savings}%</span>
        </div>
        <div className="sim-meter">
          <div className="sim-meter-fill green" style={{ width: `${savings}%` }} />
        </div>
      </div>

      {/* Side-by-side comparison */}
      <div className="sim-comparison">
        {/* No Cache */}
        <div className="sim-comparison-col">
          <h5 className="no-cache">❌ Without KV Cache</h5>
          <div className="sim-section-label">Tokens recomputed this step:</div>
          <div className="sim-token-row">
            {currentStep > 0 && allTokens.map((tok, i) => (
              <span key={`nc-${i}`} className="sim-token recomputed active-compute">
                {tok}
              </span>
            ))}
            {currentStep === 0 && <span className="sim-empty">Press Play to begin</span>}
          </div>
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '0.25rem' }}>
            {currentStep > 0 && `${allTokens.length} tokens recomputed`}
          </div>
        </div>

        {/* With Cache */}
        <div className="sim-comparison-col">
          <h5 className="with-cache">✅ With KV Cache</h5>
          <div className="sim-section-label">Cached (reused):</div>
          <div className="sim-token-row">
            {currentStep > 0 && allTokens.slice(0, -1).map((tok, i) => (
              <span key={`c-${i}`} className="sim-token cached">
                {tok}
              </span>
            ))}
          </div>
          {currentStep > 0 && (
            <>
              <div className="sim-section-label" style={{ marginTop: '0.35rem' }}>Computed (new):</div>
              <div className="sim-token-row">
                <span className="sim-token generated active-compute">
                  {allTokens[allTokens.length - 1]}
                </span>
              </div>
            </>
          )}
          {currentStep === 0 && <span className="sim-empty">Press Play to begin</span>}
          <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '0.25rem' }}>
            {currentStep > 0 && `1 token computed, ${allTokens.length - 1} cached`}
          </div>
        </div>
      </div>

      {/* Generated sequence */}
      <div className="viz-card">
        <h4>Generated Sequence</h4>
        <div className="sim-token-row">
          {promptTokens.map((tok, i) => (
            <span key={`p-${i}`} className="sim-token prompt">{tok}</span>
          ))}
          {generatedTokens.slice(0, currentStep).map((tok, i) => (
            <span key={`g-${i}`} className="sim-token generated">{tok}</span>
          ))}
          {currentStep < maxSteps && (
            <span style={{ color: '#4b5563', fontSize: '0.75rem' }}>▏</span>
          )}
        </div>
        <div className="sim-legend" style={{ marginTop: '0.75rem' }}>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(99, 102, 241, 0.5)' }} />
            Prompt
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(34, 197, 94, 0.5)' }} />
            Generated
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(234, 179, 8, 0.4)' }} />
            Cached (reused)
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(239, 68, 68, 0.4)' }} />
            Recomputed
          </div>
        </div>
      </div>
    </div>
  );
}
