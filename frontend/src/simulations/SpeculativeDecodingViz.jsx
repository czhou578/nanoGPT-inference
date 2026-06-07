import { useMemo } from 'react';
import { SPECULATIVE_DECODING_DATA } from './simulationData';

/**
 * SpeculativeDecodingViz — Draft model proposes K tokens, target model verifies.
 * Shows accept/reject per token, resampling, and bonus tokens.
 */
export default function SpeculativeDecodingViz({ step }) {
  const { K, rounds } = SPECULATIVE_DECODING_DATA;

  // Each "step" here reveals one round of speculation
  const currentRound = Math.min(step, rounds.length);

  const stats = useMemo(() => {
    let totalAccepted = 0;
    let totalRejected = 0;
    let totalBonusTokens = 0;
    let totalDraftCalls = 0;
    let totalTargetCalls = 0;

    for (let r = 0; r < currentRound; r++) {
      const round = rounds[r];
      const accepted = round.accepted.filter(Boolean).length;
      const rejected = round.accepted.length - accepted;
      totalAccepted += accepted;
      totalRejected += rejected;
      if (round.bonus) totalBonusTokens++;
      totalDraftCalls += K; // draft generates K tokens
      totalTargetCalls += 1; // target verifies in 1 forward pass
    }

    // Tokens produced per target forward pass
    const tokensProduced = totalAccepted + totalBonusTokens + (totalRejected > 0 ? totalRejected : 0); // resampled tokens count too
    const avgTokensPerCall = totalTargetCalls > 0 ? ((totalAccepted + totalBonusTokens + Math.min(totalRejected, currentRound)) / totalTargetCalls).toFixed(1) : '0';

    return { totalAccepted, totalRejected, totalBonusTokens, totalTargetCalls, avgTokensPerCall };
  }, [currentRound]);

  // Collect all accepted tokens in sequence
  const outputTokens = useMemo(() => {
    const tokens = [];
    for (let r = 0; r < currentRound; r++) {
      const round = rounds[r];
      for (let i = 0; i < round.accepted.length; i++) {
        if (round.accepted[i]) {
          tokens.push({ text: round.draftTokens[i], type: 'accepted' });
        } else {
          tokens.push({ text: round.resample, type: 'resampled' });
          break;
        }
      }
      if (round.bonus && round.accepted.every(Boolean)) {
        tokens.push({ text: round.bonus, type: 'bonus' });
      }
    }
    return tokens;
  }, [currentRound]);

  return (
    <div className="sim-col" style={{ gap: '1.25rem' }}>
      {/* Stats */}
      <div className="sim-stats">
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#a5b4fc' }}>{currentRound}/{rounds.length}</span>
          <span className="sim-stat-label">Round</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#86efac' }}>{stats.totalAccepted}</span>
          <span className="sim-stat-label">Accepted</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fca5a5' }}>{stats.totalRejected}</span>
          <span className="sim-stat-label">Rejected</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#fde047' }}>{stats.totalBonusTokens}</span>
          <span className="sim-stat-label">Bonus</span>
        </div>
        <div className="sim-stat">
          <span className="sim-stat-value" style={{ color: '#d8b4fe' }}>{stats.avgTokensPerCall}</span>
          <span className="sim-stat-label">Tok/Call</span>
        </div>
      </div>

      {/* Efficiency bar */}
      <div className="viz-card" style={{ padding: '0.75rem 1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
          <span className="sim-section-label">Acceptance Rate</span>
          <span style={{ fontSize: '0.7rem', color: '#86efac', fontWeight: 600 }}>
            {stats.totalAccepted + stats.totalRejected > 0
              ? Math.round((stats.totalAccepted / (stats.totalAccepted + stats.totalRejected)) * 100)
              : 0}%
          </span>
        </div>
        <div className="sim-meter">
          <div
            className="sim-meter-fill green"
            style={{
              width: `${stats.totalAccepted + stats.totalRejected > 0
                ? (stats.totalAccepted / (stats.totalAccepted + stats.totalRejected)) * 100
                : 0}%`
            }}
          />
        </div>
      </div>

      {/* Speculation Rounds */}
      <div className="viz-card">
        <h4>Speculation Rounds</h4>
        <div className="sim-col" style={{ gap: '0.75rem' }}>
          {currentRound === 0 && <span className="sim-empty">Press Play to begin</span>}
          {rounds.slice(0, currentRound).map((round, rIdx) => {
            const allAccepted = round.accepted.every(Boolean);
            return (
              <div key={rIdx} className="spec-round">
                <div className="spec-round-header">
                  <span>Round {rIdx + 1}</span>
                  <span style={{ color: allAccepted ? '#86efac' : '#fde047' }}>
                    {round.accepted.filter(Boolean).length}/{round.accepted.length} accepted
                    {allAccepted && ' + bonus!'}
                  </span>
                </div>

                {/* Draft → Verify flow */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '0.65rem', color: '#6b7280', width: '3.5rem' }}>Draft:</span>
                  <div className="spec-tokens">
                    {round.draftTokens.map((tok, i) => (
                      <span key={i} className="sim-token draft">{tok}</span>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '0.65rem', color: '#6b7280', width: '3.5rem' }}>Verify:</span>
                  <div className="spec-tokens">
                    {round.draftTokens.slice(0, round.accepted.length).map((tok, i) => (
                      <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.15rem' }}>
                        <span className={`sim-token ${round.accepted[i] ? 'accepted' : 'rejected'}`}>
                          {tok}
                        </span>
                        <span className="spec-verdict">
                          {round.accepted[i] ? '✓' : '✗'}
                        </span>
                      </span>
                    ))}
                    {!allAccepted && round.resample && (
                      <>
                        <span className="spec-arrow">→</span>
                        <span className="sim-token resampled">{round.resample}</span>
                      </>
                    )}
                    {allAccepted && round.bonus && (
                      <>
                        <span className="spec-arrow">+</span>
                        <span className="sim-token generated" style={{ animation: 'tokenAppear 0.4s ease both' }}>
                          {round.bonus}
                        </span>
                        <span style={{ fontSize: '0.6rem', color: '#86efac' }}>bonus!</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Output Sequence */}
      <div className="viz-card">
        <h4>Output Sequence</h4>
        <div className="sim-token-row">
          {outputTokens.length === 0 && <span className="sim-empty">No tokens yet</span>}
          {outputTokens.map((tok, i) => (
            <span key={i} className={`sim-token ${tok.type}`}>
              {tok.text}
            </span>
          ))}
        </div>
        <div className="sim-legend" style={{ marginTop: '0.75rem' }}>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(34, 197, 94, 0.5)' }} />
            Accepted
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(239, 68, 68, 0.4)' }} />
            Rejected
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(249, 115, 22, 0.5)' }} />
            Resampled
          </div>
          <div className="sim-legend-item">
            <div className="sim-legend-dot" style={{ background: 'rgba(34, 197, 94, 0.5)', boxShadow: '0 0 6px rgba(34,197,94,0.5)' }} />
            Bonus
          </div>
        </div>
      </div>

      {/* How it works */}
      <div className="viz-card" style={{ padding: '0.75rem 1rem' }}>
        <div style={{ fontSize: '0.7rem', color: '#6b7280', lineHeight: 1.6 }}>
          <strong style={{ color: '#9ca3af' }}>How it works:</strong> A fast draft model proposes {K} candidate tokens.
          The target model verifies all {K} in <strong style={{ color: '#d8b4fe' }}>one forward pass</strong>.
          If a token is rejected, we resample from the target distribution.
          If <em>all</em> are accepted, we get a free bonus token.
        </div>
      </div>
    </div>
  );
}
