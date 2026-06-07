import { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { MODES } from './simulations/simulationData';
import KVCacheViz from './simulations/KVCacheViz';
import ContinuousBatchingViz from './simulations/ContinuousBatchingViz';
import ChunkedPrefillViz from './simulations/ChunkedPrefillViz';
import PagedAttentionViz from './simulations/PagedAttentionViz';
import SchedulingViz from './simulations/SchedulingViz';
import SpeculativeDecodingViz from './simulations/SpeculativeDecodingViz';
import './SimulationPage.css';
import './simulations/simulations.css';

const SPEED_OPTIONS = [
  { label: '0.5×', ms: 1200 },
  { label: '1×',   ms: 600 },
  { label: '2×',   ms: 300 },
  { label: '4×',   ms: 150 },
];

const MAX_STEPS = {
  'kv-cache': 8,
  'continuous-batching': 20,
  'chunked-prefill': 18,
  'paged-attention': 12,
  'scheduling': 15,
  'speculative-decoding': 4,
};

const VIZ_COMPONENTS = {
  'kv-cache': KVCacheViz,
  'continuous-batching': ContinuousBatchingViz,
  'chunked-prefill': ChunkedPrefillViz,
  'paged-attention': PagedAttentionViz,
  'scheduling': SchedulingViz,
  'speculative-decoding': SpeculativeDecodingViz,
};

export default function SimulationPage() {
  const [activeMode, setActiveMode] = useState(MODES[0].id);
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1); // default 1×
  const timerRef = useRef(null);

  const maxStep = MAX_STEPS[activeMode] || 20;
  const currentMode = MODES.find(m => m.id === activeMode);
  const VizComponent = VIZ_COMPONENTS[activeMode];

  // Auto-advance timer
  useEffect(() => {
    if (isPlaying && step < maxStep) {
      timerRef.current = setTimeout(() => {
        setStep(s => s + 1);
      }, SPEED_OPTIONS[speedIdx].ms);
    } else if (step >= maxStep) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timerRef.current);
  }, [isPlaying, step, speedIdx, maxStep]);

  const handleModeChange = useCallback((modeId) => {
    setActiveMode(modeId);
    setStep(0);
    setIsPlaying(false);
  }, []);

  const handlePlayPause = useCallback(() => {
    if (step >= maxStep) {
      setStep(0);
      setIsPlaying(true);
    } else {
      setIsPlaying(p => !p);
    }
  }, [step, maxStep]);

  const handleStepForward = useCallback(() => {
    setIsPlaying(false);
    setStep(s => Math.min(s + 1, maxStep));
  }, [maxStep]);

  const handleStepBack = useCallback(() => {
    setIsPlaying(false);
    setStep(s => Math.max(s - 1, 0));
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setStep(0);
  }, []);

  return (
    <div className="sim-page">
      {/* Back link */}
      <Link to="/" className="sim-back-link">← Back to Knowledge Base</Link>

      {/* Header */}
      <header className="sim-header">
        <h1>Inference Strategies</h1>
        <p>Interactive visualizations of LLM inference optimization techniques</p>
      </header>

      {/* Mode Selector */}
      <nav className="mode-selector" role="tablist" aria-label="Simulation modes">
        {MODES.map(mode => (
          <button
            key={mode.id}
            className={`mode-btn ${activeMode === mode.id ? 'active' : ''}`}
            onClick={() => handleModeChange(mode.id)}
            role="tab"
            aria-selected={activeMode === mode.id}
            id={`tab-${mode.id}`}
          >
            <span className="mode-icon">{mode.icon}</span>
            {mode.label}
          </button>
        ))}
      </nav>

      {/* Description */}
      {currentMode && (
        <div className="sim-description" key={currentMode.id}>
          {currentMode.description}
        </div>
      )}

      {/* Transport Controls */}
      <div className="transport-controls">
        <button className="transport-btn" onClick={handleReset} title="Reset" id="btn-reset">⟲</button>
        <button className="transport-btn" onClick={handleStepBack} title="Step back" id="btn-step-back">⏮</button>
        <button
          className={`transport-btn play ${isPlaying ? 'active' : ''}`}
          onClick={handlePlayPause}
          title={isPlaying ? 'Pause' : 'Play'}
          id="btn-play-pause"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button className="transport-btn" onClick={handleStepForward} title="Step forward" id="btn-step-forward">⏭</button>
        <span className="transport-step-label">
          Step {step}/{maxStep}
        </span>

        {/* Speed selector */}
        <div className="speed-selector">
          {SPEED_OPTIONS.map((opt, i) => (
            <button
              key={opt.label}
              className={`speed-btn ${speedIdx === i ? 'active' : ''}`}
              onClick={() => setSpeedIdx(i)}
              id={`speed-${opt.label}`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Visualization */}
      <div className="sim-viz-area" role="tabpanel" aria-labelledby={`tab-${activeMode}`}>
        {VizComponent && <VizComponent step={step} />}
      </div>
    </div>
  );
}
