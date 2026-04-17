import { useEffect, useState, useRef } from 'react';
import './PipelineVisualizer.css';

const BLOCK_SIZE = 16;
const DISPLAY_BLOCKS = 64; // Show the first 64 physical blocks
const STEP_DELAY_MS = 120;

// ─── Sub-components ──────────────────────────────────────────────────────────

function SchedulerBadge({ status }) {
  const map = {
    WAITING: { label: 'WAITING', cls: 'badge-waiting' },
    RUNNING: { label: 'RUNNING', cls: 'badge-running' },
    DONE:    { label: 'DONE',    cls: 'badge-done'    },
  };
  const info = map[status] ?? map['WAITING'];
  return <span className={`scheduler-badge ${info.cls}`}>{info.label}</span>;
}

function TokenSequenceBar({ numPromptTokens, numVisionTokens, totalTokens }) {
  const tokens = [];
  for (let i = 0; i < totalTokens; i++) {
    let cls = 'token-chip generated';
    if (i < numVisionTokens)                       cls = 'token-chip vision';
    else if (i < numVisionTokens + (numPromptTokens - numVisionTokens)) cls = 'token-chip text';
    tokens.push(<div key={i} className={cls} title={`Token ${i}`} />);
  }
  return (
    <div className="token-bar-scroll">
      <div className="token-bar">{tokens}</div>
    </div>
  );
}

function BlockMemoryGrid({ blockTable, freeBlocks }) {
  // blockTable: array of physical block indices currently allocated
  const allocated = new Set(blockTable);
  return (
    <div className="block-grid">
      {Array.from({ length: DISPLAY_BLOCKS }, (_, i) => (
        <div
          key={i}
          className={`block-cell ${allocated.has(i) ? 'block-allocated' : 'block-free'}`}
          title={`Block ${i}: ${allocated.has(i) ? 'ALLOCATED' : 'FREE'}`}
        />
      ))}
    </div>
  );
}

function VisionPanel({ imagePreview, numVisionTokens }) {
  const COLS = 4;
  const ROWS = Math.ceil(numVisionTokens / COLS);
  return (
    <div className="vision-panel">
      <div className="vision-image-wrapper">
        {imagePreview && <img src={imagePreview} alt="Input" className="vision-img" />}
        <div
          className="patch-overlay"
          style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, gridTemplateRows: `repeat(${ROWS}, 1fr)` }}
        >
          {Array.from({ length: numVisionTokens }, (_, i) => (
            <div key={i} className="patch-cell" />
          ))}
        </div>
      </div>
      <p className="vision-caption">{numVisionTokens} vision patch tokens encoded</p>
    </div>
  );
}

function DecodeStepLog({ steps }) {
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [steps]);

  return (
    <div className="step-log">
      {steps.map((s, idx) => (
        <div key={idx} className="step-row fade-in-row">
          <span className="step-num">Step {s.step + 1}</span>
          <span className="step-token">token_id <strong>{s.new_token_id}</strong></span>
          <span className="step-tokens">tokens: <strong>{s.total_tokens}</strong></span>
          <span className="step-blocks">
            blocks: <strong>{s.block_table.length}</strong>
            <span className="step-free"> ({s.free_blocks} free)</span>
          </span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

// ─── Main Visualizer ─────────────────────────────────────────────────────────

export default function PipelineVisualizer({ data, imagePreview }) {
  const [displayedSteps, setDisplayedSteps] = useState([]);
  const [currentStep, setCurrentStep]       = useState(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!data?.trace?.length) return;
    setDisplayedSteps([]);
    setCurrentStep(null);

    let idx = 0;
    const tick = () => {
      if (idx >= data.trace.length) return;
      const step = data.trace[idx];
      setDisplayedSteps(prev => [...prev, step]);
      setCurrentStep(step);
      idx++;
      timerRef.current = setTimeout(tick, STEP_DELAY_MS);
    };
    timerRef.current = setTimeout(tick, 200);
    return () => clearTimeout(timerRef.current);
  }, [data]);

  if (!data) return null;

  const numVisionTokens = data.num_vision_tokens ?? 0;
  const numPromptTokens = currentStep?.num_prompt_tokens ?? 0;
  const totalTokens     = currentStep?.total_tokens ?? numPromptTokens;
  const blockTable      = currentStep?.block_table ?? [];
  const freeBlocks      = currentStep?.free_blocks ?? 0;
  const status          = displayedSteps.length >= (data.trace?.length ?? 0) ? 'DONE' : 'RUNNING';

  return (
    <div className="visualizer">
      {/* ── Header ── */}
      <div className="viz-header">
        <h3>Pipeline Visualization</h3>
        <SchedulerBadge status={displayedSteps.length === 0 ? 'WAITING' : status} />
      </div>

      {/* ── Row 1: Vision + Token Sequence ── */}
      <div className="viz-row two-col">
        <section className="viz-section">
          <h4>① Vision Encoder</h4>
          <VisionPanel imagePreview={imagePreview} numVisionTokens={numVisionTokens} />
        </section>

        <section className="viz-section">
          <h4>② Token Sequence</h4>
          <div className="legend">
            <span className="legend-chip vision" />Vision
            <span className="legend-chip text" />Text
            <span className="legend-chip generated" />Generated
          </div>
          <TokenSequenceBar
            numVisionTokens={numVisionTokens}
            numPromptTokens={numPromptTokens}
            totalTokens={totalTokens}
          />
          <p className="viz-caption">{totalTokens} tokens in sequence</p>
        </section>
      </div>

      {/* ── Row 2: Block Memory Grid ── */}
      <section className="viz-section">
        <h4>③ Block Memory (first {DISPLAY_BLOCKS} of {DISPLAY_BLOCKS}+ physical blocks)</h4>
        <div className="block-legend">
          <span className="block-cell block-allocated" /> Allocated &nbsp;
          <span className="block-cell block-free" /> Free
        </div>
        <BlockMemoryGrid blockTable={blockTable} freeBlocks={freeBlocks} />
        <p className="viz-caption">
          {blockTable.length} block{blockTable.length !== 1 ? 's' : ''} allocated
          &nbsp;·&nbsp; block size = {BLOCK_SIZE} tokens
        </p>
      </section>

      {/* ── Row 3: Decode steps log ── */}
      <section className="viz-section">
        <h4>④ Decode Steps</h4>
        <DecodeStepLog steps={displayedSteps} />
      </section>
    </div>
  );
}
