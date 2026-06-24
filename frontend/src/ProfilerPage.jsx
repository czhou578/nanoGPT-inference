/**
 * ProfilerPage - Main page component for the inference profiler.
 *
 * Provides a drag-and-drop zone or file picker to load trace.json files,
 * a dropdown of bundled example traces, and renders the timeline chart
 * with the category legend.
 */

import { useState, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import useTraceData, { getCategoryColor } from './profiler/useTraceData';
import CategoryLegend from './profiler/CategoryLegend';
import TimelineChart from './profiler/TimelineChart';
import './ProfilerPage.css';

/** Bundled example traces served from /public/traces/ */
const EXAMPLE_TRACES = [
  {
    id: 'interleaving',
    label: 'Interleaved Prefill + Decode',
    file: '/traces/trace_interleaving.json',
    description: '4 requests with chunked prefill interleaved with decode steps',
  },
];

export default function ProfilerPage() {
  const { traceData, loading, error, fileName, loadFromFile, loadFromUrl, clear } = useTraceData();
  const [dragging, setDragging] = useState(false);
  const [selectedSpan, setSelectedSpan] = useState(null);
  const fileInputRef = useRef(null);

  // ── Drag & drop handlers ──────────────────────────────────
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);

    const file = e.dataTransfer?.files?.[0];
    if (file && file.name.endsWith('.json')) {
      loadFromFile(file);
    }
  }, [loadFromFile]);

  const handleFileInput = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      loadFromFile(file);
    }
  }, [loadFromFile]);

  const handleExampleSelect = useCallback((trace) => {
    loadFromUrl(trace.file, trace.label);
  }, [loadFromUrl]);

  const handleSpanClick = useCallback((span) => {
    setSelectedSpan(prev => prev === span ? null : span);
  }, []);

  const formatDuration = (us) => {
    if (us < 1000) return `${us}µs`;
    if (us < 1000000) return `${(us / 1000).toFixed(1)}ms`;
    return `${(us / 1000000).toFixed(2)}s`;
  };

  return (
    <div className="profiler-page">
      {/* Back link */}
      <Link to="/" className="profiler-back-link" id="profiler-back-link">
        ← Back to Knowledge Base
      </Link>

      {/* Header */}
      <header className="profiler-header">
        <h1>
          <span className="profiler-title-icon">⚡</span>
          Inference Profiler
        </h1>
        <p>
          Flame-chart timeline of inference execution — visualize where time is
          spent across scheduling, attention, memory, and sampling.
        </p>
      </header>

      {/* File loading section */}
      {!traceData && !loading && (
        <div className="profiler-load-section">
          {/* Drag & drop zone */}
          <div
            className={`profiler-drop-zone ${dragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            id="profiler-drop-zone"
          >
            <div className="drop-icon">📂</div>
            <div className="drop-text">
              <strong>Drop a trace.json here</strong>
              <span>or click to browse</span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileInput}
              style={{ display: 'none' }}
              id="profiler-file-input"
            />
          </div>

          {/* Divider */}
          <div className="profiler-divider">
            <span>or load an example</span>
          </div>

          {/* Example traces */}
          <div className="profiler-examples">
            {EXAMPLE_TRACES.map((trace) => (
              <button
                key={trace.id}
                className="example-btn"
                onClick={() => handleExampleSelect(trace)}
                id={`example-${trace.id}`}
              >
                <span className="example-label">{trace.label}</span>
                <span className="example-desc">{trace.description}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="profiler-loading">
          <div className="loading-spinner" />
          <span>Loading trace…</span>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="profiler-error" id="profiler-error">
          <strong>Error:</strong> {error}
          <button className="error-dismiss" onClick={clear}>Dismiss</button>
        </div>
      )}

      {/* Trace loaded: show timeline */}
      {traceData && !loading && (
        <div className="profiler-content">
          {/* Toolbar */}
          <div className="profiler-toolbar">
            <div className="toolbar-info">
              <span className="toolbar-filename">📄 {fileName}</span>
              <span className="toolbar-stats">
                {traceData.totalSpans} spans · {traceData.totalRequests} requests · {formatDuration(traceData.totalDuration)}
              </span>
            </div>
            <button
              className="toolbar-close"
              onClick={clear}
              title="Close trace"
              id="profiler-close-trace"
            >
              ✕ Close
            </button>
          </div>

          {/* Category Legend */}
          <CategoryLegend byCategory={traceData.byCategory} />

          {/* Summary bar */}
          <div className="profiler-summary-bar">
            <div className="summary-breakdown">
              {Object.entries(traceData.byCategory)
                .sort(([, a], [, b]) => b.totalUs - a.totalUs)
                .map(([cat, data]) => {
                  const color = getCategoryColor(cat);
                  return (
                    <div
                      key={cat}
                      className="breakdown-segment"
                      style={{
                        flex: data.pct,
                        backgroundColor: color.bg,
                      }}
                      title={`${color.label}: ${data.pct}%`}
                    />
                  );
                })}
            </div>
          </div>

          {/* Timeline Chart */}
          <div className="profiler-chart-wrapper">
            <TimelineChart
              traceData={traceData}
              onSpanClick={handleSpanClick}
            />
          </div>

          {/* Selected span detail panel */}
          {selectedSpan && (
            <div className="profiler-detail-panel" id="profiler-detail-panel">
              <div className="detail-header">
                <h3>{selectedSpan.name}</h3>
                <button
                  className="detail-close"
                  onClick={() => setSelectedSpan(null)}
                >
                  ✕
                </button>
              </div>
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-key">Category</span>
                  <span className="detail-val">{selectedSpan.category}</span>
                </div>
                {!selectedSpan.isInstant && (
                  <div className="detail-item">
                    <span className="detail-key">Duration</span>
                    <span className="detail-val">{formatDuration(selectedSpan.durationUs)}</span>
                  </div>
                )}
                <div className="detail-item">
                  <span className="detail-key">Lane</span>
                  <span className="detail-val">{selectedSpan.swimlane}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-key">Start</span>
                  <span className="detail-val">{formatDuration(selectedSpan.startUs)}</span>
                </div>
                {Object.entries(selectedSpan.metadata).map(([k, v]) => (
                  <div className="detail-item" key={k}>
                    <span className="detail-key">{k}</span>
                    <span className="detail-val">
                      {typeof v === 'boolean' ? (v ? '✓' : '✗') : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
