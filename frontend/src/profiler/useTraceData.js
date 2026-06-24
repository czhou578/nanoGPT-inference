/**
 * useTraceData - Hook to load and process Chrome Trace Event JSON.
 *
 * Accepts either a File object (drag-and-drop / file picker) or a URL string
 * (bundled example traces). Returns processed span data grouped by request
 * (swimlane), by category, and summary statistics for the timeline chart.
 */

import { useState, useCallback } from 'react';

/**
 * Category color palette - carefully chosen for visual distinction on dark backgrounds.
 * Each category gets a hue-shifted color from a cohesive palette.
 */
export const CATEGORY_COLORS = {
  attention:        { bg: '#6366f1', fg: '#c7d2fe', label: 'Attention'        },
  compute:          { bg: '#6366f1', fg: '#c7d2fe', label: 'Compute'          },
  memory:           { bg: '#f59e0b', fg: '#fef3c7', label: 'Memory'           },
  scheduling:       { bg: '#10b981', fg: '#d1fae5', label: 'Scheduling'       },
  sampling:         { bg: '#ec4899', fg: '#fce7f3', label: 'Sampling'         },
  prefill:          { bg: '#3b82f6', fg: '#dbeafe', label: 'Prefill'          },
  decode:           { bg: '#8b5cf6', fg: '#ede9fe', label: 'Decode'           },
  speculative:      { bg: '#f97316', fg: '#ffedd5', label: 'Speculative'      },
  cache_management: { bg: '#14b8a6', fg: '#ccfbf1', label: 'Cache Mgmt'      },
  lifecycle:        { bg: '#64748b', fg: '#e2e8f0', label: 'Lifecycle'        },
};

const DEFAULT_COLOR = { bg: '#6b7280', fg: '#e5e7eb', label: 'Other' };

export function getCategoryColor(category) {
  return CATEGORY_COLORS[category] || DEFAULT_COLOR;
}

/**
 * Process raw Chrome Trace Event JSON into structured timeline data.
 */
function processTraceData(raw) {
  const events = raw.traceEvents || [];
  const metadata = raw.metadata || {};
  const embeddedSummary = raw.summary || null;

  // Only process "X" (complete) and "i" (instant) events
  const spans = events
    .filter(e => e.ph === 'X' || e.ph === 'i')
    .map(e => ({
      name:      e.name,
      category:  e.cat,
      startUs:   e.ts,
      endUs:     e.ph === 'X' ? e.ts + e.dur : e.ts,
      durationUs: e.ph === 'X' ? e.dur : 0,
      requestId: e.tid === 'system' ? null : e.tid,
      swimlane:  e.tid,
      metadata:  e.args || {},
      isInstant: e.ph === 'i',
    }));

  if (spans.length === 0) {
    return null;
  }

  // Time range
  const minTime = Math.min(...spans.map(s => s.startUs));
  const maxTime = Math.max(...spans.map(s => s.endUs));
  const totalDuration = maxTime - minTime;

  // Group by swimlane (tid)
  const swimlanes = {};
  for (const span of spans) {
    const lane = span.swimlane;
    if (!swimlanes[lane]) {
      swimlanes[lane] = [];
    }
    swimlanes[lane].push(span);
  }

  // Sort swimlanes: "system" first, then req_0, req_1, ...
  const sortedLaneKeys = Object.keys(swimlanes).sort((a, b) => {
    if (a === 'system') return -1;
    if (b === 'system') return 1;
    // Extract numeric suffix for natural sort
    const numA = parseInt(a.replace(/\D/g, ''), 10);
    const numB = parseInt(b.replace(/\D/g, ''), 10);
    if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
    return a.localeCompare(b);
  });

  // Group by category for legend / summary
  const byCategory = {};
  for (const span of spans) {
    if (!byCategory[span.category]) {
      byCategory[span.category] = { totalUs: 0, count: 0, spans: [] };
    }
    byCategory[span.category].totalUs += span.durationUs;
    byCategory[span.category].count += 1;
    byCategory[span.category].spans.push(span);
  }

  // Compute per-category percentages
  const categoryTotalUs = Object.values(byCategory).reduce((s, c) => s + c.totalUs, 0);
  for (const cat of Object.keys(byCategory)) {
    byCategory[cat].pct = categoryTotalUs > 0
      ? Math.round(1000 * byCategory[cat].totalUs / categoryTotalUs) / 10
      : 0;
  }

  // Per-request summary
  const byRequest = {};
  for (const span of spans) {
    if (span.requestId) {
      if (!byRequest[span.requestId]) {
        byRequest[span.requestId] = { startUs: span.startUs, endUs: span.endUs, spanCount: 0 };
      }
      const entry = byRequest[span.requestId];
      entry.startUs = Math.min(entry.startUs, span.startUs);
      entry.endUs = Math.max(entry.endUs, span.endUs);
      entry.spanCount += 1;
    }
  }
  for (const req of Object.keys(byRequest)) {
    byRequest[req].totalUs = byRequest[req].endUs - byRequest[req].startUs;
  }

  return {
    spans,
    swimlanes,
    sortedLaneKeys,
    byCategory,
    byRequest,
    minTime,
    maxTime,
    totalDuration,
    metadata,
    embeddedSummary,
    totalSpans: spans.length,
    totalRequests: Object.keys(byRequest).length,
  };
}

/**
 * React hook for loading and managing trace data.
 */
export default function useTraceData() {
  const [traceData, setTraceData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);

  const loadFromFile = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setFileName(file.name);
    try {
      const text = await file.text();
      const raw = JSON.parse(text);
      const processed = processTraceData(raw);
      if (!processed) {
        throw new Error('No trace events found in file');
      }
      setTraceData(processed);
    } catch (err) {
      setError(err.message);
      setTraceData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadFromUrl = useCallback(async (url, label) => {
    setLoading(true);
    setError(null);
    setFileName(label || url.split('/').pop());
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
      const raw = await resp.json();
      const processed = processTraceData(raw);
      if (!processed) {
        throw new Error('No trace events found in file');
      }
      setTraceData(processed);
    } catch (err) {
      setError(err.message);
      setTraceData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setTraceData(null);
    setFileName(null);
    setError(null);
  }, []);

  return {
    traceData,
    loading,
    error,
    fileName,
    loadFromFile,
    loadFromUrl,
    clear,
  };
}
