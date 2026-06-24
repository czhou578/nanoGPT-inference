/**
 * TimelineChart - SVG-based flame chart for inference trace visualization.
 *
 * Each request is a horizontal swimlane. Spans are colored rectangles
 * positioned by (startUs, endUs) on the x-axis and by request on the y-axis.
 * Instant events render as diamond markers.
 *
 * SVG is used instead of Canvas for built-in interactivity (hover, click).
 * For traces from NanoGPT (typically 100-500 spans), SVG performance is fine.
 */

import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { getCategoryColor } from './useTraceData';

const LANE_HEIGHT = 48;
const LANE_GAP = 6;
const LANE_PADDING = 4;
const SPAN_HEIGHT = 32;
const LABEL_WIDTH = 120;
const PADDING_TOP = 20;
const PADDING_BOTTOM = 20;
const TICK_HEIGHT = 30;
const MIN_SPAN_WIDTH = 3;
const INSTANT_SIZE = 8;

function formatTime(us) {
  if (us < 1000) return `${us}µs`;
  if (us < 1000000) return `${(us / 1000).toFixed(1)}ms`;
  return `${(us / 1000000).toFixed(2)}s`;
}

export default function TimelineChart({ traceData, onSpanHover, onSpanClick }) {
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(900);
  const [hoveredSpan, setHoveredSpan] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Measure container
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const {
    swimlanes,
    sortedLaneKeys,
    minTime,
    maxTime,
    totalDuration,
  } = traceData;

  const chartWidth = containerWidth - LABEL_WIDTH - 24;
  const totalHeight = PADDING_TOP + TICK_HEIGHT
    + sortedLaneKeys.length * (LANE_HEIGHT + LANE_GAP)
    + PADDING_BOTTOM;

  // Time → x position
  const timeToX = useCallback((us) => {
    if (totalDuration === 0) return LABEL_WIDTH;
    return LABEL_WIDTH + ((us - minTime) / totalDuration) * chartWidth;
  }, [minTime, totalDuration, chartWidth]);

  // Compute tick marks
  const ticks = useMemo(() => {
    const targetTickCount = Math.max(4, Math.floor(chartWidth / 100));
    const rawStep = totalDuration / targetTickCount;

    // Round to a nice number
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalized = rawStep / magnitude;
    let niceStep;
    if (normalized <= 1.5) niceStep = magnitude;
    else if (normalized <= 3.5) niceStep = 2 * magnitude;
    else if (normalized <= 7.5) niceStep = 5 * magnitude;
    else niceStep = 10 * magnitude;

    const result = [];
    let t = Math.ceil(minTime / niceStep) * niceStep;
    while (t <= maxTime) {
      result.push(t);
      t += niceStep;
    }
    return result;
  }, [minTime, maxTime, totalDuration, chartWidth]);

  const handleSpanMouseEnter = useCallback((span, event) => {
    setHoveredSpan(span);
    const rect = containerRef.current.getBoundingClientRect();
    setTooltipPos({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
    onSpanHover?.(span);
  }, [onSpanHover]);

  const handleSpanMouseMove = useCallback((event) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setTooltipPos({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  }, []);

  const handleSpanMouseLeave = useCallback(() => {
    setHoveredSpan(null);
    onSpanHover?.(null);
  }, [onSpanHover]);

  const handleSpanClick = useCallback((span) => {
    onSpanClick?.(span);
  }, [onSpanClick]);

  return (
    <div className="timeline-chart-container" ref={containerRef}>
      <svg
        width={containerWidth}
        height={totalHeight}
        className="timeline-svg"
      >
        {/* Background */}
        <rect
          x={0}
          y={0}
          width={containerWidth}
          height={totalHeight}
          fill="transparent"
        />

        {/* Time axis ticks */}
        <g className="timeline-ticks">
          {ticks.map((t, i) => {
            const x = timeToX(t);
            return (
              <g key={i}>
                <line
                  x1={x} y1={PADDING_TOP}
                  x2={x} y2={totalHeight - PADDING_BOTTOM}
                  stroke="rgba(255,255,255,0.06)"
                  strokeWidth={1}
                />
                <text
                  x={x}
                  y={PADDING_TOP + 16}
                  fill="#6b7280"
                  fontSize="11"
                  fontFamily="'SF Mono', 'Fira Code', monospace"
                  textAnchor="middle"
                >
                  {formatTime(t - minTime)}
                </text>
              </g>
            );
          })}
        </g>

        {/* Swimlanes */}
        {sortedLaneKeys.map((laneKey, laneIdx) => {
          const laneY = PADDING_TOP + TICK_HEIGHT + laneIdx * (LANE_HEIGHT + LANE_GAP);
          const spans = swimlanes[laneKey];
          const isSystem = laneKey === 'system';
          const laneLabel = isSystem ? 'system' : laneKey;

          return (
            <g key={laneKey} className="timeline-lane">
              {/* Lane background */}
              <rect
                x={LABEL_WIDTH}
                y={laneY}
                width={chartWidth}
                height={LANE_HEIGHT}
                fill={isSystem ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.015)'}
                rx={6}
              />

              {/* Lane label */}
              <text
                x={LABEL_WIDTH - 12}
                y={laneY + LANE_HEIGHT / 2 + 4}
                fill={isSystem ? '#9ca3af' : '#a5b4fc'}
                fontSize="12"
                fontFamily="'SF Mono', 'Fira Code', monospace"
                fontWeight="500"
                textAnchor="end"
              >
                {laneLabel}
              </text>

              {/* Spans */}
              {spans.map((span, spanIdx) => {
                const color = getCategoryColor(span.category);
                const isHovered = hoveredSpan === span;

                if (span.isInstant) {
                  // Instant event → diamond marker
                  const cx = timeToX(span.startUs);
                  const cy = laneY + LANE_HEIGHT / 2;
                  return (
                    <g
                      key={spanIdx}
                      className="timeline-instant"
                      onMouseEnter={(e) => handleSpanMouseEnter(span, e)}
                      onMouseMove={handleSpanMouseMove}
                      onMouseLeave={handleSpanMouseLeave}
                      onClick={() => handleSpanClick(span)}
                      style={{ cursor: 'pointer' }}
                    >
                      <polygon
                        points={`${cx},${cy - INSTANT_SIZE} ${cx + INSTANT_SIZE},${cy} ${cx},${cy + INSTANT_SIZE} ${cx - INSTANT_SIZE},${cy}`}
                        fill={color.bg}
                        fillOpacity={isHovered ? 1 : 0.7}
                        stroke={isHovered ? '#fff' : 'none'}
                        strokeWidth={isHovered ? 1.5 : 0}
                      />
                    </g>
                  );
                }

                // Duration span → rectangle
                const x1 = timeToX(span.startUs);
                const x2 = timeToX(span.endUs);
                const rawWidth = x2 - x1;
                const width = Math.max(rawWidth, MIN_SPAN_WIDTH);
                const spanY = laneY + LANE_PADDING + (LANE_HEIGHT - SPAN_HEIGHT) / 2 - LANE_PADDING;

                return (
                  <g
                    key={spanIdx}
                    className="timeline-span"
                    onMouseEnter={(e) => handleSpanMouseEnter(span, e)}
                    onMouseMove={handleSpanMouseMove}
                    onMouseLeave={handleSpanMouseLeave}
                    onClick={() => handleSpanClick(span)}
                    style={{ cursor: 'pointer' }}
                  >
                    <rect
                      x={x1}
                      y={spanY}
                      width={width}
                      height={SPAN_HEIGHT}
                      rx={4}
                      fill={color.bg}
                      fillOpacity={isHovered ? 0.95 : 0.75}
                      stroke={isHovered ? '#fff' : 'rgba(255,255,255,0.1)'}
                      strokeWidth={isHovered ? 1.5 : 0.5}
                    />
                    {/* Show name label if span is wide enough */}
                    {width > 50 && (
                      <text
                        x={x1 + 6}
                        y={spanY + SPAN_HEIGHT / 2 + 4}
                        fill={color.fg}
                        fontSize="10"
                        fontFamily="'Inter', sans-serif"
                        fontWeight="500"
                        clipPath={`inset(0 0 0 0)`}
                      >
                        {span.name.length > Math.floor(width / 6)
                          ? span.name.slice(0, Math.floor(width / 6)) + '…'
                          : span.name
                        }
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {hoveredSpan && (
        <div
          className="timeline-tooltip"
          style={{
            left: Math.min(tooltipPos.x + 16, containerWidth - 280),
            top: tooltipPos.y - 10,
          }}
        >
          <div className="tooltip-header">
            <span
              className="tooltip-swatch"
              style={{ backgroundColor: getCategoryColor(hoveredSpan.category).bg }}
            />
            <span className="tooltip-name">{hoveredSpan.name}</span>
          </div>
          <div className="tooltip-row">
            <span className="tooltip-key">Category</span>
            <span className="tooltip-val">{getCategoryColor(hoveredSpan.category).label}</span>
          </div>
          {!hoveredSpan.isInstant && (
            <div className="tooltip-row">
              <span className="tooltip-key">Duration</span>
              <span className="tooltip-val">{formatTime(hoveredSpan.durationUs)}</span>
            </div>
          )}
          {hoveredSpan.swimlane !== 'system' && (
            <div className="tooltip-row">
              <span className="tooltip-key">Request</span>
              <span className="tooltip-val">{hoveredSpan.swimlane}</span>
            </div>
          )}
          {/* Metadata entries */}
          {Object.entries(hoveredSpan.metadata).map(([key, val]) => (
            <div className="tooltip-row" key={key}>
              <span className="tooltip-key">{key}</span>
              <span className="tooltip-val">
                {typeof val === 'boolean' ? (val ? '✓' : '✗') : String(val)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
