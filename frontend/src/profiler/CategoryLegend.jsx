/**
 * CategoryLegend - Color legend for span categories.
 *
 * Shows each active category as a colored pill with its label and
 * time percentage. Used at the top of the timeline to help users
 * decode span colors at a glance.
 */

import { getCategoryColor, CATEGORY_COLORS } from './useTraceData';

export default function CategoryLegend({ byCategory }) {
  if (!byCategory) return null;

  // Sort categories by total time descending so the most prominent shows first
  const sortedCats = Object.entries(byCategory)
    .sort(([, a], [, b]) => b.totalUs - a.totalUs);

  return (
    <div className="profiler-legend">
      {sortedCats.map(([cat, data]) => {
        const color = getCategoryColor(cat);
        return (
          <div key={cat} className="legend-item">
            <span
              className="legend-swatch"
              style={{ backgroundColor: color.bg }}
            />
            <span className="legend-label">{color.label}</span>
            <span className="legend-pct">{data.pct}%</span>
          </div>
        );
      })}
    </div>
  );
}
