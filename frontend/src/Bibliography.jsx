import { useState, useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import { getAllSources } from './notesRegistry';

function Bibliography() {
  const [filter, setFilter] = useState('all');
  const allSources = useMemo(() => getAllSources(), []);

  // Deduplicate and group sources
  const grouped = useMemo(() => {
    const seen = new Set();
    const papers = [];
    const links = [];
    const primary = [];

    for (const source of allSources) {
      const key = source.url || source.text;
      if (seen.has(key)) {
        // Add note reference to existing entry
        const existing =
          papers.find(s => (s.url || s.text) === key) ||
          links.find(s => (s.url || s.text) === key) ||
          primary.find(s => (s.url || s.text) === key);
        if (existing && !existing.referencedBy.includes(source.noteTitle)) {
          existing.referencedBy.push(source.noteTitle);
          existing.referencedSlugs.push(source.noteSlug);
        }
        continue;
      }
      seen.add(key);

      const entry = {
        ...source,
        referencedBy: [source.noteTitle],
        referencedSlugs: [source.noteSlug],
      };

      if (source.type === 'paper') {
        papers.push(entry);
      } else if (source.type === 'link') {
        links.push(entry);
      } else {
        primary.push(entry);
      }
    }

    return { papers, links, primary };
  }, [allSources]);

  const getFilteredSources = () => {
    switch (filter) {
      case 'papers': return { papers: grouped.papers, links: [], primary: [] };
      case 'links': return { papers: [], links: grouped.links, primary: [] };
      case 'primary': return { papers: [], links: [], primary: grouped.primary };
      default: return grouped;
    }
  };

  const filtered = getFilteredSources();

  const renderSource = (source, index) => (
    <div key={index} className="bib-entry">
      <div className="bib-content">
        {source.url ? (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="bib-link"
          >
            {source.text || source.url}
          </a>
        ) : (
          <span className="bib-text">{source.text}</span>
        )}
      </div>
      <div className="bib-refs">
        <span className="bib-refs-label">Referenced in:</span>
        {source.referencedBy.map((title, i) => (
          <NavLink
            key={i}
            to={`/notes/${source.referencedSlugs[i]}`}
            className="bib-ref-link"
          >
            {title}
          </NavLink>
        ))}
      </div>
    </div>
  );

  const totalCount =
    filtered.papers.length + filtered.links.length + filtered.primary.length;

  return (
    <div className="bibliography-page">
      <div className="bib-header">
        <h1 className="bib-title">📚 Bibliography</h1>
        <p className="bib-subtitle">
          All sources referenced across the knowledge base
        </p>

        <div className="bib-filters">
          {[
            { id: 'all', label: 'All' },
            { id: 'papers', label: 'Papers' },
            { id: 'primary', label: 'Primary Sources' },
            { id: 'links', label: 'URLs' },
          ].map(f => (
            <button
              key={f.id}
              className={`bib-filter-btn ${filter === f.id ? 'active' : ''}`}
              onClick={() => setFilter(f.id)}
            >
              {f.label}
            </button>
          ))}
        </div>

        <p className="bib-count">{totalCount} source{totalCount !== 1 ? 's' : ''}</p>
      </div>

      <div className="bib-sections">
        {filtered.primary.length > 0 && (
          <section className="bib-section">
            <h2 className="bib-section-title">Primary Sources</h2>
            {filtered.primary.map(renderSource)}
          </section>
        )}

        {filtered.papers.length > 0 && (
          <section className="bib-section">
            <h2 className="bib-section-title">Academic Papers & References</h2>
            {filtered.papers.map(renderSource)}
          </section>
        )}

        {filtered.links.length > 0 && (
          <section className="bib-section">
            <h2 className="bib-section-title">URLs & External Resources</h2>
            {filtered.links.map(renderSource)}
          </section>
        )}
      </div>
    </div>
  );
}

export default Bibliography;
