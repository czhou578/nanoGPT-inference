import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { getCategoriesWithNotes } from './notesRegistry';

function Sidebar({ isOpen, onClose }) {
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState({});
  const location = useLocation();
  const categoriesWithNotes = getCategoriesWithNotes();

  const toggleCategory = (id) => {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const filteredCategories = categoriesWithNotes.map(cat => ({
    ...cat,
    notes: cat.notes.filter(note =>
      note.title.toLowerCase().includes(search.toLowerCase())
    ),
  })).filter(cat => cat.placeholder || cat.notes.length > 0);

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-header">
        <NavLink to="/" className="sidebar-logo" onClick={onClose}>
          <span className="logo-icon">⚡</span>
          <span className="logo-text">Inference KB</span>
        </NavLink>
      </div>

      <div className="sidebar-search">
        <input
          type="text"
          placeholder="Search articles..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="search-input"
        />
        {search && (
          <button className="search-clear" onClick={() => setSearch('')}>×</button>
        )}
      </div>

      <nav className="sidebar-nav">
        {filteredCategories.map(cat => (
          <div key={cat.id} className="nav-category">
            <button
              className={`nav-category-header ${expanded[cat.id] ? 'expanded' : ''}`}
              onClick={() => !cat.placeholder && toggleCategory(cat.id)}
              disabled={cat.placeholder}
            >
              <span className="nav-category-icon">{cat.icon}</span>
              <span className="nav-category-name">{cat.name}</span>
              {cat.placeholder ? (
                <span className="coming-soon-badge">Soon</span>
              ) : (
                <>
                  <span className="nav-category-count">{cat.notes.length}</span>
                  <span className="nav-chevron">{expanded[cat.id] ? '▾' : '▸'}</span>
                </>
              )}
            </button>

            {expanded[cat.id] && !cat.placeholder && (
              <div className="nav-articles">
                {cat.notes.map(note => (
                  <NavLink
                    key={note.slug}
                    to={`/notes/${note.slug}`}
                    className={({ isActive }) => `nav-article-link ${isActive ? 'active' : ''}`}
                    onClick={onClose}
                  >
                    {note.title}
                  </NavLink>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <NavLink
          to="/bibliography"
          className={({ isActive }) => `sidebar-footer-link ${isActive ? 'active' : ''}`}
          onClick={onClose}
        >
          📚 Bibliography
        </NavLink>
        <NavLink
          to="/visualizer"
          className="sidebar-footer-link"
          onClick={onClose}
        >
          🔬 Inference Visualizer
        </NavLink>
      </div>
    </aside>
  );
}

export default Sidebar;
