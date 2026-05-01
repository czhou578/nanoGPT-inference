import { NavLink } from 'react-router-dom';
import { getCategoriesWithNotes } from './notesRegistry';

function LandingPage() {
  const categoriesWithNotes = getCategoriesWithNotes();

  return (
    <div className="landing-page">
      <div className="landing-hero">
        <div className="hero-glow" />
        <h1 className="landing-title">
          LLM Inference
          <span className="title-accent"> Knowledge Base</span>
        </h1>
        <p className="landing-subtitle">
          First-principles explanations of inference optimization techniques, hardware fundamentals, and investment frameworks.
        </p>
        <div className="landing-stats">
          <div className="stat">
            <span className="stat-number">{categoriesWithNotes.filter(c => !c.placeholder).length}</span>
            <span className="stat-label">Topics</span>
          </div>
          <div className="stat">
            <span className="stat-number">{categoriesWithNotes.reduce((sum, c) => sum + c.notes.length, 0)}</span>
            <span className="stat-label">Articles</span>
          </div>
          <div className="stat">
            <span className="stat-number">{categoriesWithNotes.filter(c => c.placeholder).length}</span>
            <span className="stat-label">Coming Soon</span>
          </div>
        </div>
      </div>

      <div className="category-grid">
        {categoriesWithNotes.map(cat => (
          <div
            key={cat.id}
            className={`category-card ${cat.placeholder ? 'placeholder' : ''}`}
          >
            <div className="card-icon">{cat.icon}</div>
            <h3 className="card-title">{cat.name}</h3>

            {cat.placeholder ? (
              <span className="card-badge">Coming Soon</span>
            ) : (
              <>
                <span className="card-count">
                  {cat.notes.length} article{cat.notes.length !== 1 ? 's' : ''}
                </span>
                <div className="card-articles">
                  {cat.notes.map(note => (
                    <NavLink
                      key={note.slug}
                      to={`/notes/${note.slug}`}
                      className="card-article-link"
                    >
                      {note.title}
                    </NavLink>
                  ))}
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default LandingPage;
