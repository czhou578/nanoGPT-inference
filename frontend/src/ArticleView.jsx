import { useParams, NavLink } from 'react-router-dom';
import { useState, useEffect, useRef, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { getNoteBySlug, categories } from './notesRegistry';

function ArticleView() {
  const { slug } = useParams();
  const note = getNoteBySlug(slug);
  const [readingProgress, setReadingProgress] = useState(0);
  const [activeSection, setActiveSection] = useState('');
  const contentRef = useRef(null);

  // Extract TOC from markdown headings
  const toc = useMemo(() => {
    if (!note) return [];
    const headings = [];
    const lines = note.content.split('\n');
    for (const line of lines) {
      const match = line.match(/^(#{1,3})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const text = match[2].replace(/[*_`]/g, '');
        const id = text
          .toLowerCase()
          .replace(/[^a-z0-9\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .replace(/^-|-$/g, '');
        headings.push({ level, text, id });
      }
    }
    return headings;
  }, [note]);

  // Reading progress
  useEffect(() => {
    const handleScroll = () => {
      const el = contentRef.current;
      if (!el) return;
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      setReadingProgress(docHeight > 0 ? Math.min((scrollTop / docHeight) * 100, 100) : 0);

      // Update active section for TOC highlighting
      const headingElements = el.querySelectorAll('h1[id], h2[id], h3[id]');
      let current = '';
      for (const heading of headingElements) {
        if (heading.getBoundingClientRect().top <= 120) {
          current = heading.id;
        }
      }
      setActiveSection(current);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [slug]);

  // Scroll to top on article change
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  if (!note) {
    return (
      <div className="article-not-found">
        <h2>Article not found</h2>
        <p>The article "{slug}" doesn't exist.</p>
        <NavLink to="/" className="back-link">← Back to Knowledge Base</NavLink>
      </div>
    );
  }

  const category = categories.find(c => c.id === note.categoryId);

  const scrollToHeading = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Custom heading renderer that adds IDs for TOC linking
  const headingRenderer = (level) => {
    const Tag = `h${level}`;
    return ({ children, ...props }) => {
      const text = typeof children === 'string'
        ? children
        : Array.isArray(children)
          ? children.map(c => (typeof c === 'string' ? c : c?.props?.children || '')).join('')
          : '';
      const id = text
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '');
      return <Tag id={id} {...props}>{children}</Tag>;
    };
  };

  return (
    <div className="article-view">
      {/* Reading progress bar */}
      <div className="reading-progress-bar">
        <div
          className="reading-progress-fill"
          style={{ width: `${readingProgress}%` }}
        />
      </div>

      <div className="article-layout">
        {/* Table of Contents */}
        {toc.length > 3 && (
          <aside className="article-toc">
            <div className="toc-sticky">
              <h4 className="toc-title">On this page</h4>
              <nav className="toc-nav">
                {toc.map((heading, i) => (
                  <button
                    key={i}
                    className={`toc-link toc-level-${heading.level} ${activeSection === heading.id ? 'active' : ''}`}
                    onClick={() => scrollToHeading(heading.id)}
                  >
                    {heading.text}
                  </button>
                ))}
              </nav>
            </div>
          </aside>
        )}

        {/* Article content */}
        <article className="article-body" ref={contentRef}>
          <div className="article-meta">
            {category && (
              <span className="article-category-badge">
                {category.icon} {category.name}
              </span>
            )}
          </div>

          <ReactMarkdown
            remarkPlugins={[remarkMath, remarkGfm]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
            components={{
              h1: headingRenderer(1),
              h2: headingRenderer(2),
              h3: headingRenderer(3),
              // Style tables
              table: ({ children, ...props }) => (
                <div className="table-wrapper">
                  <table {...props}>{children}</table>
                </div>
              ),
              // Style code blocks
              pre: ({ children, ...props }) => (
                <div className="code-block-wrapper">
                  <pre {...props}>{children}</pre>
                </div>
              ),
              // Style blockquotes
              blockquote: ({ children, ...props }) => (
                <blockquote className="styled-blockquote" {...props}>{children}</blockquote>
              ),
            }}
          >
            {note.content}
          </ReactMarkdown>
        </article>
      </div>
    </div>
  );
}

export default ArticleView;
