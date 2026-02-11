import { useState, type KeyboardEvent } from 'react';
import { Link, NavLink, useNavigate } from 'react-router';
import { cn } from '@/lib/utils';

const NAV_LINKS = [
  { to: '/', label: 'Home' },
  { to: '/search', label: 'Search' },
  { to: '/explore', label: 'Explore' },
  { to: '/validate', label: 'Validate' },
  { to: '/compare', label: 'Compare' },
  { to: '/perturbation', label: 'Perturbation' },
  { to: '/spatial', label: 'Spatial' },
  { to: '/submit', label: 'Submit' },
  { to: '/chat', label: 'Chat' },
  { to: '/about', label: 'About' },
] as const;

export function Header() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');

  function handleSearch() {
    const trimmed = query.trim().toUpperCase();
    if (trimmed) {
      navigate(`/gene/${encodeURIComponent(trimmed)}`);
      setQuery('');
    }
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleSearch();
  }

  return (
    <header className="sticky top-0 z-50 border-b border-border-light bg-bg-primary shadow-sm">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 text-lg font-bold text-text-primary no-underline">
          <span className="text-2xl">&#128300;</span>
          <span>CytoAtlas</span>
        </Link>

        {/* Navigation */}
        <nav className="flex flex-1 items-center gap-1 overflow-x-auto">
          {NAV_LINKS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'whitespace-nowrap rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-text-inverse'
                    : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary',
                )
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Search */}
        <div className="flex items-center gap-1">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search genes..."
            className="w-40 rounded-md border border-border-light px-3 py-1.5 text-sm outline-none focus:border-primary"
          />
          <button
            onClick={handleSearch}
            className="rounded-md bg-primary px-3 py-1.5 text-sm text-text-inverse hover:bg-primary-dark"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          </button>
        </div>

        {/* API Docs */}
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="whitespace-nowrap rounded-md border border-border-light px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary"
        >
          API Docs
        </a>
      </div>
    </header>
  );
}
