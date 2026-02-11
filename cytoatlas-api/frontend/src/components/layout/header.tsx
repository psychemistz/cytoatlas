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
    <header className="sticky top-0 z-50 bg-slate-900 shadow-md">
      <div className="mx-auto flex max-w-[1400px] items-center gap-5 px-5 py-2.5">
        {/* Logo */}
        <Link to="/" className="flex shrink-0 items-center gap-2 text-lg font-bold text-white no-underline hover:text-white">
          <span className="text-2xl">&#128300;</span>
          <span>CytoAtlas</span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-0.5 overflow-x-auto">
          {NAV_LINKS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'whitespace-nowrap rounded-md px-3 py-1.5 text-[13px] font-semibold transition-colors',
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-white hover:bg-white/15',
                )
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex-1" />

        {/* Search */}
        <div className="flex max-w-[280px] items-center">
          <div className="relative flex-1">
            <svg
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400"
              width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search genes..."
              className="w-full rounded-lg border border-slate-600 bg-slate-800 py-1.5 pl-8 pr-3 text-sm text-white placeholder-slate-400 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400"
            />
          </div>
        </div>

        {/* API Docs */}
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="whitespace-nowrap rounded-md border border-slate-600 px-3 py-1.5 text-[13px] font-semibold text-white no-underline hover:bg-white/15"
        >
          API Docs
        </a>
      </div>
    </header>
  );
}
