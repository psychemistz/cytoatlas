import { useState, type KeyboardEvent } from 'react';
import { Link, useNavigate } from 'react-router';
import { useAtlases } from '@/api/hooks/use-atlas';
import { STATS, EXAMPLE_GENES, PLACEHOLDER_ATLASES } from '@/lib/constants';
import { formatNumber } from '@/lib/utils';

export default function Home() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const { data: atlases } = useAtlases();

  const coreAtlases = atlases?.filter((a) => a.source_type === 'builtin') ?? PLACEHOLDER_ATLASES;

  function handleSearch() {
    const trimmed = query.trim().toUpperCase();
    if (trimmed) navigate(`/gene/${encodeURIComponent(trimmed)}`);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleSearch();
  }

  return (
    <div>
      {/* Hero */}
      <section className="bg-gradient-to-b from-bg-secondary to-bg-primary px-4 py-16 text-center">
        <h1 className="mb-4 text-4xl font-bold text-text-primary">Pan-Disease Cytokine Activity Atlas</h1>
        <p className="mx-auto mb-8 max-w-2xl text-lg text-text-secondary">
          Explore cytokine and secreted protein activities across <strong>17+ million immune cells</strong> from
          multiple single-cell atlases
        </p>
        <div className="mx-auto flex max-w-lg gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search genes (e.g., IFNG, TNF, IL6, IL17A...)"
            className="flex-1 rounded-lg border border-border-light px-4 py-3 text-base outline-none focus:border-primary"
          />
          <button
            onClick={handleSearch}
            className="rounded-lg bg-primary px-6 py-3 font-medium text-text-inverse hover:bg-primary-dark"
          >
            Search
          </button>
        </div>
        <div className="mt-4 flex items-center justify-center gap-2 text-sm text-text-muted">
          <span>Try:</span>
          {EXAMPLE_GENES.map((gene) => (
            <Link
              key={gene}
              to={`/gene/${gene}`}
              className="rounded-md bg-bg-tertiary px-2.5 py-1 text-primary hover:bg-border-light"
            >
              {gene}
            </Link>
          ))}
        </div>
      </section>

      {/* Stats */}
      <section className="border-b border-border-light bg-bg-primary px-4 py-12">
        <div className="mx-auto grid max-w-4xl grid-cols-2 gap-6 md:grid-cols-4">
          {STATS.map(({ value, label }) => (
            <div key={label} className="text-center">
              <div className="text-3xl font-bold text-primary">{value}</div>
              <div className="mt-1 text-sm text-text-secondary">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Atlas Cards */}
      <section className="px-4 py-12">
        <h2 className="mb-8 text-center text-2xl font-bold">Core Atlases</h2>
        <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3">
          {coreAtlases.map((atlas) => (
            <Link
              key={atlas.name}
              to={`/atlas/${atlas.name}`}
              className="rounded-xl border border-border-light p-6 shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-text-primary">{atlas.display_name}</h3>
                <span className="rounded bg-accent/10 px-2 py-0.5 text-xs font-medium text-accent">
                  Grade {atlas.validation_grade}
                </span>
              </div>
              <p className="mb-4 text-sm text-text-secondary">{atlas.description}</p>
              <div className="flex gap-4 text-xs text-text-muted">
                <span>{formatNumber(atlas.n_cells)} cells</span>
                <span>{atlas.n_samples.toLocaleString()} samples</span>
                <span>{atlas.n_cell_types} cell types</span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="bg-bg-secondary px-4 py-12">
        <h2 className="mb-8 text-center text-2xl font-bold">What You Can Do</h2>
        <div className="mx-auto grid max-w-5xl gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(({ icon, title, description, to, quickLinks }) => (
            <Link
              key={title}
              to={to}
              className="rounded-xl border border-border-light bg-bg-primary p-6 shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="mb-3 text-3xl">{icon}</div>
              <h3 className="mb-2 text-lg font-semibold text-text-primary">{title}</h3>
              <p className="mb-3 text-sm text-text-secondary">{description}</p>
              {quickLinks && (
                <div className="flex gap-2">
                  {quickLinks.map(({ label, href }) => (
                    <span
                      key={label}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        navigate(href);
                      }}
                      className="cursor-pointer rounded bg-bg-tertiary px-2 py-0.5 text-xs text-primary hover:bg-border-light"
                    >
                      {label}
                    </span>
                  ))}
                </div>
              )}
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

interface Feature {
  icon: string;
  title: string;
  description: string;
  to: string;
  quickLinks?: { label: string; href: string }[];
}

const FEATURES: Feature[] = [
  {
    icon: '\u{1F50D}',
    title: 'Search',
    description: 'Search genes to view expression and cytokine/protein activity across all atlases',
    to: '/search',
    quickLinks: [
      { label: 'IFNG', href: '/gene/IFNG' },
      { label: 'TNF', href: '/gene/TNF' },
      { label: 'IL6', href: '/gene/IL6' },
    ],
  },
  {
    icon: '\u{1F4CA}',
    title: 'Explore',
    description: 'Interactive heatmaps, scatter plots, and correlation analyses',
    to: '/explore',
  },
  {
    icon: '\u{2705}',
    title: 'Validate',
    description: '5-type credibility assessment for inference quality',
    to: '/validate',
  },
  {
    icon: '\u{1F4C8}',
    title: 'Compare',
    description: 'Cross-atlas comparison of cytokine activities',
    to: '/compare',
  },
  {
    icon: '\u{1F4C1}',
    title: 'Submit',
    description: 'Upload your H5AD data for CytoSig/SecAct inference',
    to: '/submit',
  },
  {
    icon: '\u{1F4AC}',
    title: 'Chat',
    description: 'Ask questions with AI-powered natural language assistant',
    to: '/chat',
  },
];
