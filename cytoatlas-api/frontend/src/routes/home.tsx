import { useState, type KeyboardEvent } from 'react';
import { Link, useNavigate } from 'react-router';
import { useAtlases } from '@/api/hooks/use-atlas';
import { STATS, EXAMPLE_GENES, PLACEHOLDER_ATLASES } from '@/lib/constants';
import { AtlasCard } from '@/components/ui/atlas-card';

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
    <div className="mx-auto max-w-[1400px] px-8 py-8">
      {/* Hero */}
      <section className="mb-8 rounded-xl bg-gradient-to-br from-primary to-primary-dark px-6 py-12 text-center text-text-inverse">
        <div className="mx-auto max-w-[800px]">
          <h1 className="mb-4 text-[2.5rem] font-semibold leading-tight text-text-inverse">
            Pan-Disease Cytokine Activity Atlas
          </h1>
          <p className="mb-8 text-xl opacity-90">
            Explore cytokine and secreted protein activities across{' '}
            <strong>17+ million immune cells</strong> from multiple single-cell atlases
          </p>

          <div className="mx-auto mb-6 flex max-w-[600px] gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search genes (e.g., IFNG, TNF, IL6, IL17A...)"
              className="flex-1 rounded-lg border-none px-6 py-3 text-lg text-text-primary outline-none"
            />
            <button
              onClick={handleSearch}
              className="rounded-lg bg-accent px-8 py-3 font-semibold text-text-inverse hover:bg-[#0d9668]"
            >
              Search
            </button>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2">
            <span className="opacity-80">Try:</span>
            {EXAMPLE_GENES.map((gene) => (
              <Link
                key={gene}
                to={`/gene/${gene}`}
                className="rounded-md border border-white/30 bg-white/20 px-3 py-1 text-sm text-text-inverse no-underline hover:bg-white/30"
              >
                {gene}
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="mb-8">
        <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
          {STATS.map(({ value, label }) => (
            <div
              key={label}
              className="rounded-lg border border-border-light bg-bg-primary p-6 text-center shadow-sm"
            >
              <div className="mb-1 text-[2rem] font-bold text-primary">{value}</div>
              <div className="text-sm text-text-secondary">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Atlas Cards */}
      <section className="mb-8">
        <h2 className="mb-6 text-[1.75rem] font-semibold">Core Atlases</h2>
        <div className="grid gap-6" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' }}>
          {coreAtlases.map((atlas) => (
            <AtlasCard
              key={atlas.name}
              name={atlas.name}
              displayName={atlas.display_name}
              description={atlas.description}
              nCells={atlas.n_cells}
              nSamples={atlas.n_samples}
              nCellTypes={atlas.n_cell_types}
              validationGrade={atlas.validation_grade}
              sourceType={atlas.source_type}
            />
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mb-8">
        <h2 className="mb-6 text-[1.75rem] font-semibold">What You Can Do</h2>
        <div className="grid gap-6" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))' }}>
          {FEATURES.map(({ icon, title, description, to, quickLinks }) => (
            <div
              key={title}
              onClick={() => navigate(to)}
              className="cursor-pointer rounded-lg border border-border-light bg-bg-primary p-8 text-center shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="mb-4 text-[2.5rem]">{icon}</div>
              <h3 className="mb-2 font-semibold">{title}</h3>
              <p className="text-[0.9375rem] text-text-secondary">{description}</p>
              {quickLinks && (
                <div className="mt-3 flex flex-wrap justify-center gap-2">
                  {quickLinks.map(({ label, href }) => (
                    <Link
                      key={label}
                      to={href}
                      onClick={(e) => e.stopPropagation()}
                      className="rounded px-2 py-0.5 text-xs text-text-secondary no-underline hover:bg-bg-tertiary"
                    >
                      {label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
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
    icon: '\u{1F50E}',
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
