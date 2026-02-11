import { Link, useNavigate } from 'react-router';
import { useAtlases } from '@/api/hooks/use-atlas';
import { STATS, EXAMPLE_GENES, PLACEHOLDER_ATLASES } from '@/lib/constants';
import { AtlasCard } from '@/components/ui/atlas-card';

const CORE_ATLAS_NAMES = ['cima', 'inflammation', 'scatlas'];

export default function Home() {
  const navigate = useNavigate();
  const { data: atlases } = useAtlases();

  const coreAtlases = atlases?.filter((a) => CORE_ATLAS_NAMES.includes(a.name)) ?? PLACEHOLDER_ATLASES;

  return (
    <div className="mx-auto max-w-[1400px] px-8 py-8">
      {/* Hero */}
      <section className="relative mb-10 overflow-hidden rounded-2xl bg-gradient-to-br from-[#0f172a] via-[#1e3a8a] to-[#1e40af] px-8 py-16 text-center shadow-xl">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(59,130,246,0.3),transparent_60%)]" />
        <div className="relative mx-auto max-w-[800px]">
          <h1 className="mb-5 text-[2.75rem] font-bold leading-tight tracking-tight text-white">
            Pan-Disease Cytokine Activity Atlas
          </h1>
          <p className="mb-8 text-lg leading-relaxed text-blue-100">
            Explore cytokine and secreted protein activities across{' '}
            <strong className="text-white">17+ million immune cells</strong> from multiple single-cell atlases
          </p>

          {/* Quick gene links */}
          <div className="flex flex-wrap items-center justify-center gap-3">
            <span className="text-sm font-bold text-white">Try:</span>
            {EXAMPLE_GENES.map((gene) => (
              <Link
                key={gene}
                to={`/gene/${gene}`}
                className="rounded-full bg-blue-500 px-5 py-1.5 text-sm font-bold text-white no-underline shadow-md transition-all hover:-translate-y-0.5 hover:bg-blue-400 hover:text-white hover:shadow-lg"
              >
                {gene}
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="mb-10">
        <div className="grid grid-cols-2 gap-5 md:grid-cols-4">
          {STATS.map(({ value, label }) => (
            <div
              key={label}
              className="rounded-xl border border-border-light bg-white p-6 text-center shadow-sm"
            >
              <div className="mb-1 text-[2rem] font-extrabold text-primary">{value}</div>
              <div className="text-sm font-medium text-text-secondary">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Core Atlases */}
      <section className="mb-10">
        <h2 className="mb-1 text-2xl font-bold text-text-primary">Core Atlases</h2>
        <p className="mb-6 text-sm text-text-muted">Three curated single-cell datasets with full CytoSig and SecAct inference</p>
        <div className="grid gap-6" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
          {coreAtlases.map((atlas) => (
            <AtlasCard
              key={atlas.name}
              name={atlas.name}
              displayName={atlas.display_name}
              description={atlas.description}
              nCells={atlas.n_cells}
              nSamples={atlas.n_samples}
              nCellTypes={atlas.n_cell_types}
            />
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mb-10">
        <h2 className="mb-1 text-2xl font-bold text-text-primary">What You Can Do</h2>
        <p className="mb-6 text-sm text-text-muted">Interactive tools for exploring, validating, and comparing cytokine activities</p>
        <div className="grid gap-5" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
          {FEATURES.map(({ icon, title, description, to, quickLinks }) => (
            <div
              key={title}
              onClick={() => navigate(to)}
              className="group cursor-pointer rounded-xl border border-border-light bg-white p-7 text-center shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-lg"
            >
              <div className="mb-4 text-[2.5rem]">{icon}</div>
              <h3 className="mb-2 text-base font-bold text-text-primary">{title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{description}</p>
              {quickLinks && (
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {quickLinks.map(({ label, href }) => (
                    <Link
                      key={label}
                      to={href}
                      onClick={(e) => e.stopPropagation()}
                      className="rounded-full bg-bg-tertiary px-3 py-1 text-xs font-medium text-primary no-underline hover:bg-primary/10"
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
