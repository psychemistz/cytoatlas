import { useState, type KeyboardEvent } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router';

const POPULAR_GENES = [
  { gene: 'IFNG', cs: true, sa: true },
  { gene: 'TNF', cs: true, sa: true },
  { gene: 'IL6', cs: true, sa: true },
  { gene: 'IL17A', cs: true, sa: false },
  { gene: 'IL10', cs: true, sa: true },
  { gene: 'TGFB1', cs: true, sa: true },
  { gene: 'IL1B', cs: true, sa: true },
  { gene: 'CCL2', cs: false, sa: true },
  { gene: 'CXCL10', cs: false, sa: true },
  { gene: 'IL2', cs: true, sa: false },
];

const TABS_INFO = [
  { icon: '\u{1F9EC}', title: 'Gene Expression', desc: 'Expression levels by cell type across atlases' },
  { icon: '\u{1F4CA}', title: 'CytoSig Activity', desc: 'Cytokine signaling activity (44 cytokines)' },
  { icon: '\u{1F52C}', title: 'SecAct Activity', desc: 'Secreted protein activity (1,170 proteins)' },
  { icon: '\u{1FA7A}', title: 'Disease Associations', desc: 'Differential activity in diseases vs healthy' },
  { icon: '\u{1F4C8}', title: 'Correlations', desc: 'Age, BMI, and biochemistry correlations' },
];

export default function Search() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get('q') ?? '');

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
      <section className="relative mb-10 overflow-hidden rounded-2xl bg-gradient-to-br from-[#0f172a] via-[#1e3a8a] to-[#1e40af] px-8 py-14 text-center shadow-xl">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(59,130,246,0.3),transparent_60%)]" />
        <div className="relative mx-auto max-w-[700px]">
          <h1 className="mb-4 text-[2.5rem] font-bold leading-tight tracking-tight text-white">
            Gene Search
          </h1>
          <p className="mb-8 text-lg leading-relaxed text-blue-100">
            Enter a gene symbol to view expression, cytokine/protein activity,
            disease associations, and correlations across all atlases
          </p>

          {/* Search bar */}
          <div className="mx-auto flex max-w-xl gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter gene symbol (e.g., IFNG, TNF, IL6...)"
              autoFocus
              className="flex-1 rounded-lg border-0 bg-white/95 px-5 py-3.5 text-base text-slate-900 shadow-md outline-none placeholder:text-slate-400 focus:ring-2 focus:ring-blue-300"
            />
            <button
              onClick={handleSearch}
              className="rounded-lg bg-blue-500 px-7 py-3.5 font-semibold text-white shadow-md transition-all hover:-translate-y-0.5 hover:bg-blue-400 hover:shadow-lg"
            >
              Search
            </button>
          </div>
        </div>
      </section>

      {/* Popular genes */}
      <section className="mb-10">
        <h2 className="mb-1 text-2xl font-bold text-text-primary">Popular Genes</h2>
        <p className="mb-6 text-sm text-text-muted">Quick access to commonly studied cytokines and secreted proteins</p>
        <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))' }}>
          {POPULAR_GENES.map(({ gene, cs, sa }) => (
            <Link
              key={gene}
              to={`/gene/${gene}`}
              className="group flex items-center justify-between rounded-xl border border-border-light bg-white px-5 py-4 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-lg"
            >
              <span className="text-base font-bold text-text-primary group-hover:text-primary">{gene}</span>
              <span className="flex gap-1">
                <Badge active={cs} type="cs">CS</Badge>
                <Badge active={sa} type="sa">SA</Badge>
              </span>
            </Link>
          ))}
        </div>
        <div className="mt-4 flex gap-5 text-xs text-text-muted">
          <span className="flex items-center gap-1.5">
            <Badge active type="cs">CS</Badge> CytoSig (43 cytokines)
          </span>
          <span className="flex items-center gap-1.5">
            <Badge active type="sa">SA</Badge> SecAct (1,170 proteins)
          </span>
        </div>
      </section>

      {/* What you'll see */}
      <section className="mb-10">
        <h2 className="mb-1 text-2xl font-bold text-text-primary">What You'll See</h2>
        <p className="mb-6 text-sm text-text-muted">Each gene detail page includes five interactive analysis tabs</p>
        <div className="grid gap-5" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          {TABS_INFO.map(({ icon, title, desc }) => (
            <div
              key={title}
              className="rounded-xl border border-border-light bg-white p-6 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-lg"
            >
              <div className="mb-3 text-[2rem]">{icon}</div>
              <h3 className="mb-1.5 text-sm font-bold text-text-primary">{title}</h3>
              <p className="text-sm leading-relaxed text-text-secondary">{desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function Badge({ active, children }: { active: boolean; type?: 'cs' | 'sa'; children: React.ReactNode }) {
  const colorClass =
    !active
      ? 'bg-bg-tertiary text-text-muted'
      : 'bg-primary/10 text-primary';

  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-bold leading-none ${colorClass}`}>
      {active ? children : '\u2014'}
    </span>
  );
}
