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
  { title: 'Gene Expression', desc: 'Expression levels by cell type across atlases' },
  { title: 'CytoSig Activity', desc: 'Cytokine signaling activity (44 cytokines)' },
  { title: 'SecAct Activity', desc: 'Secreted protein activity (1,170 proteins)' },
  { title: 'Disease Associations', desc: 'Differential activity in diseases vs healthy' },
  { title: 'Correlations', desc: 'Age, BMI, and biochemistry correlations' },
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
    <div className="mx-auto max-w-[1000px] px-4 py-12">
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-3xl font-bold">Search</h1>
        <p className="text-text-secondary">
          Enter a gene symbol to view expression, cytokine/protein activity, cell type specificity, disease
          associations, and organ/tissue patterns.
        </p>
      </div>

      {/* Search bar */}
      <div className="mx-auto mb-8 flex max-w-lg gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter gene symbol (e.g., IFNG, TNF, IL6...)"
          autoFocus
          className="flex-1 rounded-lg border border-border-light px-4 py-3 text-base outline-none focus:border-primary"
        />
        <button
          onClick={handleSearch}
          className="rounded-lg bg-primary px-6 py-3 font-medium text-text-inverse hover:bg-primary-dark"
        >
          Search
        </button>
      </div>

      {/* Popular genes */}
      <div className="mb-8">
        <p className="mb-3 text-sm font-medium text-text-secondary">Popular genes:</p>
        <div className="flex flex-wrap gap-2">
          {POPULAR_GENES.map(({ gene, cs, sa }) => (
            <Link
              key={gene}
              to={`/gene/${gene}`}
              className="flex items-center gap-1.5 rounded-lg border border-border-light px-3 py-2 text-sm font-medium hover:bg-bg-tertiary"
            >
              {gene}
              <span className="flex gap-0.5">
                <Badge active={cs} type="cs">CS</Badge>
                <Badge active={sa} type="sa">SA</Badge>
              </span>
            </Link>
          ))}
        </div>
        <div className="mt-3 flex gap-4 text-xs text-text-muted">
          <span className="flex items-center gap-1">
            <Badge active type="cs">CS</Badge> CytoSig (43 cytokines)
          </span>
          <span className="flex items-center gap-1">
            <Badge active type="sa">SA</Badge> SecAct (1,170 proteins)
          </span>
        </div>
      </div>

      {/* Info card */}
      <div className="rounded-xl border border-border-light p-6">
        <h3 className="mb-3 font-semibold">What you'll see</h3>
        <ul className="space-y-2">
          {TABS_INFO.map(({ title, desc }) => (
            <li key={title} className="text-sm text-text-secondary">
              <strong className="text-text-primary">{title}</strong> &mdash; {desc}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Badge({ active, type = 'cs', children }: { active: boolean; type?: 'cs' | 'sa'; children: React.ReactNode }) {
  const colorClass =
    !active
      ? 'bg-bg-tertiary text-text-muted'
      : type === 'sa'
        ? 'bg-purple-500/20 text-purple-600'
        : 'bg-primary/10 text-primary';

  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-bold leading-none ${colorClass}`}>
      {active ? children : '\u2014'}
    </span>
  );
}
