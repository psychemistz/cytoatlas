import { useState, useMemo } from 'react';
import {
  useConservedSignatures,
  useConsistencyHeatmap,
  useSignatureReliability,
} from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface ConservedTabProps {
  signatureType: string;
}

function correlationColor(r: number | null): string {
  if (r === null || r === undefined) return 'text-text-muted';
  if (r > 0.7) return 'text-green-600';
  if (r >= 0.4) return 'text-blue-600';
  if (r >= 0.2) return 'text-yellow-600';
  return 'text-red-600';
}

export default function ConservedTab({ signatureType }: ConservedTabProps) {
  const [minAtlases, setMinAtlases] = useState(2);
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');

  const { data: conserved, isLoading: consLoading } = useConservedSignatures(
    signatureType,
    minAtlases,
  );
  const { data: heatmap, isLoading: hmLoading } = useConsistencyHeatmap(signatureType);
  const { data: reliability, isLoading: relLoading } = useSignatureReliability(signatureType);

  const barData = useMemo(() => {
    if (!conserved) return null;
    const sorted = [...conserved]
      .sort((a, b) => b.conservation_score - a.conservation_score)
      .slice(0, 30);
    return {
      categories: sorted.map((s) => s.signature),
      values: sorted.map((s) => s.conservation_score),
    };
  }, [conserved]);

  const categorySummary = useMemo(() => {
    if (!reliability?.signatures) return null;
    let high = 0, moderate = 0, atlasSpecific = 0;
    for (const s of reliability.signatures) {
      const r = s.mean_correlation ?? 0;
      if (r > 0.7) high++;
      else if (r >= 0.4) moderate++;
      else atlasSpecific++;
    }
    return { high, moderate, atlasSpecific, total: reliability.signatures.length };
  }, [reliability]);

  const filteredRows = useMemo(() => {
    if (!reliability?.signatures) return [];
    const q = search.toLowerCase();
    return reliability.signatures.filter((row) => {
      if (q && !row.signature.toLowerCase().includes(q)) return false;
      if (category === 'high' && (row.mean_correlation ?? 0) <= 0.7) return false;
      if (category === 'moderate' && ((row.mean_correlation ?? 0) <= 0.4 || (row.mean_correlation ?? 0) > 0.7)) return false;
      if (category === 'specific' && (row.mean_correlation ?? 0) >= 0.4) return false;
      return true;
    });
  }, [reliability, search, category]);

  if (consLoading || hmLoading) return <Spinner message="Loading conserved signatures..." />;

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Min Atlases"
          options={[
            { value: '2', label: '2+ Atlases' },
            { value: '3', label: '3 Atlases (all)' },
          ]}
          value={String(minAtlases)}
          onChange={(v) => setMinAtlases(Number(v))}
        />
        <SelectFilter
          label="Conservation"
          options={[
            { value: 'all', label: 'All Signatures' },
            { value: 'high', label: 'Highly Conserved (r > 0.7)' },
            { value: 'moderate', label: 'Moderately Conserved (r > 0.4)' },
            { value: 'specific', label: 'Atlas-Specific (r < 0.4)' },
          ]}
          value={category}
          onChange={setCategory}
        />
      </FilterBar>

      {categorySummary && (
        <div className="grid grid-cols-4 gap-4">
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-green-600">{categorySummary.high}</div>
            <div className="text-xs text-text-muted">Highly Conserved</div>
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-blue-600">{categorySummary.moderate}</div>
            <div className="text-xs text-text-muted">Moderately Conserved</div>
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-yellow-600">{categorySummary.atlasSpecific}</div>
            <div className="text-xs text-text-muted">Atlas-Specific</div>
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-text-primary">{categorySummary.total}</div>
            <div className="text-xs text-text-muted">Total Signatures</div>
          </div>
        </div>
      )}

      {barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Conserved Signatures
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Signatures with consistent activity patterns across {minAtlases}+ atlases.
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Conservation Score"
            yTitle="Signature"
            title="Conserved Signatures"
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Activity Consistency Heatmap
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Signature Consistency Across Atlases"
            xTitle="Atlas"
            yTitle="Signature"
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}

      {relLoading && <Spinner message="Loading signature reliability..." />}

      {reliability && (
        <div className="mt-6">
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">Signature Details</h3>
          <input
            type="text"
            placeholder="Search signatures..."
            className="mb-3 w-full rounded-md border border-border-light bg-bg-primary px-3 py-2 text-sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="max-h-[500px] overflow-auto rounded-md border border-border-light">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Signature</th>
                  <th className="px-3 py-2 text-left font-medium">Category</th>
                  <th className="px-3 py-2 text-right font-medium">Mean r</th>
                  <th className="px-3 py-2 text-right font-medium">CIMA-Inflam</th>
                  <th className="px-3 py-2 text-right font-medium">CIMA-scAtlas</th>
                  <th className="px-3 py-2 text-right font-medium">Inflam-scAtlas</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => {
                  const cimaInflam = row.correlations?.cima_vs_inflammation?.r ?? null;
                  const cimaScatlas = row.correlations?.cima_vs_scatlas?.r ?? null;
                  const inflamScatlas = row.correlations?.inflammation_vs_scatlas?.r ?? null;
                  return (
                    <tr
                      key={row.signature}
                      className="border-t border-border-light hover:bg-bg-tertiary"
                    >
                      <td className="px-3 py-2 font-medium">{row.signature}</td>
                      <td className="px-3 py-2 text-text-muted">{row.category ?? '—'}</td>
                      <td className={`px-3 py-2 text-right font-mono ${correlationColor(row.mean_correlation)}`}>
                        {row.mean_correlation != null ? row.mean_correlation.toFixed(3) : '—'}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${correlationColor(cimaInflam)}`}>
                        {cimaInflam != null ? cimaInflam.toFixed(3) : '—'}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${correlationColor(cimaScatlas)}`}>
                        {cimaScatlas != null ? cimaScatlas.toFixed(3) : '—'}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${correlationColor(inflamScatlas)}`}>
                        {inflamScatlas != null ? inflamScatlas.toFixed(3) : '—'}
                      </td>
                    </tr>
                  );
                })}
                {filteredRows.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-4 text-center text-text-muted">
                      {search ? 'No signatures match your search.' : 'No signature data available.'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
