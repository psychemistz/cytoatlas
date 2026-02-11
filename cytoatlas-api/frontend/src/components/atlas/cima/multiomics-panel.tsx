import { useState, useMemo } from 'react';
import { useCimaBiochemistry, useCimaMetabolites } from '@/api/hooks/use-cima';
import type { BiochemCorrelation, MetaboliteCorrelation } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter, ToggleGroup } from '@/components/ui/filter-bar';
import { BarChart } from '@/components/charts/bar-chart';

interface MultiomicsPanelProps {
  signatureType: string;
}

const SUBSET_OPTIONS = [
  { value: 'all', label: 'All Signatures' },
  { value: 'inflammatory', label: 'Inflammatory' },
  { value: 'regulatory', label: 'Regulatory' },
  { value: 'th17', label: 'Th17 Axis' },
];

const SUBSET_SIGNATURES: Record<string, string[]> = {
  inflammatory: ['IL-1B', 'IL-6', 'TNF', 'IL-1A', 'IL-18', 'IL1B', 'IL6', 'TNFA', 'IL1A', 'IL18'],
  regulatory: ['IL-10', 'TGF-B', 'IL-4', 'IL-13', 'IL-35', 'IL10', 'TGFB1', 'IL4', 'IL13'],
  th17: ['IL-17A', 'IL-17F', 'IL-21', 'IL-22', 'IL-23A', 'IL17A', 'IL17F', 'IL21', 'IL22', 'IL23A'],
};

const THRESHOLD_OPTIONS = [
  { value: '0.2', label: '|\u03C1| \u2265 0.2' },
  { value: '0.3', label: '|\u03C1| \u2265 0.3' },
  { value: '0.4', label: '|\u03C1| \u2265 0.4' },
  { value: '0.5', label: '|\u03C1| \u2265 0.5' },
];

function matchesSubset(signature: string, subset: string): boolean {
  if (subset === 'all') return true;
  const subsetSigs = SUBSET_SIGNATURES[subset];
  if (!subsetSigs) return true;
  const sigLower = signature.toLowerCase();
  return subsetSigs.some((s) => sigLower.includes(s.toLowerCase()));
}

export default function MultiomicsPanel({ signatureType }: MultiomicsPanelProps) {
  const [viewMode, setViewMode] = useState<'bar' | 'table'>('bar');
  const [subset, setSubset] = useState('all');
  const [threshold, setThreshold] = useState('0.4');

  const biochemQuery = useCimaBiochemistry(signatureType);
  const metabQuery = useCimaMetabolites(signatureType);

  const isLoading = biochemQuery.isLoading || metabQuery.isLoading;
  const error = biochemQuery.error || metabQuery.error;

  const thresholdNum = parseFloat(threshold);

  // Filter biochemistry data
  const filteredBiochem = useMemo(() => {
    if (!biochemQuery.data) return [];
    return biochemQuery.data.filter(
      (d) => Math.abs(d.rho) >= thresholdNum && matchesSubset(d.signature, subset),
    );
  }, [biochemQuery.data, thresholdNum, subset]);

  // Filter metabolite data
  const filteredMetab = useMemo(() => {
    if (!metabQuery.data) return [];
    return metabQuery.data.filter(
      (d) => Math.abs(d.rho) >= thresholdNum && matchesSubset(d.signature, subset),
    );
  }, [metabQuery.data, thresholdNum, subset]);

  // Combined and sorted for bar chart (top 30)
  const barData = useMemo(() => {
    const combined: { label: string; rho: number; type: string }[] = [
      ...filteredBiochem.map((d) => ({
        label: `${d.signature} \u00D7 ${d.marker}`,
        rho: d.rho,
        type: 'Biochemistry',
      })),
      ...filteredMetab.map((d) => ({
        label: `${d.signature} \u00D7 ${d.metabolite}`,
        rho: d.rho,
        type: 'Metabolite',
      })),
    ];
    const sorted = combined.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 30);
    if (sorted.length === 0) return null;
    // Reverse so highest absolute rho appears at top in horizontal layout
    sorted.reverse();
    return {
      categories: sorted.map((d) => d.label),
      values: sorted.map((d) => d.rho),
      colors: sorted.map((d) => (d.rho >= 0 ? '#b2182b' : '#2166ac')),
    };
  }, [filteredBiochem, filteredMetab]);

  // Top 5 tables
  const topBiochem = useMemo(
    () =>
      [...filteredBiochem].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 5),
    [filteredBiochem],
  );
  const topMetab = useMemo(
    () =>
      [...filteredMetab].sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho)).slice(0, 5),
    [filteredMetab],
  );

  // Summary counts
  const summary = useMemo(() => {
    const allFiltered = [
      ...filteredBiochem.map((d) => d.signature),
      ...filteredMetab.map((d) => d.signature),
    ];
    const cytokines = new Set(allFiltered);
    const biochemMarkers = new Set(filteredBiochem.map((d) => d.marker));
    const metabolites = new Set(filteredMetab.map((d) => d.metabolite));
    return {
      cytokineCount: cytokines.size,
      biochemCount: biochemMarkers.size,
      metaboliteCount: metabolites.size,
      biochemEdges: filteredBiochem.length,
      metabEdges: filteredMetab.length,
      totalEdges: filteredBiochem.length + filteredMetab.length,
    };
  }, [filteredBiochem, filteredMetab]);

  if (isLoading) return <Spinner message="Loading multi-omics data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load multi-omics data: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <ToggleGroup
          options={[
            { value: 'bar', label: 'Bar Chart' },
            { value: 'table', label: 'Top Correlations' },
          ]}
          value={viewMode}
          onChange={(v) => setViewMode(v as 'bar' | 'table')}
          label="View"
        />
        <SelectFilter
          label="Subset"
          options={SUBSET_OPTIONS}
          value={subset}
          onChange={setSubset}
        />
        <SelectFilter
          label="Threshold"
          options={THRESHOLD_OPTIONS}
          value={threshold}
          onChange={setThreshold}
        />
      </FilterBar>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4 sm:grid-cols-6">
        <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
          <div className="text-lg font-bold text-primary">{summary.cytokineCount}</div>
          <div className="text-xs text-text-muted">Cytokines</div>
        </div>
        <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
          <div className="text-lg font-bold text-green-600">{summary.biochemCount}</div>
          <div className="text-xs text-text-muted">Biochem Markers</div>
        </div>
        <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
          <div className="text-lg font-bold text-orange-500">{summary.metaboliteCount}</div>
          <div className="text-xs text-text-muted">Metabolites</div>
        </div>
        <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
          <div className="text-lg font-bold text-text-secondary">{summary.biochemEdges}</div>
          <div className="text-xs text-text-muted">Biochem Edges</div>
        </div>
        <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
          <div className="text-lg font-bold text-text-secondary">{summary.metabEdges}</div>
          <div className="text-xs text-text-muted">Metab Edges</div>
        </div>
        <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
          <div className="text-lg font-bold text-text-primary">{summary.totalEdges}</div>
          <div className="text-xs text-text-muted">Total Edges</div>
        </div>
      </div>

      {/* Bar Chart View */}
      {viewMode === 'bar' && barData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Cross-Omic Correlations
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Spearman correlations between cytokine activity and biochemistry/metabolite markers.
            Red bars = positive correlation, blue bars = negative.
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman \u03C1"
            yTitle="Cytokine \u00D7 Marker"
            title="Top Cross-Omic Correlations"
            colors={barData.colors}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {/* Table View */}
      {viewMode === 'table' && (
        <div className="grid gap-6 md:grid-cols-2">
          <CorrelationTable
            title="Top Biochemistry Correlations"
            variableLabel="Marker"
            items={topBiochem.map((d) => ({
              signature: d.signature,
              variable: d.marker,
              rho: d.rho,
            }))}
          />
          <CorrelationTable
            title="Top Metabolite Correlations"
            variableLabel="Metabolite"
            items={topMetab.map((d) => ({
              signature: d.signature,
              variable: d.metabolite,
              rho: d.rho,
            }))}
          />
        </div>
      )}

      {/* No data message */}
      {viewMode === 'bar' && !barData && filteredBiochem.length === 0 && filteredMetab.length === 0 && (
        <p className="py-8 text-center text-text-muted">
          No correlations found with current filters (threshold: |\u03C1| &ge; {threshold})
        </p>
      )}

      {/* Footer note */}
      <p className="text-xs text-text-muted">
        Showing correlations with |\u03C1| &ge; {threshold}.
        {subset !== 'all'
          ? ` Filtered to ${SUBSET_OPTIONS.find((o) => o.value === subset)?.label} signatures.`
          : ''}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Reusable correlation table sub-component                           */
/* ------------------------------------------------------------------ */

interface CorrelationTableProps {
  title: string;
  variableLabel: string;
  items: { signature: string; variable: string; rho: number }[];
}

function CorrelationTable({ title, variableLabel, items }: CorrelationTableProps) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-text-secondary">{title}</h3>
      <div className="overflow-x-auto rounded-lg border border-border-light">
        <table className="w-full text-left text-sm">
          <thead className="bg-bg-secondary">
            <tr>
              <th className="px-3 py-2 font-medium">Cytokine</th>
              <th className="px-3 py-2 font-medium">{variableLabel}</th>
              <th className="px-3 py-2 text-right font-medium">\u03C1</th>
            </tr>
          </thead>
          <tbody>
            {items.map((d, i) => (
              <tr key={i} className="border-t border-border-light hover:bg-bg-tertiary">
                <td className="px-3 py-1.5">{d.signature}</td>
                <td className="px-3 py-1.5">{d.variable}</td>
                <td
                  className={`px-3 py-1.5 text-right font-mono ${
                    d.rho >= 0 ? 'text-red-600' : 'text-blue-600'
                  }`}
                >
                  {d.rho.toFixed(4)}
                </td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr>
                <td colSpan={3} className="px-3 py-4 text-center text-text-muted">
                  No correlations above threshold
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
