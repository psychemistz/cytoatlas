import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { CorrelationData } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { BarChart } from '@/components/charts/bar-chart';
import { CORRELATION_COLORSCALE } from '@/components/charts/chart-defaults';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

interface SeverityPanelProps {
  signatureType: string;
}

const SIGNIFICANCE_THRESHOLD = 0.05;

export default function SeverityPanel({ signatureType }: SeverityPanelProps) {
  const [disease, setDisease] = useState('all');

  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'severity', signatureType],
    queryFn: () =>
      get<CorrelationData[]>('/atlases/inflammation/correlations/severity', {
        signature_type: signatureType,
      }),
  });

  const diseaseOptions = useMemo(() => {
    if (!data) return [{ value: 'all', label: 'All Diseases' }];
    const diseases = [
      ...new Set(data.filter((d) => d.cell_type).map((d) => d.cell_type!)),
    ].sort();
    return [
      { value: 'all', label: 'All Diseases' },
      ...diseases.map((d) => ({ value: d, label: d })),
    ];
  }, [data]);

  const heatmap = useMemo(() => {
    if (!data) return null;

    const allDiseases = [
      ...new Set(data.filter((d) => d.cell_type).map((d) => d.cell_type!)),
    ].sort();
    const diseases = disease === 'all'
      ? allDiseases
      : allDiseases.filter((d) => d === disease);
    const signatures = [...new Set(data.map((d) => d.signature))].sort();

    const lookup = new Map(
      data
        .filter((d) => d.cell_type)
        .map((d) => [
          `${d.cell_type}||${d.signature}`,
          { rho: d.rho, p: d.p_value },
        ]),
    );

    // Null out non-significant values (set to null for display)
    const z = diseases.map((dis) =>
      signatures.map((sig) => {
        const entry = lookup.get(`${dis}||${sig}`);
        if (!entry) return 0;
        return entry.p <= SIGNIFICANCE_THRESHOLD ? entry.rho : 0;
      }),
    );

    return { z, x: signatures, y: diseases };
  }, [data, disease]);

  const topSignatures = useMemo(() => {
    if (!data) return null;

    const filtered = disease === 'all'
      ? data.filter((d) => d.cell_type && d.p_value <= SIGNIFICANCE_THRESHOLD)
      : data.filter((d) => d.cell_type === disease && d.p_value <= SIGNIFICANCE_THRESHOLD);

    if (filtered.length === 0) return null;

    // For "all" diseases, pick the entry with the largest |rho| per signature
    const bestPerSig = new Map<string, { rho: number; absRho: number }>();
    for (const d of filtered) {
      const absRho = Math.abs(d.rho);
      const existing = bestPerSig.get(d.signature);
      if (!existing || absRho > existing.absRho) {
        bestPerSig.set(d.signature, { rho: d.rho, absRho });
      }
    }

    const sorted = [...bestPerSig.entries()]
      .sort((a, b) => b[1].absRho - a[1].absRho)
      .slice(0, 20);

    // Reverse so largest bars appear at top in horizontal layout
    sorted.reverse();

    const categories = sorted.map(([sig]) => sig);
    const values = sorted.map(([, v]) => v.rho);
    const positiveValues = values.map((v) => (v >= 0 ? v : 0));
    const negativeValues = values.map((v) => (v < 0 ? v : 0));

    return { categories, positiveValues, negativeValues };
  }, [data, disease]);

  if (isLoading) return <Spinner message="Loading severity correlations..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load severity data: {(error as Error).message}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No severity correlation data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Disease"
          options={diseaseOptions}
          value={disease}
          onChange={setDisease}
        />
      </FilterBar>

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Disease Severity Correlation
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Spearman correlation between signature activity and disease severity.
            Non-significant correlations (p &gt; {SIGNIFICANCE_THRESHOLD}) are
            zeroed out.
          </p>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title="Severity Correlation (Spearman rho)"
            xTitle="Signature"
            yTitle="Disease"
            colorbarTitle="Spearman rho"
            colorscale={CORRELATION_COLORSCALE}
            symmetric
          />
        </div>
      )}

      {topSignatures && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top Signatures by |rho|{disease !== 'all' ? ` — ${disease}` : ''}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Top 20 signatures ranked by absolute Spearman correlation with
            disease severity. Red bars indicate positive correlation, blue bars
            indicate negative correlation.
          </p>
          <BarChart
            categories={topSignatures.categories}
            series={[
              { name: 'Positive rho', values: topSignatures.positiveValues },
              { name: 'Negative rho', values: topSignatures.negativeValues },
            ]}
            orientation="h"
            barmode="relative"
            xTitle="Spearman rho"
            yTitle="Signature"
            title={`Top Severity-Correlated Signatures${disease !== 'all' ? ` (${disease})` : ''}`}
            colors={['#b2182b', '#2166ac']}
            height={Math.max(400, topSignatures.categories.length * 25)}
          />
        </div>
      )}
    </div>
  );
}
