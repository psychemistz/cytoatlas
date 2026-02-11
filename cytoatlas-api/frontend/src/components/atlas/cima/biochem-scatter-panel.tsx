import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { useCimaBiochemistry, useCimaSignatures } from '@/api/hooks/use-cima';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

interface BiochemScatterPanelProps {
  signatureType: string;
}

interface BiochemSample {
  marker_value: number;
  activity_value: number;
  sample_id: string;
  sex?: string;
  age_bin?: string;
  bmi_bin?: string;
}

const COLOR_OPTIONS = [
  { value: 'none', label: 'None' },
  { value: 'sex', label: 'Sex' },
  { value: 'age_bin', label: 'Age group' },
  { value: 'bmi_bin', label: 'BMI group' },
];

const COLOR_MAP: Record<string, string> = {
  Male: '#3b82f6',
  Female: '#ef4444',
  // Fallback palette for categorical bins
  _0: '#3b82f6',
  _1: '#ef4444',
  _2: '#22c55e',
  _3: '#f59e0b',
  _4: '#8b5cf6',
  _5: '#06b6d4',
};

function assignColors(data: BiochemSample[], colorBy: string): string[] {
  if (colorBy === 'none') return data.map(() => '#3b82f6');

  const values = data.map((d) => {
    if (colorBy === 'sex') return d.sex ?? 'Unknown';
    if (colorBy === 'age_bin') return d.age_bin ?? 'Unknown';
    if (colorBy === 'bmi_bin') return d.bmi_bin ?? 'Unknown';
    return 'Unknown';
  });

  const unique = [...new Set(values)].sort();
  const palette = new Map<string, string>();
  unique.forEach((v, i) => {
    palette.set(v, COLOR_MAP[v] ?? COLOR_MAP[`_${i}`] ?? '#6b7280');
  });

  return values.map((v) => palette.get(v) ?? '#6b7280');
}

function computeCorrelation(x: number[], y: number[]): { r: number; p: number } {
  const n = x.length;
  if (n < 3) return { r: 0, p: 1 };
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dxi = x[i] - mx;
    const dyi = y[i] - my;
    num += dxi * dyi;
    dx2 += dxi * dxi;
    dy2 += dyi * dyi;
  }
  const r = dx2 > 0 && dy2 > 0 ? num / Math.sqrt(dx2 * dy2) : 0;
  const t = r * Math.sqrt((n - 2) / (1 - r * r + 1e-15));
  const p = Math.min(1, 2 * Math.exp(-0.717 * Math.abs(t) - 0.416 * t * t / n));
  return { r, p };
}

export default function BiochemScatterPanel({ signatureType }: BiochemScatterPanelProps) {
  const biochemQuery = useCimaBiochemistry(signatureType);
  const sigQuery = useCimaSignatures(signatureType);

  const markers = useMemo(() => {
    if (!biochemQuery.data) return [];
    return [...new Set(biochemQuery.data.map((d) => d.marker))].sort();
  }, [biochemQuery.data]);

  const signatures = useMemo(() => {
    if (!sigQuery.data) return [];
    return sigQuery.data;
  }, [sigQuery.data]);

  const [marker, setMarker] = useState('');
  const [signature, setSignature] = useState('');
  const [colorBy, setColorBy] = useState('none');

  // Set defaults when data arrives
  useMemo(() => {
    if (markers.length > 0 && !marker) setMarker(markers[0]);
  }, [markers, marker]);

  useMemo(() => {
    if (signatures.length > 0 && !signature) setSignature(signatures[0]);
  }, [signatures, signature]);

  const { data: scatterData, isLoading: scatterLoading, error: scatterError } = useQuery({
    queryKey: ['cima', 'biochem-scatter', signatureType, marker, signature],
    queryFn: () =>
      get<BiochemSample[]>('/atlases/cima/scatter/biochem-samples', {
        signature_type: signatureType,
        marker,
        signature,
      }),
    enabled: !!marker && !!signature,
  });

  const chartData = useMemo(() => {
    if (!scatterData || scatterData.length === 0) return null;
    const x = scatterData.map((d) => d.marker_value);
    const y = scatterData.map((d) => d.activity_value);
    const labels = scatterData.map((d) => d.sample_id);
    const colors = assignColors(scatterData, colorBy);
    const stats = computeCorrelation(x, y);
    return { x, y, labels, colors, stats };
  }, [scatterData, colorBy]);

  const isLoading = biochemQuery.isLoading || sigQuery.isLoading;

  if (isLoading) return <Spinner message="Loading biochemistry scatter data..." />;

  if (biochemQuery.error || sigQuery.error) {
    const err = biochemQuery.error || sigQuery.error;
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load data: {(err as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-1 text-sm font-semibold text-text-secondary">
          Biochemistry-Activity Sample Scatter
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Per-sample correlation between biochemistry marker level and cytokine activity.
        </p>
      </div>

      <FilterBar>
        <SelectFilter
          label="X-axis marker"
          options={markers.map((m) => ({ value: m, label: m }))}
          value={marker}
          onChange={setMarker}
        />
        <SelectFilter
          label="Y-axis signature"
          options={signatures.map((s) => ({ value: s, label: s }))}
          value={signature}
          onChange={setSignature}
        />
        <SelectFilter
          label="Color by"
          options={COLOR_OPTIONS}
          value={colorBy}
          onChange={setColorBy}
        />
      </FilterBar>

      {scatterLoading && <Spinner message="Loading scatter data..." />}

      {scatterError && (
        <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Failed to load scatter data: {(scatterError as Error).message}
        </div>
      )}

      {chartData && chartData.x.length > 0 && (
        <ScatterChart
          x={chartData.x}
          y={chartData.y}
          labels={chartData.labels}
          colors={chartData.colors}
          xTitle={marker}
          yTitle={`${signature} Activity`}
          title={`${marker} vs ${signature}`}
          showTrendLine
          stats={chartData.stats}
        />
      )}

      {chartData && chartData.x.length === 0 && (
        <p className="py-8 text-center text-text-muted">
          No sample-level data for this marker/signature combination
        </p>
      )}
    </div>
  );
}
