import { useState, useMemo } from 'react';
import type { OrganSignature } from '@/api/types/activity';
import {
  useScatlasOrganSignatures,
  useScatlasCancerSignatures,
} from '@/api/hooks/use-scatlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, ToggleGroup } from '@/components/ui/filter-bar';
import { HeatmapChart } from '@/components/charts/heatmap-chart';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

type ViewMode = 'normal' | 'cancer' | 'comparison';

const VIEW_OPTIONS = [
  { value: 'normal', label: 'Normal' },
  { value: 'cancer', label: 'Cancer' },
  { value: 'comparison', label: 'Comparison' },
];

interface TissueAtlasPanelProps {
  signatureType: string;
}

function buildHeatmap(data: OrganSignature[]) {
  const organs = [...new Set(data.map((d) => d.organ))].sort();
  const signatures = [...new Set(data.map((d) => d.signature))].sort();
  const lookup = new Map(
    data.map((d) => [`${d.organ}||${d.signature}`, d.mean_activity]),
  );
  const z = organs.map((org) =>
    signatures.map((sig) => lookup.get(`${org}||${sig}`) ?? 0),
  );
  return { z, x: signatures, y: organs };
}

function buildBoxplot(data: OrganSignature[]) {
  const organMap = new Map<string, number[]>();
  for (const d of data) {
    if (!organMap.has(d.organ)) organMap.set(d.organ, []);
    organMap.get(d.organ)!.push(d.mean_activity);
  }
  const sorted = [...organMap.entries()].sort((a, b) => {
    const medA = a[1].sort((x, y) => x - y)[Math.floor(a[1].length / 2)];
    const medB = b[1].sort((x, y) => x - y)[Math.floor(b[1].length / 2)];
    return medB - medA;
  });
  return {
    groups: sorted.map(([organ]) => organ),
    values: sorted.map(([, vals]) => vals),
  };
}

function buildDiffHeatmap(
  normalData: OrganSignature[],
  cancerData: OrganSignature[],
) {
  const normalSigs = new Set(normalData.map((d) => d.signature));
  const cancerSigs = new Set(cancerData.map((d) => d.signature));
  const signatures = [...normalSigs].filter((s) => cancerSigs.has(s)).sort();

  const normalLookup = new Map(
    normalData.map((d) => [`${d.organ}||${d.signature}`, d.mean_activity]),
  );
  const cancerLookup = new Map(
    cancerData.map((d) => [`${d.organ}||${d.signature}`, d.mean_activity]),
  );

  const normalOrgans = [...new Set(normalData.map((d) => d.organ))].sort();
  const cancerTypes = [...new Set(cancerData.map((d) => d.organ))].sort();
  const allRows = [
    ...normalOrgans.map((o) => `${o} (normal)`),
    ...cancerTypes.map((c) => `${c} (cancer)`),
  ];

  const z = [
    ...normalOrgans.map((org) =>
      signatures.map((sig) => normalLookup.get(`${org}||${sig}`) ?? 0),
    ),
    ...cancerTypes.map((ct) =>
      signatures.map((sig) => cancerLookup.get(`${ct}||${sig}`) ?? 0),
    ),
  ];

  return { z, x: signatures, y: allRows };
}

export default function TissueAtlasPanel({
  signatureType,
}: TissueAtlasPanelProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('normal');

  const normalQuery = useScatlasOrganSignatures(signatureType);
  const cancerQuery = useScatlasCancerSignatures(signatureType);

  const activeQuery = viewMode === 'cancer' ? cancerQuery : normalQuery;
  const isLoading =
    viewMode === 'comparison'
      ? normalQuery.isLoading || cancerQuery.isLoading
      : activeQuery.isLoading;
  const error =
    viewMode === 'comparison'
      ? normalQuery.error || cancerQuery.error
      : activeQuery.error;

  const heatmap = useMemo(() => {
    if (viewMode === 'comparison') {
      if (!normalQuery.data || !cancerQuery.data) return null;
      return buildDiffHeatmap(normalQuery.data, cancerQuery.data);
    }
    const data = viewMode === 'cancer' ? cancerQuery.data : normalQuery.data;
    if (!data) return null;
    return buildHeatmap(data);
  }, [viewMode, normalQuery.data, cancerQuery.data]);

  const boxplot = useMemo(() => {
    if (viewMode === 'comparison') return null;
    const data = viewMode === 'cancer' ? cancerQuery.data : normalQuery.data;
    if (!data) return null;
    return buildBoxplot(data);
  }, [viewMode, normalQuery.data, cancerQuery.data]);

  if (isLoading) return <Spinner message="Loading tissue atlas data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load tissue atlas data: {(error as Error).message}
      </div>
    );
  }

  const hasData =
    viewMode === 'comparison'
      ? normalQuery.data?.length && cancerQuery.data?.length
      : (viewMode === 'cancer' ? cancerQuery.data : normalQuery.data)?.length;

  if (!hasData) {
    return (
      <p className="py-8 text-center text-text-muted">
        No tissue atlas data available
      </p>
    );
  }

  const modeLabel =
    viewMode === 'normal'
      ? 'Normal Tissue'
      : viewMode === 'cancer'
        ? 'Cancer Type'
        : 'Normal vs Cancer Comparison';

  return (
    <div className="space-y-6">
      <FilterBar>
        <ToggleGroup
          options={VIEW_OPTIONS}
          value={viewMode}
          onChange={(v) => setViewMode(v as ViewMode)}
          label="View"
        />
      </FilterBar>

      {boxplot && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Activity Distribution by {viewMode === 'cancer' ? 'Cancer Type' : 'Organ'}
          </h3>
          <BoxplotChart
            groups={boxplot.groups}
            values={boxplot.values}
            title={`${modeLabel}: Activity Distribution`}
            yTitle="Mean Activity (z-score)"
            showPoints={false}
            height={Math.max(500, boxplot.groups.length * 30 + 150)}
          />
        </div>
      )}

      {heatmap && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {modeLabel} -- Signature Activity Heatmap
          </h3>
          <HeatmapChart
            z={heatmap.z}
            x={heatmap.x}
            y={heatmap.y}
            title={`${modeLabel}: Signature x ${viewMode === 'comparison' ? 'Tissue/Cancer' : viewMode === 'cancer' ? 'Cancer Type' : 'Organ'}`}
            xTitle="Signature"
            yTitle={viewMode === 'comparison' ? 'Tissue / Cancer Type' : viewMode === 'cancer' ? 'Cancer Type' : 'Organ'}
            colorbarTitle="Mean Activity"
            symmetric
          />
        </div>
      )}
    </div>
  );
}
