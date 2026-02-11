import { useState, useMemo } from 'react';
import type { Data, Layout } from 'plotly.js-dist-min';
import { useSpatialDatasets, useSpatialCoordinates } from '@/api/hooks/use-spatial';
import { PlotlyChart } from '@/components/charts/plotly-chart';
import { HEATMAP_COLORSCALE, COLORS } from '@/components/charts/chart-defaults';
import { Spinner } from '@/components/ui/loading-skeleton';

interface SpatialMapTabProps {
  signatureType: string;
}

type ColorMode = 'activity' | 'cell_type';

const CATEGORICAL_COLORS = [
  COLORS.primary,
  COLORS.green,
  COLORS.amber,
  COLORS.purple,
  COLORS.red,
  COLORS.gray,
  COLORS.darkSlate,
  '#06b6d4',
  '#ec4899',
  '#84cc16',
];

export default function SpatialMapTab({ signatureType }: SpatialMapTabProps) {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [colorMode, setColorMode] = useState<ColorMode>('activity');

  const { data: datasets, isLoading: datasetsLoading } = useSpatialDatasets();
  const { data: coordinates, isLoading: coordsLoading, error: coordsError } =
    useSpatialCoordinates(selectedDataset, signatureType);

  const { traces, layout } = useMemo(() => {
    if (!coordinates || coordinates.length === 0) {
      return { traces: [] as Data[], layout: {} as Partial<Layout> };
    }

    let plotTraces: Data[];

    if (colorMode === 'activity') {
      const activities = coordinates.map((c) => c.activity ?? 0);
      plotTraces = [
        {
          type: 'scattergl' as const,
          mode: 'markers' as const,
          x: coordinates.map((c) => c.x_coord),
          y: coordinates.map((c) => c.y_coord),
          marker: {
            color: activities,
            colorscale: HEATMAP_COLORSCALE as unknown as string,
            size: 3,
            opacity: 0.8,
            colorbar: {
              title: { text: 'Activity', side: 'right' as const },
              len: 0.9,
            },
          },
          hovertemplate:
            'x: %{x:.1f}<br>y: %{y:.1f}<br>Activity: %{marker.color:.3f}<extra></extra>',
        },
      ];
    } else {
      // Group by cell type for categorical coloring
      const cellTypeGroups = new Map<string, { x: number[]; y: number[] }>();
      for (const c of coordinates) {
        const ct = c.cell_type ?? 'Unknown';
        const group = cellTypeGroups.get(ct) ?? { x: [], y: [] };
        group.x.push(c.x_coord);
        group.y.push(c.y_coord);
        cellTypeGroups.set(ct, group);
      }

      plotTraces = Array.from(cellTypeGroups.entries()).map(
        ([cellType, group], i) => ({
          type: 'scattergl' as const,
          mode: 'markers' as const,
          name: cellType,
          x: group.x,
          y: group.y,
          marker: {
            color: CATEGORICAL_COLORS[i % CATEGORICAL_COLORS.length],
            size: 3,
            opacity: 0.8,
          },
          hovertemplate: `${cellType}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>`,
        }),
      );
    }

    const plotLayout: Partial<Layout> = {
      xaxis: {
        title: { text: 'X Coordinate' },
        scaleanchor: 'y' as const,
      },
      yaxis: {
        title: { text: 'Y Coordinate' },
        autorange: 'reversed' as const,
      },
      height: 600,
      showlegend: colorMode === 'cell_type',
    };

    return { traces: plotTraces, layout: plotLayout };
  }, [coordinates, colorMode]);

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Spatial Activity Map
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Visualize activity z-scores or cell type annotations mapped onto spatial
          coordinates for a selected dataset.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-border-light bg-bg-secondary p-3">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-text-secondary">Dataset</label>
          {datasetsLoading ? (
            <span className="text-sm text-text-muted">Loading...</span>
          ) : (
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="rounded-md border border-border-light bg-bg-primary px-3 py-1.5 text-sm outline-none focus:border-primary"
            >
              <option value="">Select a dataset</option>
              {(datasets ?? []).map((ds) => (
                <option key={ds.dataset_id} value={ds.dataset_id}>
                  {ds.filename ?? ds.dataset_id} ({ds.technology} - {ds.tissue})
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-text-secondary">Color by</label>
          <div className="flex rounded-md border border-border-light bg-bg-primary">
            <button
              onClick={() => setColorMode('activity')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors first:rounded-l-md last:rounded-r-md ${
                colorMode === 'activity'
                  ? 'bg-primary text-text-inverse'
                  : 'text-text-secondary hover:bg-bg-tertiary'
              }`}
            >
              Activity
            </button>
            <button
              onClick={() => setColorMode('cell_type')}
              className={`px-3 py-1.5 text-sm font-medium transition-colors first:rounded-l-md last:rounded-r-md ${
                colorMode === 'cell_type'
                  ? 'bg-primary text-text-inverse'
                  : 'text-text-secondary hover:bg-bg-tertiary'
              }`}
            >
              Cell Type
            </button>
          </div>
        </div>
      </div>

      {/* Map display */}
      {!selectedDataset && (
        <p className="py-12 text-center text-text-muted">
          Select a dataset above to visualize spatial coordinates
        </p>
      )}

      {selectedDataset && coordsLoading && (
        <Spinner message="Loading spatial coordinates..." />
      )}

      {selectedDataset && coordsError && (
        <p className="py-8 text-center text-red-600">
          Failed to load coordinates: {(coordsError as Error).message}
        </p>
      )}

      {selectedDataset && !coordsLoading && !coordsError && traces.length > 0 && (
        <PlotlyChart data={traces} layout={layout} />
      )}

      {selectedDataset && !coordsLoading && !coordsError && traces.length === 0 && (
        <p className="py-8 text-center text-text-muted">
          No coordinate data available for this dataset
        </p>
      )}
    </div>
  );
}
