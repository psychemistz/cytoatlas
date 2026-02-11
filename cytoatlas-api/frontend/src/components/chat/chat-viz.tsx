import { PlotlyChart } from '@/components/charts/plotly-chart';
import type { ChatVisualization } from '@/api/types/chat';
import type { Data, Layout } from 'plotly.js-dist-min';

interface ChatVizProps {
  visualization: ChatVisualization;
}

interface CytoAtlasVizData {
  labels?: string[];
  values?: number[];
  x_labels?: string[];
  y_labels?: string[];
  headers?: string[];
  rows?: unknown[][];
}

interface CytoAtlasVizConfig {
  x_label?: string;
  y_label?: string;
}

function convertToPlotly(viz: ChatVisualization): { data: Data[]; layout: Partial<Layout> } | null {
  const vizData = viz.data as CytoAtlasVizData | undefined;
  const vizConfig = (viz as { config?: CytoAtlasVizConfig }).config;

  if (!vizData) return null;

  const titleObj = viz.title ? { text: viz.title } : undefined;

  if (viz.type === 'bar_chart' && vizData.labels && vizData.values) {
    const colors = vizData.values.map((v) => (v >= 0 ? '#3b82f6' : '#ef4444'));
    return {
      data: [
        {
          type: 'bar',
          x: vizData.labels,
          y: vizData.values,
          marker: { color: colors },
        } as Data,
      ],
      layout: {
        title: titleObj,
        xaxis: { title: vizConfig?.x_label ? { text: vizConfig.x_label } : undefined, tickangle: -45 },
        yaxis: { title: vizConfig?.y_label ? { text: vizConfig.y_label } : undefined },
      },
    };
  }

  if (viz.type === 'heatmap' && vizData.x_labels && vizData.y_labels && vizData.values) {
    return {
      data: [
        {
          type: 'heatmap',
          x: vizData.x_labels,
          y: vizData.y_labels,
          z: vizData.values,
          colorscale: 'RdBu',
          reversescale: true,
        } as Data,
      ],
      layout: {
        title: titleObj,
      },
    };
  }

  if (viz.type === 'table' && vizData.headers && vizData.rows) {
    const headerValues = vizData.headers;
    const cellValues = headerValues.map((_, colIdx) =>
      vizData.rows!.map((row) => String(row[colIdx] ?? '')),
    );
    return {
      data: [
        {
          type: 'table',
          header: { values: headerValues, fill: { color: '#3b82f6' }, font: { color: 'white' } },
          cells: { values: cellValues, fill: { color: ['#f8fafc', '#f1f5f9'] } },
        } as Data,
      ],
      layout: {
        title: titleObj,
      },
    };
  }

  return null;
}

export function ChatViz({ visualization }: ChatVizProps) {
  // Check for native Plotly format
  const hasPlotlyData =
    visualization.data && Array.isArray(visualization.data) && visualization.data.length > 0;

  // Try converting CytoAtlas format to Plotly
  const converted = hasPlotlyData ? null : convertToPlotly(visualization);

  const plotData = hasPlotlyData
    ? (visualization.data as Data[])
    : converted?.data;
  const plotLayout = hasPlotlyData
    ? (visualization.layout as Partial<Layout>)
    : converted?.layout;

  if (plotData) {
    return (
      <div className="my-3 overflow-hidden rounded-lg border border-border-light bg-bg-primary">
        <PlotlyChart
          data={plotData}
          layout={
            {
              ...plotLayout,
              autosize: true,
              height: 400,
              margin: { t: 40, r: 20, b: 80, l: 60 },
            } as Partial<Layout>
          }
          style={{ height: 400 }}
          id={visualization.container_id}
        />
      </div>
    );
  }

  return (
    <div className="my-3 flex items-center justify-center rounded-lg border border-border-light bg-bg-tertiary p-8 text-sm text-text-muted">
      <div className="text-center">
        <p className="font-medium">Visualization</p>
        <p className="mt-1 text-xs">{visualization.container_id}</p>
        {visualization.type && (
          <p className="mt-1 text-xs text-text-muted">Type: {visualization.type}</p>
        )}
      </div>
    </div>
  );
}
