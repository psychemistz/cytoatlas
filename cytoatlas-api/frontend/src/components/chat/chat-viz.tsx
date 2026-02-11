import { PlotlyChart } from '@/components/charts/plotly-chart';
import type { ChatVisualization } from '@/api/types/chat';
import type { Data, Layout } from 'plotly.js-dist-min';

interface ChatVizProps {
  visualization: ChatVisualization;
}

export function ChatViz({ visualization }: ChatVizProps) {
  const hasPlotlyData =
    visualization.data && Array.isArray(visualization.data) && visualization.data.length > 0;

  if (hasPlotlyData) {
    return (
      <div className="my-3 overflow-hidden rounded-lg border border-border-light bg-bg-primary">
        <PlotlyChart
          data={visualization.data as Data[]}
          layout={
            {
              ...(visualization.layout as Partial<Layout>),
              autosize: true,
              height: 400,
              margin: { t: 40, r: 20, b: 50, l: 60 },
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
