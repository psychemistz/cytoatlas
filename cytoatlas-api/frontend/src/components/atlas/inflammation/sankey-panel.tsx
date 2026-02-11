import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import { Spinner } from '@/components/ui/loading-skeleton';
import { SankeyChart } from '@/components/charts/sankey-chart';

interface SankeyNode {
  name: string;
  category?: string;
}

interface SankeyLink {
  source: number;
  target: number;
  value: number;
}

interface DiseaseFlowData {
  nodes: SankeyNode[];
  links: SankeyLink[];
}

export default function SankeyPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['inflammation', 'disease-flow'],
    queryFn: () => get<DiseaseFlowData>('/atlases/inflammation/disease-flow'),
  });

  if (isLoading) return <Spinner message="Loading disease flow data..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load disease flow data: {(error as Error).message}
      </div>
    );
  }

  if (!data || !data.nodes || data.nodes.length === 0) {
    return (
      <p className="py-8 text-center text-text-muted">
        No disease flow data available
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm font-semibold text-text-secondary">
          Disease Flow Diagram
        </h3>
        <p className="mb-2 text-xs text-text-muted">
          Sample distribution across diseases, disease groups, and cohorts.
        </p>
        <SankeyChart
          nodes={data.nodes}
          links={data.links}
          title="Disease Flow: Disease - Disease Group - Cohort"
          height={600}
        />
      </div>
    </div>
  );
}
