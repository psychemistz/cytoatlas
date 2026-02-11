import { lazy, Suspense } from 'react';
import { useAppStore } from '@/stores/app-store';
import { useSpatialSummary } from '@/api/hooks/use-spatial';
import { TabPanel } from '@/components/ui/tab-panel';
import { SignatureToggle } from '@/components/ui/signature-toggle';
import { StatCard } from '@/components/ui/stat-card';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';

const OverviewTab = lazy(() => import('@/components/spatial/overview-tab'));
const TissueActivityTab = lazy(() => import('@/components/spatial/tissue-activity-tab'));
const TechComparisonTab = lazy(() => import('@/components/spatial/tech-comparison-tab'));
const GeneCoverageTab = lazy(() => import('@/components/spatial/gene-coverage-tab'));
const SpatialMapTab = lazy(() => import('@/components/spatial/spatial-map-tab'));

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'tissue-activity', label: 'Tissue Activity' },
  { id: 'tech-comparison', label: 'Tech Comparison' },
  { id: 'gene-coverage', label: 'Gene Coverage' },
  { id: 'spatial-map', label: 'Spatial Map' },
];

export default function Spatial() {
  const signatureType = useAppStore((s) => s.signatureType);
  const { data: summary } = useSpatialSummary();

  const sigtype = signatureType === 'CytoSig' ? 'cytosig' : 'secact';

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-8">
      <div className="mb-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">Spatial Transcriptomics</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Cytokine and secreted protein activity across 251 spatial datasets from 8 technologies
              (SpatialCorpus-110M, ~110M cells)
            </p>
          </div>
          <SignatureToggle />
        </div>

        {summary && (
          <div className="grid grid-cols-2 gap-4 rounded-lg border border-border-light bg-bg-secondary p-4 sm:grid-cols-4">
            <StatCard
              value={summary.total_cells.toLocaleString()}
              label="Total Cells"
            />
            <StatCard
              value={summary.total_datasets.toLocaleString()}
              label="Datasets"
            />
            <StatCard
              value={summary.technologies.toLocaleString()}
              label="Technologies"
            />
            <StatCard
              value={summary.tissues.toLocaleString()}
              label="Tissues"
            />
          </div>
        )}
      </div>

      <TabPanel tabs={TABS} defaultTab="overview">
        {(tabId) => (
          <ErrorBoundary>
            <Suspense fallback={<Spinner message="Loading tab..." />}>
              {tabId === 'overview' && <OverviewTab signatureType={sigtype} />}
              {tabId === 'tissue-activity' && (
                <TissueActivityTab signatureType={sigtype} />
              )}
              {tabId === 'tech-comparison' && (
                <TechComparisonTab signatureType={sigtype} />
              )}
              {tabId === 'gene-coverage' && (
                <GeneCoverageTab signatureType={sigtype} />
              )}
              {tabId === 'spatial-map' && (
                <SpatialMapTab signatureType={sigtype} />
              )}
            </Suspense>
          </ErrorBoundary>
        )}
      </TabPanel>
    </div>
  );
}
