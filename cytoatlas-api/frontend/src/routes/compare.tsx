import { lazy, Suspense } from 'react';
import { useAppStore } from '@/stores/app-store';
import { TabPanel } from '@/components/ui/tab-panel';
import { SignatureToggle } from '@/components/ui/signature-toggle';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';

const OverviewTab = lazy(() => import('@/components/compare/overview-tab'));
const CelltypeMappingTab = lazy(() => import('@/components/compare/celltype-mapping-tab'));
const AtlasComparisonTab = lazy(() => import('@/components/compare/atlas-comparison-tab'));
const ConservedTab = lazy(() => import('@/components/compare/conserved-tab'));
const MetaAnalysisTab = lazy(() => import('@/components/compare/meta-analysis-tab'));

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'celltype-mapping', label: 'Cell Type Mapping' },
  { id: 'atlas-comparison', label: 'Atlas Comparison' },
  { id: 'conserved', label: 'Conserved Signatures' },
  { id: 'meta-analysis', label: 'Meta-Analysis' },
];

export default function Compare() {
  const signatureType = useAppStore((s) => s.signatureType);

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-8">
      <div className="mb-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">Cross-Atlas Comparison</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Compare cytokine activity patterns across CIMA, Inflammation Atlas, and scAtlas
            </p>
          </div>
          <SignatureToggle />
        </div>
      </div>

      <TabPanel tabs={TABS} defaultTab="overview">
        {(tabId) => (
          <ErrorBoundary>
            <Suspense fallback={<Spinner message="Loading tab..." />}>
              {tabId === 'overview' && <OverviewTab signatureType={signatureType} />}
              {tabId === 'celltype-mapping' && <CelltypeMappingTab />}
              {tabId === 'atlas-comparison' && (
                <AtlasComparisonTab signatureType={signatureType} />
              )}
              {tabId === 'conserved' && <ConservedTab signatureType={signatureType} />}
              {tabId === 'meta-analysis' && (
                <MetaAnalysisTab signatureType={signatureType} />
              )}
            </Suspense>
          </ErrorBoundary>
        )}
      </TabPanel>
    </div>
  );
}
