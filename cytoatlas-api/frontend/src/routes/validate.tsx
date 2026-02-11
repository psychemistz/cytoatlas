import { lazy, Suspense, useEffect } from 'react';
import { useSearchParams } from 'react-router';
import { useAppStore } from '@/stores/app-store';
import { useValidationStore } from '@/stores/validation-store';
import { useValidationAtlases } from '@/api/hooks/use-validation';
import { TabPanel } from '@/components/ui/tab-panel';
import { SignatureToggle } from '@/components/ui/signature-toggle';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';

const SummaryTab = lazy(() => import('@/components/validate/summary-tab'));
const BulkRnaseqTab = lazy(() => import('@/components/validate/bulk-rnaseq-tab'));
const DonorLevelTab = lazy(() => import('@/components/validate/donor-level-tab'));
const CelltypeLevelTab = lazy(() => import('@/components/validate/celltype-level-tab'));
const SinglecellTab = lazy(() => import('@/components/validate/singlecell-tab'));

const TABS = [
  { id: 'summary', label: 'Summary' },
  { id: 'bulk-rnaseq', label: 'Bulk RNA-seq' },
  { id: 'donor', label: 'Donor Level' },
  { id: 'celltype', label: 'Cell-Type Level' },
  { id: 'singlecell', label: 'Single Cell' },
];

export default function Validate() {
  const [searchParams] = useSearchParams();
  const signatureType = useAppStore((s) => s.signatureType);
  const { selectedAtlas, setSelectedAtlas } = useValidationStore();
  const { data: atlases, isLoading: atlasesLoading } = useValidationAtlases();

  useEffect(() => {
    const atlasParam = searchParams.get('atlas');
    if (atlasParam && !selectedAtlas) {
      setSelectedAtlas(atlasParam);
    }
  }, [searchParams, selectedAtlas, setSelectedAtlas]);

  useEffect(() => {
    if (atlases && atlases.length > 0 && !selectedAtlas) {
      setSelectedAtlas(atlases[0]);
    }
  }, [atlases, selectedAtlas, setSelectedAtlas]);

  const atlasOptions = (atlases || []).map((a) => ({ value: a, label: a }));
  const sigtype = signatureType === 'CytoSig' ? 'cytosig' : 'secact';

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <div className="mb-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">Activity Validation</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Validation of activity predictions by correlating inferred scores with
              signature gene expression across multiple aggregation levels
            </p>
          </div>
          <SignatureToggle />
        </div>

        {atlasesLoading ? (
          <Spinner message="Loading atlases..." />
        ) : (
          <FilterBar>
            <SelectFilter
              label="Atlas"
              options={atlasOptions}
              value={selectedAtlas}
              onChange={setSelectedAtlas}
            />
          </FilterBar>
        )}
      </div>

      <TabPanel tabs={TABS} defaultTab="summary">
        {(tabId) => (
          <ErrorBoundary>
            <Suspense fallback={<Spinner message="Loading tab..." />}>
              {tabId === 'summary' && <SummaryTab sigtype={sigtype} />}
              {tabId === 'bulk-rnaseq' && <BulkRnaseqTab sigtype={sigtype} />}
              {tabId === 'donor' && (
                <DonorLevelTab atlas={selectedAtlas} sigtype={sigtype} />
              )}
              {tabId === 'celltype' && (
                <CelltypeLevelTab atlas={selectedAtlas} sigtype={sigtype} />
              )}
              {tabId === 'singlecell' && (
                <SinglecellTab atlas={selectedAtlas} sigtype={sigtype} />
              )}
            </Suspense>
          </ErrorBoundary>
        )}
      </TabPanel>
    </div>
  );
}
