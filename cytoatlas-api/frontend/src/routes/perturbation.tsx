import { lazy, Suspense } from 'react';
import { useAppStore } from '@/stores/app-store';
import { usePerturbationSummary } from '@/api/hooks/use-perturbation';
import { TabPanel } from '@/components/ui/tab-panel';
import { SignatureToggle } from '@/components/ui/signature-toggle';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';

const CytokineResponseTab = lazy(
  () => import('@/components/perturbation/cytokine-response-tab'),
);
const GroundTruthTab = lazy(
  () => import('@/components/perturbation/ground-truth-tab'),
);
const DrugSensitivityTab = lazy(
  () => import('@/components/perturbation/drug-sensitivity-tab'),
);
const DoseResponseTab = lazy(
  () => import('@/components/perturbation/dose-response-tab'),
);
const PathwayActivationTab = lazy(
  () => import('@/components/perturbation/pathway-activation-tab'),
);

const TABS = [
  { id: 'cytokine-response', label: 'Cytokine Response' },
  { id: 'ground-truth', label: 'Ground Truth' },
  { id: 'drug-sensitivity', label: 'Drug Sensitivity' },
  { id: 'dose-response', label: 'Dose-Response' },
  { id: 'pathway-activation', label: 'Pathway Activation' },
];

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

export default function Perturbation() {
  const signatureType = useAppStore((s) => s.signatureType);
  const { data: summary } = usePerturbationSummary();
  const sigtype = signatureType === 'CytoSig' ? 'cytosig' : 'secact';

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <div className="mb-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">Perturbation Analysis</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Cytokine perturbation responses (parse_10M) and drug sensitivity profiling (Tahoe-100M)
            </p>
          </div>
          <SignatureToggle />
        </div>

        {summary && (
          <div className="flex flex-wrap gap-6 rounded-lg border border-border-light bg-bg-secondary p-4 text-sm">
            <div>
              <span className="font-medium text-text-secondary">parse_10M:</span>{' '}
              <span className="font-semibold">{formatNumber(summary.parse10m.total_cells)}</span> cells,{' '}
              <span className="font-semibold">{summary.parse10m.total_cytokines}</span> cytokines,{' '}
              <span className="font-semibold">{summary.parse10m.total_cell_types}</span> cell types
            </div>
            <div>
              <span className="font-medium text-text-secondary">Tahoe:</span>{' '}
              <span className="font-semibold">{formatNumber(summary.tahoe.total_cells)}</span> cells,{' '}
              <span className="font-semibold">{summary.tahoe.total_drugs}</span> drugs,{' '}
              <span className="font-semibold">{summary.tahoe.total_cell_lines}</span> cell lines,{' '}
              <span className="font-semibold">{summary.tahoe.total_plates}</span> plates
            </div>
          </div>
        )}
      </div>

      <TabPanel tabs={TABS} defaultTab="cytokine-response">
        {(tabId) => (
          <ErrorBoundary>
            <Suspense fallback={<Spinner message="Loading tab..." />}>
              {tabId === 'cytokine-response' && (
                <CytokineResponseTab signatureType={sigtype} />
              )}
              {tabId === 'ground-truth' && (
                <GroundTruthTab signatureType={sigtype} />
              )}
              {tabId === 'drug-sensitivity' && (
                <DrugSensitivityTab signatureType={sigtype} />
              )}
              {tabId === 'dose-response' && <DoseResponseTab />}
              {tabId === 'pathway-activation' && <PathwayActivationTab />}
            </Suspense>
          </ErrorBoundary>
        )}
      </TabPanel>
    </div>
  );
}
